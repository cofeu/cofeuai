#include "cofeu/model.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cofeu {

namespace {

// LayerNorm: x -> (x - mean) / sqrt(var + eps) * w + b
void layer_norm(const std::vector<float>& x,
                const std::vector<float>& w,
                const std::vector<float>& b,
                std::vector<float>& out) {
    size_t n = x.size();
    float mean = 0.0f;
    for (float v : x) mean += v;
    mean /= static_cast<float>(n);

    float var = 0.0f;
    for (float v : x) {
        float d = v - mean;
        var += d * d;
    }
    var /= static_cast<float>(n);

    float inv_std = 1.0f / std::sqrt(var + 1e-5f);
    out.resize(n);
    for (size_t i = 0; i < n; i++) {
        out[i] = (x[i] - mean) * inv_std * w[i] + b[i];
    }
}

// GELU aktivasyonu (tanh yaklaşımı)
float gelu(float x) {
    return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Linear katman: out = x @ W^T  (PyTorch nn.Linear ile aynı).
// W, (out_features, in_features) şeklinde saklanır.
void linear(const Tensor& x, const Tensor& W, Tensor& out) {
    size_t m = x.rows;       // satır sayısı (T)
    size_t in = x.cols;      // giriş boyutu
    size_t out_f = W.rows;   // çıkış boyutu
    out.resize(m, out_f);
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < out_f; j++) {
            float sum = 0.0f;
            for (size_t t = 0; t < in; t++) {
                sum += x.at(i, t) * W.at(j, t);
            }
            out.at(i, j) = sum;
        }
    }
}

// Softmax (son boyut üzerinde)
void softmax(std::vector<float>& x) {
    float maxv = *std::max_element(x.begin(), x.end());
    float sum = 0.0f;
    for (float& v : x) {
        v = std::exp(v - maxv);
        sum += v;
    }
    for (float& v : x) v /= sum;
}

} // namespace

bool Transformer::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    auto read_int = [&]() -> int {
        int v;
        f.read(reinterpret_cast<char*>(&v), sizeof(int));
        return v;
    };
    auto read_floats = [&](size_t n) -> std::vector<float> {
        std::vector<float> v(n);
        f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
        return v;
    };
    auto read_tensor = [&](size_t r, size_t c) -> Tensor {
        Tensor t(r, c);
        f.read(reinterpret_cast<char*>(t.data.data()), r * c * sizeof(float));
        return t;
    };

    config_.vocab_size = read_int();
    config_.n_embd = read_int();
    config_.n_head = read_int();
    config_.n_layer = read_int();
    config_.block_size = read_int();

    int V = config_.vocab_size;
    int C = config_.n_embd;
    int L = config_.n_layer;

    token_emb_ = read_tensor(V, C);
    pos_emb_ = read_tensor(config_.block_size, C);

    blocks_.resize(L);
    for (auto& blk : blocks_) {
        blk.ln1_w = read_floats(C);
        blk.ln1_b = read_floats(C);
        blk.q_w = read_tensor(C, C);
        blk.k_w = read_tensor(C, C);
        blk.v_w = read_tensor(C, C);
        blk.proj_w = read_tensor(C, C);
        blk.ln2_w = read_floats(C);
        blk.ln2_b = read_floats(C);
        blk.fc1_w = read_tensor(4 * C, C);
        blk.fc1_b = read_floats(4 * C);
        blk.fc2_w = read_tensor(C, 4 * C);
        blk.fc2_b = read_floats(C);
    }

    lnf_w = read_floats(C);
    lnf_b = read_floats(C);
    head_w_ = read_tensor(V, C);

    return f.good() || f.eof();
}

std::vector<float> Transformer::forward(const std::vector<int>& idx,
                                        std::vector<KVCache>* cache) {
    int C = config_.n_embd;
    int H = config_.n_head;
    int head_dim = C / H;
    size_t T = idx.size();

    // Pozisyon ofseti (cache varsa devam eden pozisyonlar)
    size_t pos_offset = 0;
    if (cache != nullptr && !cache->empty()) {
        pos_offset = cache->front().k.cols;
    }

    // x: (T, C)
    Tensor x(T, C);
    for (size_t t = 0; t < T; t++) {
        for (int c = 0; c < C; c++) {
            x.at(t, c) = token_emb_.at(idx[t], c) + pos_emb_.at(pos_offset + t, c);
        }
    }

    // Yeni cache (cache null ise oluştur)
    std::vector<KVCache> new_cache;
    if (cache != nullptr) {
        new_cache.resize(blocks_.size());
    }

    for (size_t bi = 0; bi < blocks_.size(); bi++) {
        const auto& blk = blocks_[bi];

        // --- LayerNorm 1 + Attention ---
        Tensor ln1(T, C);
        for (size_t t = 0; t < T; t++) {
            std::vector<float> row(C);
            for (int c = 0; c < C; c++) row[c] = x.at(t, c);
            std::vector<float> out;
            layer_norm(row, blk.ln1_w, blk.ln1_b, out);
            for (int c = 0; c < C; c++) ln1.at(t, c) = out[c];
        }

        // Q, K, V: (T, C)
        Tensor Q, K, V;
        linear(ln1, blk.q_w, Q);
        linear(ln1, blk.k_w, K);
        linear(ln1, blk.v_w, V);

        // K/V cache ile birleştir
        Tensor K_full, V_full;
        size_t T_total;
        if (cache != nullptr && !cache->empty()) {
            const auto& prev = (*cache)[bi];
            // K_full: (C, T_total) = [prev.k | K^T]
            T_total = prev.k.cols + T;
            K_full.resize(C, T_total);
            V_full.resize(C, T_total);
            for (int c = 0; c < C; c++) {
                for (size_t s = 0; s < prev.k.cols; s++) {
                    K_full.at(c, s) = prev.k.at(c, s);
                    V_full.at(c, s) = prev.v.at(c, s);
                }
                for (size_t t = 0; t < T; t++) {
                    K_full.at(c, prev.k.cols + t) = K.at(t, c);
                    V_full.at(c, prev.k.cols + t) = V.at(t, c);
                }
            }
        } else {
            T_total = T;
            K_full.resize(C, T_total);
            V_full.resize(C, T_total);
            for (int c = 0; c < C; c++) {
                for (size_t t = 0; t < T; t++) {
                    K_full.at(c, t) = K.at(t, c);
                    V_full.at(c, t) = V.at(t, c);
                }
            }
        }

        // Attention: her başlık için ayrı hesapla (OpenMP paralel)
        Tensor attn_out(T, C);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int h = 0; h < H; h++) {
            for (size_t t = 0; t < T; t++) {
                int off = h * head_dim;
                size_t q_pos = pos_offset + t;  // sorgu pozisyonu
                size_t n_keys = q_pos + 1;      // causal: kendi pozisyonuna kadar
                std::vector<float> scores(n_keys);
                for (size_t s = 0; s < n_keys; s++) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        dot += Q.at(t, off + d) * K_full.at(off + d, s);
                    }
                    scores[s] = dot / std::sqrt(static_cast<float>(head_dim));
                }
                softmax(scores);

                for (int d = 0; d < head_dim; d++) {
                    float sum = 0.0f;
                    for (size_t s = 0; s < n_keys; s++) {
                        sum += scores[s] * V_full.at(off + d, s);
                    }
                    attn_out.at(t, off + d) = sum;
                }
            }
        }

        // Projection
        Tensor proj;
        linear(attn_out, blk.proj_w, proj);

        // Residual
        for (size_t t = 0; t < T; t++) {
            for (int c = 0; c < C; c++) {
                x.at(t, c) += proj.at(t, c);
            }
        }

        // --- LayerNorm 2 + MLP ---
        Tensor ln2(T, C);
        for (size_t t = 0; t < T; t++) {
            std::vector<float> row(C);
            for (int c = 0; c < C; c++) row[c] = x.at(t, c);
            std::vector<float> out;
            layer_norm(row, blk.ln2_w, blk.ln2_b, out);
            for (int c = 0; c < C; c++) ln2.at(t, c) = out[c];
        }

        // fc1: (T, 4C)
        Tensor h(T, 4 * C);
        linear(ln2, blk.fc1_w, h);
        for (size_t t = 0; t < T; t++) {
            for (int c = 0; c < 4 * C; c++) {
                h.at(t, c) = gelu(h.at(t, c) + blk.fc1_b[c]);
            }
        }

        // fc2: (T, C)
        Tensor mlp_out;
        linear(h, blk.fc2_w, mlp_out);
        for (size_t t = 0; t < T; t++) {
            for (int c = 0; c < C; c++) {
                x.at(t, c) += mlp_out.at(t, c) + blk.fc2_b[c];
            }
        }

        // Cache'i güncelle
        if (cache != nullptr) {
            new_cache[bi].k = K_full;
            new_cache[bi].v = V_full;
        }
    }

    // Final LayerNorm
    Tensor lnf(T, C);
    for (size_t t = 0; t < T; t++) {
        std::vector<float> row(C);
        for (int c = 0; c < C; c++) row[c] = x.at(t, c);
        std::vector<float> out;
        layer_norm(row, lnf_w, lnf_b, out);
        for (int c = 0; c < C; c++) lnf.at(t, c) = out[c];
    }

    // Head: son pozisyonun logitleri
    std::vector<float> logits(config_.vocab_size);
    size_t last = T - 1;
    for (int v = 0; v < config_.vocab_size; v++) {
        float sum = 0.0f;
        for (int c = 0; c < C; c++) {
            sum += head_w_.at(v, c) * lnf.at(last, c);
        }
        logits[v] = sum;
    }

    if (cache != nullptr) {
        *cache = std::move(new_cache);
    }
    return logits;
}

std::vector<int> Transformer::generate(const std::vector<int>& prompt_ids,
                                       int max_new_tokens,
                                       float temperature,
                                       unsigned seed) {
    std::vector<int> idx = prompt_ids;
    std::mt19937 rng(seed);

    // İlk adım: prompt'u işle, KV cache'i doldur
    std::vector<int> cond;
    int start = std::max(0, static_cast<int>(idx.size()) - config_.block_size);
    cond.assign(idx.begin() + start, idx.end());

    std::vector<KVCache> cache;
    std::vector<float> logits = forward(cond, &cache);

    for (int step = 0; step < max_new_tokens; step++) {
        // Temperature uygula
        int next;
        if (temperature > 0.0f) {
            std::vector<float> probs = logits;
            for (float& l : probs) l /= temperature;
            softmax(probs);

            // Multinomial örnekleme
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            float r = dist(rng);
            float cum = 0.0f;
            next = config_.vocab_size - 1;
            for (int v = 0; v < config_.vocab_size; v++) {
                cum += probs[v];
                if (r <= cum) {
                    next = v;
                    break;
                }
            }
        } else {
            // Greedy
            next = std::max_element(logits.begin(), logits.end()) - logits.begin();
        }
        idx.push_back(next);

        // Sadece yeni token'ı işle, cache'i kullan
        std::vector<int> new_tok = {next};
        logits = forward(new_tok, &cache);
    }
    return idx;
}

} // namespace cofeu

// --- Python (ctypes) için C API ---

extern "C" {

// Modeli cofeu.bin'den yükle. Başarılıysa handle döner.
void* cofeu_model_load(const char* path) {
    auto* model = new cofeu::Transformer();
    if (!model->load(path)) {
        delete model;
        return nullptr;
    }
    return static_cast<void*>(model);
}

void cofeu_model_free(void* handle) {
    delete static_cast<cofeu::Transformer*>(handle);
}

// Prompt token id'lerinden metin üretir. Çıktı token id'lerini out_ids'e yazar.
int cofeu_model_generate(void* handle, const int* prompt_ids, int prompt_len,
                         int max_new_tokens, float temperature, unsigned seed,
                         int* out_ids, int max_out) {
    auto* model = static_cast<cofeu::Transformer*>(handle);
    std::vector<int> prompt(prompt_ids, prompt_ids + prompt_len);
    auto result = model->generate(prompt, max_new_tokens, temperature, seed);
    int n = static_cast<int>(result.size());
    if (n > max_out) n = max_out;
    for (int i = 0; i < n; i++) out_ids[i] = result[i];
    return n;
}

} // extern "C"