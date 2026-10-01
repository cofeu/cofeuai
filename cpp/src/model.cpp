#include "cofeu/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <new>
#include <random>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cofeu {

namespace {

// Paralel döngüye geçmek için gereken asgari iş yükü (çarpma işlemi).
// Altındaki problemlerde thread başlatma + barrier maliyeti hesabı aşar.
constexpr size_t kParallelMinWork = 1 << 17;

// LayerNorm: x -> (x - mean) / sqrt(var + eps) * w + b
// PyTorch nn.LayerNorm ile aynı: popülasyon varyansı, eps=1e-5.
void layer_norm(const float* x, size_t n, const float* w, const float* b, float* out) {
    float mean = 0.0f;
    #pragma omp simd reduction(+:mean)
    for (size_t i = 0; i < n; i++) mean += x[i];
    mean /= static_cast<float>(n);

    float var = 0.0f;
    #pragma omp simd reduction(+:var)
    for (size_t i = 0; i < n; i++) {
        const float d = x[i] - mean;
        var += d * d;
    }
    var /= static_cast<float>(n);

    const float inv_std = 1.0f / std::sqrt(var + 1e-5f);
    #pragma omp simd
    for (size_t i = 0; i < n; i++) out[i] = (x[i] - mean) * inv_std * w[i] + b[i];
}

// GELU — PyTorch'un F.gelu(x) varsayılanı (tam erf formülü).
// Eğitim bu formülle yapıldığı için C++ tarafında da birebir aynı olanı
// kullanıyoruz; tanh yaklaşımı küçük ama birikimli bir kayma yaratıyor.
inline float gelu(float x) {
    return 0.5f * x * (1.0f + std::erf(x * 0.7071067811865476f));
}

// Softmax (yerinde). Toplam sıfır olursa NaN yerine düzgün dağılım döner.
void softmax(float* x, int n) {
    float maxv = -INFINITY;
    for (int i = 0; i < n; i++) maxv = std::max(maxv, x[i]);

    float sum = 0.0f;
    #pragma omp simd reduction(+:sum)
    for (int i = 0; i < n; i++) {
        const float e = std::exp(x[i] - maxv);
        x[i] = e;
        sum += e;
    }
    if (sum <= 0.0f || !std::isfinite(sum)) {
        const float uniform = 1.0f / static_cast<float>(n);
        for (int i = 0; i < n; i++) x[i] = uniform;
        return;
    }
    const float inv = 1.0f / sum;
    #pragma omp simd
    for (int i = 0; i < n; i++) x[i] *= inv;
}

} // namespace

bool Transformer::load(const std::string& path, std::string* error) {
    auto set_error = [error](const std::string& msg) {
        if (error) *error = msg;
        return false;
    };

    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) return set_error("dosya açılamadı: " + path);
    const std::streamoff file_size = f.tellg();
    f.seekg(0, std::ios::beg);

    // Kısa okumaları yakalamak için: her read() sonucunu denetleyeceğiz ve
    // ayrıca dosya boyutunu beklenen boyutla karşılaştıracağız.
    auto read_int = [&](int& v) -> bool {
        return static_cast<bool>(f.read(reinterpret_cast<char*>(&v), sizeof(int)));
    };
    auto read_floats = [&](size_t n, std::vector<float>& v) -> bool {
        v.resize(n);
        if (n == 0) return true;
        return static_cast<bool>(f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float)));
    };
    auto read_tensor = [&](size_t r, size_t c, Tensor& t) -> bool {
        t.resize(r, c);
        if (r * c == 0) return true;
        return static_cast<bool>(f.read(reinterpret_cast<char*>(t.data_ptr()), r * c * sizeof(float)));
    };

    // --- Header ---
    // [char4 "COFE"][int32 version][int32 x6 config]
    // Magic + sürüm, eski (pos_emb'li) ve bozuk dosyaların sessizce yanlış
    // yorumlanmasını engeller; kullanıcı net bir hata mesajı alır.
    char magic[4] = {0, 0, 0, 0};
    int version = 0;
    if (!f.read(magic, 4)) return set_error("başlık okunamadı (dosya çok kısa)");
    if (std::memcmp(magic, kModelMagic, 4) != 0) {
        return set_error("geçersiz dosya imzası (COFE bekleniyordu). Bu dosya "
                         "eski formatta olabilir; python/export.py ile yeniden üretin.");
    }
    if (!read_int(version)) return set_error("sürüm okunamadı");
    if (version != kModelVersion) {
        return set_error("desteklenmeyen sürüm: " + std::to_string(version) +
                         " (bu motor sürüm " + std::to_string(kModelVersion) + " bekliyor)");
    }
    if (!read_int(config_.vocab_size) || !read_int(config_.n_embd) ||
        !read_int(config_.n_head) || !read_int(config_.n_layer) ||
        !read_int(config_.block_size) || !read_int(config_.rope_base)) {
        return set_error("başlık okunamadı (dosya çok kısa veya bozuk)");
    }
    if (config_.rope_base <= 1) return set_error("rope_base 1'den büyük olmalı");

    const int V = config_.vocab_size;
    const int C = config_.n_embd;
    const int H = config_.n_head;
    const int L = config_.n_layer;
    const int B = config_.block_size;

    if (V <= 0 || C <= 0 || L <= 0 || B <= 0) {
        return set_error("geçersiz başlık: vocab=" + std::to_string(V) + " n_embd=" +
                         std::to_string(C) + " n_layer=" + std::to_string(L) +
                         " block_size=" + std::to_string(B));
    }
    // head_dim = C / H -> H == 0 bölme hatası, bölünmeyen C sessiz yanlış matematik
    if (H <= 0) return set_error("n_head sıfır olamaz");
    if (C % H != 0) {
        return set_error("n_embd (" + std::to_string(C) + ") n_head (" + std::to_string(H) +
                         ") ile bölünmüyor");
    }
    head_dim_ = C / H;
    if (head_dim_ % 2 != 0) {
        return set_error("head_dim (" + std::to_string(head_dim_) + ") çift olmalı (RoPE)");
    }

    // Beklenen dosya boyutu: kModelHeaderBytes + tüm ağırlıklar (float32).
    // Katman başına: 12*C^2 + 9*C float (4 dikkat + fc1 + fc2 kare, kalan lineer)
    const long long per_layer = 12LL * C * C + 9LL * C;
    const long long total_floats = 2LL * V * C + L * per_layer + 2LL * C;
    const long long expected =
        static_cast<long long>(kModelHeaderBytes) + total_floats * 4LL;
    if (static_cast<long long>(file_size) != expected) {
        return set_error("dosya boyutu beklenenden farklı: " +
                         std::to_string(static_cast<long long>(file_size)) + " (beklenen " +
                         std::to_string(expected) + "). Dosya bozuk ya da farklı bir mimariyle üretilmiş.");
    }

    // --- Ağırlıklar ---
    if (!read_tensor(V, C, token_emb_)) return set_error("token_emb okunamadı");

    blocks_.resize(L);
    for (int i = 0; i < L; i++) {
        auto& blk = blocks_[static_cast<size_t>(i)];
        if (!read_floats(C, blk.ln1_w) || !read_floats(C, blk.ln1_b) ||
            !read_tensor(C, C, blk.q_w) || !read_tensor(C, C, blk.k_w) ||
            !read_tensor(C, C, blk.v_w) || !read_tensor(C, C, blk.proj_w) ||
            !read_floats(C, blk.ln2_w) || !read_floats(C, blk.ln2_b) ||
            !read_tensor(4 * C, C, blk.fc1_w) || !read_floats(4 * C, blk.fc1_b) ||
            !read_tensor(C, 4 * C, blk.fc2_w) || !read_floats(C, blk.fc2_b)) {
            return set_error("blok " + std::to_string(i) + " ağırlıkları okunamadı");
        }
    }

    if (!read_floats(C, lnf_w) || !read_floats(C, lnf_b)) return set_error("ln_f okunamadı");
    if (!read_tensor(V, C, head_w_)) return set_error("head okunamadı");

    init_rope();
    init_workspace(static_cast<size_t>(B));
    logits_.assign(static_cast<size_t>(V), 0.0f);
    probs_.assign(static_cast<size_t>(V), 0.0f);

    if (error) error->clear();
    return true;
}

void Transformer::init_rope() {
    // RoPE frekansları: theta_i = base^(-2i/head_dim), i in [0, head_dim/2)
    const int half = head_dim_ / 2;
    // Pencere kaydırılsa da pozisyonlar mutlak artar; pencere + üretim payı
    // yeterli olsun diye makul bir tavan kullanıyoruz.
    rope_max_pos_ = config_.block_size * 16;
    rope_cos_.resize(static_cast<size_t>(rope_max_pos_) * half);
    rope_sin_.resize(static_cast<size_t>(rope_max_pos_) * half);

    const double base = static_cast<double>(config_.rope_base);
    for (int p = 0; p < rope_max_pos_; p++) {
        for (int i = 0; i < half; i++) {
            const double inv_freq = std::pow(base, -2.0 * i / head_dim_);
            const double angle = static_cast<double>(p) * inv_freq;
            rope_cos_[static_cast<size_t>(p) * half + i] = static_cast<float>(std::cos(angle));
            rope_sin_[static_cast<size_t>(p) * half + i] = static_cast<float>(std::sin(angle));
        }
    }
}

void Transformer::init_workspace(size_t max_batch) {
    const size_t C = static_cast<size_t>(config_.n_embd);
    ws_.x.resize_uninit(max_batch, C);
    ws_.ln1.resize_uninit(max_batch, C);
    ws_.ln2.resize_uninit(max_batch, C);
    ws_.q.resize_uninit(max_batch, C);
    ws_.k.resize_uninit(max_batch, C);
    ws_.v.resize_uninit(max_batch, C);
    ws_.attn.resize_uninit(max_batch, C);
    ws_.proj.resize_uninit(max_batch, C);
    ws_.h.resize_uninit(max_batch, 4 * C);
    ws_.mlp.resize_uninit(max_batch, C);
    ws_.row.resize(C);
    ws_.row_out.resize(C);
    ws_.scores.resize(static_cast<size_t>(config_.block_size) + 1);
}

void Transformer::reset_cache(std::vector<KVCache>& cache, size_t cap) const {
    cache.resize(static_cast<size_t>(config_.n_layer));
    for (auto& c : cache) {
        c.k.resize_uninit(cap, static_cast<size_t>(config_.n_embd));
        c.v.resize_uninit(cap, static_cast<size_t>(config_.n_embd));
        c.head = 0;
        c.len = 0;
        c.cap = cap;
    }
}

bool Transformer::forward(const std::vector<int>& idx,
                          std::vector<KVCache>* cache,
                          int pos_base,
                          std::vector<float>& out_logits,
                          std::string* error) {
    const int C = config_.n_embd;
    const int hd = head_dim_;
    const int half = hd / 2;
    const size_t T = idx.size();

    // Boş dizi: önceki sürümde T=0 iken "son pozisyon" hesabı SIZE_MAX'a
    // taşıyor ve bellek dışı okuma ile segfault üretiyordu.
    if (T == 0) {
        if (error) *error = "forward: boş token dizisi";
        return false;
    }
    if (pos_base < 0 || pos_base + static_cast<int>(T) > rope_max_pos_) {
        if (error) *error = "pozisyon sınırı aşıldı (max " + std::to_string(rope_max_pos_) + ")";
        return false;
    }

    // Token id doğrulaması: vocab ile model uyuşmazlığı ya da bozuk girdide
    // token_emb_ tablosunun dışına çıkılmasını engeller.
    for (size_t t = 0; t < T; t++) {
        if (idx[t] < 0 || idx[t] >= config_.vocab_size) {
            if (error) {
                *error = "token id " + std::to_string(idx[t]) + " vocab sınırı dışında [0, " +
                         std::to_string(config_.vocab_size) + ")";
            }
            return false;
        }
    }

    const size_t cap = cache ? cache->front().cap : 0;
    if (cache && T > cap) {
        if (error) *error = "token sayısı bağlam penceresinden büyük";
        return false;
    }

    // --- Embedding ---
    {
        Tensor& x = ws_.x;
        x.resize_uninit(T, static_cast<size_t>(C));
        const float* tok_data = token_emb_.data_ptr();
        const size_t tok_stride = static_cast<size_t>(C);
        #pragma omp parallel for schedule(static) if (T * C >= kParallelMinWork)
        for (int t = 0; t < static_cast<int>(T); t++) {
            const float* row = tok_data + static_cast<size_t>(idx[t]) * tok_stride;
            std::memcpy(x.row(static_cast<size_t>(t)), row, sizeof(float) * C);
        }
    }

    // --- Katmanlar ---
    for (size_t bi = 0; bi < blocks_.size(); bi++) {
        auto& blk = blocks_[bi];
        const bool use_cache = cache != nullptr;

        // LayerNorm 1 (satır satır)
        layer_norm_batch(ws_.x, ws_.ln1, blk.ln1_w, blk.ln1_b, T, C);

        // Q, K, V
        linear(ws_.ln1, blk.q_w, ws_.q);
        linear(ws_.ln1, blk.k_w, ws_.k);
        linear(ws_.ln1, blk.v_w, ws_.v);

        // RoPE: Q ve K'ye konum bilgisi enjekte edilir. K'ye uygulandığı için
        // cache'ten eski girişleri düşürmek pozisyon tutarlılığını bozmaz.
        apply_rope(ws_.q, pos_base, T, C, hd, half, rope_cos_, rope_sin_);
        apply_rope(ws_.k, pos_base, T, C, hd, half, rope_cos_, rope_sin_);

        // KV cache'i güncelle
        size_t n_keys_base = 0;
        if (use_cache) {
            KVCache& kv = (*cache)[bi];
            const size_t cap_kv = kv.cap;

            // Yeni satırları halka tamponun sonuna yaz
            for (size_t t = 0; t < T; t++) {
                size_t slot = kv.head + kv.len;
                if (slot >= cap_kv) slot -= cap_kv;
                std::memcpy(kv.k.row(slot), ws_.k.row(t), sizeof(float) * C);
                std::memcpy(kv.v.row(slot), ws_.v.row(t), sizeof(float) * C);
                kv.len++;
                if (kv.len > cap_kv) {
                    // Pencere doldu: en eski girişi düşür
                    kv.head = (kv.head + 1) % cap_kv;
                    kv.len = cap_kv;
                }
            }

            // Sorgunun görmesi gereken anahtar sayısı: kendisi dahil, kendinden
            // önceki tüm geçerli girişler.
            n_keys_base = kv.len - T;
        }

        // Attention
        Tensor& attn = ws_.attn;
        attn.resize_uninit(T, static_cast<size_t>(C));
        if (use_cache) {
            const KVCache& kv = (*cache)[bi];
            attention_cached(attn, ws_.q, kv, T, C, hd, n_keys_base);
        } else {
            attention_prefill(attn, ws_.q, ws_.k, ws_.v, T, C, hd);
        }

        // Projeksiyon + residual
        linear(attn, blk.proj_w, ws_.proj);
        #pragma omp parallel for schedule(static) if (T * C >= kParallelMinWork)
        for (size_t t = 0; t < T; t++) {
            float* xr = ws_.x.row(t);
            const float* pr = ws_.proj.row(t);
            #pragma omp simd
            for (int c = 0; c < C; c++) xr[c] += pr[c];
        }

        // LayerNorm 2 + MLP
        layer_norm_batch(ws_.x, ws_.ln2, blk.ln2_w, blk.ln2_b, T, C);
        linear(ws_.ln2, blk.fc1_w, ws_.h);

        #pragma omp parallel for schedule(static) if (T * 4 * C >= kParallelMinWork)
        for (size_t t = 0; t < T; t++) {
            float* hr = ws_.h.row(t);
            #pragma omp simd
            for (int c = 0; c < 4 * C; c++) hr[c] = gelu(hr[c] + blk.fc1_b[static_cast<size_t>(c)]);
        }

        linear(ws_.h, blk.fc2_w, ws_.mlp);

        #pragma omp parallel for schedule(static) if (T * C >= kParallelMinWork)
        for (size_t t = 0; t < T; t++) {
            float* xr = ws_.x.row(t);
            const float* mr = ws_.mlp.row(t);
            #pragma omp simd
            for (int c = 0; c < C; c++) xr[c] += mr[c] + blk.fc2_b[static_cast<size_t>(c)];
        }
    }

    // --- Final LayerNorm + Head (yalnızca son pozisyon) ---
    Tensor& lnf = ws_.ln2;  // ln2 tamponu artık serbest
    layer_norm_row(ws_.x, lnf, lnf_w, lnf_b, T - 1, C);

    const float* last = lnf.data_ptr();
    const float* hw = head_w_.data_ptr();
    out_logits.resize(static_cast<size_t>(config_.vocab_size));

    const bool parallel_head = C >= 512 && config_.vocab_size >= 1024;
    #pragma omp parallel for schedule(static) if (parallel_head)
    for (int v = 0; v < config_.vocab_size; v++) {
        const float* w = hw + static_cast<size_t>(v) * C;
        float sum = 0.0f;
        #pragma omp simd reduction(+:sum)
        for (int c = 0; c < C; c++) sum += w[c] * last[c];
        out_logits[static_cast<size_t>(v)] = sum;
    }

    return true;
}

void Transformer::layer_norm_batch(const Tensor& src, Tensor& dst,
                                  const std::vector<float>& w, const std::vector<float>& b,
                                  size_t T, int C) {
    dst.resize_uninit(T, static_cast<size_t>(C));
    #pragma omp parallel for schedule(static) if (T * C >= kParallelMinWork)
    for (int t = 0; t < static_cast<int>(T); t++) {
        layer_norm(src.row(static_cast<size_t>(t)), static_cast<size_t>(C), w.data(), b.data(),
                   dst.row(static_cast<size_t>(t)));
    }
}

void Transformer::layer_norm_row(const Tensor& src, Tensor& dst,
                                 const std::vector<float>& w, const std::vector<float>& b,
                                 size_t row, int C) {
    dst.resize_uninit(1, static_cast<size_t>(C));
    layer_norm(src.row(row), static_cast<size_t>(C), w.data(), b.data(), dst.data_ptr());
}

void Transformer::apply_rope(Tensor& x, int pos_base, size_t T, int C, int hd, int half,
                             const std::vector<float>& cos_t, const std::vector<float>& sin_t) {
    // İki komşu kanal çifti üzerinde dönme (interleaved / GPT-J düzeni)
    #pragma omp parallel for collapse(2) schedule(static) if (T * C >= kParallelMinWork)
    for (int h = 0; h < static_cast<int>(T); h++) {
        for (int off = 0; off < C; off += hd) {
            float* row = x.row(static_cast<size_t>(h));
            const float* cs = cos_t.data() + static_cast<size_t>(pos_base + h) * half;
            const float* sn = sin_t.data() + static_cast<size_t>(pos_base + h) * half;
            #pragma omp simd
            for (int i = 0; i < half; i++) {
                const float x0 = row[off + 2 * i];
                const float x1 = row[off + 2 * i + 1];
                row[off + 2 * i] = x0 * cs[i] - x1 * sn[i];
                row[off + 2 * i + 1] = x0 * sn[i] + x1 * cs[i];
            }
        }
    }
}

void Transformer::attention_cached(Tensor& out, const Tensor& q, const KVCache& kv,
                                   size_t T, int C, int hd, size_t n_keys_base) {
    const size_t cap = kv.cap;
    const float* kdata = kv.k.data_ptr();
    const float* vdata = kv.v.data_ptr();
    const size_t stride = static_cast<size_t>(C);
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));

    #pragma omp parallel
    {
        float* s_local = new float[cap + 1];
        #pragma omp for collapse(2) schedule(static)
        for (int h = 0; h < static_cast<int>(C / hd); h++) {
            for (int t = 0; t < static_cast<int>(T); t++) {
                const int off = h * hd;
                const size_t n_keys = n_keys_base + static_cast<size_t>(t) + 1;
                const float* qr = q.row(static_cast<size_t>(t)) + off;

                // QK^T: her anahtar için satır ardışık erişim
                for (size_t s = 0; s < n_keys; s++) {
                    size_t slot = kv.head + s;
                    if (slot >= cap) slot -= cap;
                    const float* kr = kdata + slot * stride + off;
                    float dot = 0.0f;
                    #pragma omp simd reduction(+:dot)
                    for (int d = 0; d < hd; d++) dot += qr[d] * kr[d];
                    s_local[s] = dot * inv_sqrt;
                }

                softmax(s_local, static_cast<int>(n_keys));

                // P·V
                float* orow = out.row(static_cast<size_t>(t)) + off;
                for (int d = 0; d < hd; d++) orow[d] = 0.0f;
                for (size_t s = 0; s < n_keys; s++) {
                    size_t slot = kv.head + s;
                    if (slot >= cap) slot -= cap;
                    const float* vr = vdata + slot * stride + off;
                    const float p = s_local[s];
                    #pragma omp simd
                    for (int d = 0; d < hd; d++) orow[d] += p * vr[d];
                }
            }
        }
        delete[] s_local;
    }
}

void Transformer::attention_prefill(Tensor& out, const Tensor& q, const Tensor& k,
                                    const Tensor& v, size_t T, int C, int hd) {
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));

    #pragma omp parallel
    {
        std::vector<float> s_local(T + 1);
        #pragma omp for collapse(2) schedule(static)
        for (int h = 0; h < static_cast<int>(C / hd); h++) {
            for (int t = 0; t < static_cast<int>(T); t++) {
                const int off = h * hd;
                const size_t n_keys = static_cast<size_t>(t) + 1;  // nedensellik
                const float* qr = q.row(static_cast<size_t>(t)) + off;

                for (size_t s = 0; s < n_keys; s++) {
                    const float* kr = k.row(s) + off;
                    float dot = 0.0f;
                    #pragma omp simd reduction(+:dot)
                    for (int d = 0; d < hd; d++) dot += qr[d] * kr[d];
                    s_local[s] = dot * inv_sqrt;
                }

                softmax(s_local.data(), static_cast<int>(n_keys));

                float* orow = out.row(static_cast<size_t>(t)) + off;
                for (int d = 0; d < hd; d++) orow[d] = 0.0f;
                for (size_t s = 0; s < n_keys; s++) {
                    const float* vr = v.row(s) + off;
                    const float p = s_local[s];
                    #pragma omp simd
                    for (int d = 0; d < hd; d++) orow[d] += p * vr[d];
                }
            }
        }
    }
}

void Transformer::linear(const Tensor& x, const Tensor& W, Tensor& out) {
    const size_t m = x.rows;
    const size_t in = x.cols;
    const size_t out_f = W.rows;
    out.resize_uninit(m, out_f);

    const float* xd = x.data_ptr();
    const float* wd = W.data_ptr();
    float* od = out.data_ptr();

    if (m == 1) {
        // Tek token (üretim): vektör-matris çarpımı
        const float* xr = xd;
        #pragma omp parallel for schedule(static) if (out_f * in >= kParallelMinWork)
        for (size_t j = 0; j < out_f; j++) {
            const float* wr = wd + j * in;
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (size_t t = 0; t < in; t++) sum += xr[t] * wr[t];
            od[j] = sum;
        }
        return;
    }

    #pragma omp parallel for collapse(2) schedule(static) if (m * out_f * in >= kParallelMinWork)
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < out_f; j++) {
            const float* xr = xd + i * in;
            const float* wr = wd + j * in;
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (size_t t = 0; t < in; t++) sum += xr[t] * wr[t];
            od[i * out_f + j] = sum;
        }
    }
}

bool Transformer::forwardPublic(const std::vector<int>& token_ids,
                                std::vector<float>* out_logits,
                                std::string* error) {
    if (token_ids.empty()) {
        if (error) *error = "token dizisi boş";
        return false;
    }
    // Uzun promptlarda yalnızca son block_size token işlenir (sliding window).
    // Sınır kontrolü ŞART: begin()'den öncesine gidilirse (end()-block_size)
    // tanımsız davranış olur ve bozuk token id'leri işlenir.
    const size_t n = token_ids.size();
    const size_t bsz = static_cast<size_t>(config_.block_size);
    const size_t start = n > bsz ? n - bsz : 0;
    std::vector<int> window(token_ids.begin() + static_cast<std::ptrdiff_t>(start),
                            token_ids.end());
    std::vector<float> logits;
    std::string local_err;
    // pos_base = 0: pencere başı mutlak konum 0 kabul edilir (Python ile aynı).
    if (!forward(window, nullptr, 0, logits, &local_err)) {
        if (error) *error = local_err;
        return false;
    }
    if (out_logits) out_logits->swap(logits);
    if (error) error->clear();
    return true;
}

bool Transformer::generate(const std::vector<int>& prompt_ids,
                           int max_new_tokens,
                           const SamplingConfig& sampling,
                           std::vector<int>& out_ids,
                           TokenCallback callback,
                           void* user_data) {
    std::string err;
    out_ids.clear();

    if (prompt_ids.empty()) {
        err = "prompt boş (tokenize edilebilir karakter yok)";
        return false;
    }
    if (max_new_tokens <= 0) {
        // Token üretmeden prompt'u döndür
        out_ids = prompt_ids;
        return true;
    }
    for (int id : prompt_ids) {
        if (id < 0 || id >= config_.vocab_size) {
            err = "prompt'ta geçersiz token id " + std::to_string(id);
            return false;
        }
    }

    const int V = config_.vocab_size;
    const int B = config_.block_size;

    // Uzun prompt'ları pencerenin sonuna kırp
    std::vector<int> cond = prompt_ids;
    int prompt_start = static_cast<int>(cond.size()) - B;
    if (prompt_start > 0) cond.erase(cond.begin(), cond.begin() + prompt_start);

    std::vector<KVCache> cache;
    reset_cache(cache, static_cast<size_t>(B));

    // Prefill
    std::vector<float> logits;
    if (!forward(cond, &cache, 0, logits, &err)) return false;

    // RNG: seed == 0 ise gerçek rastgelelik
    std::mt19937 rng;
    if (sampling.seed == 0) {
        std::random_device rd;
        rng.seed(rd());
    } else {
        rng.seed(sampling.seed);
    }
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    const float temp = sampling.temperature > 0.0f ? sampling.temperature : 1.0f;
    const bool greedy = sampling.temperature <= 0.0f;

    // Tekrar cezası için görülen token işareti
    std::vector<char> seen(static_cast<size_t>(V), 0);
    std::vector<int> seen_list;
    seen_list.reserve(cond.size() + static_cast<size_t>(max_new_tokens));
    for (int id : cond) {
        if (!seen[static_cast<size_t>(id)]) {
            seen[static_cast<size_t>(id)] = 1;
            seen_list.push_back(id);
        }
    }

    out_ids = prompt_ids;
    int pos = static_cast<int>(cond.size());
    std::vector<float> probs(static_cast<size_t>(V));
    std::vector<int> top_idx;
    std::vector<float> top_val;

    for (int step = 0; step < max_new_tokens; step++) {
        // --- Örnekleme ---
        std::vector<float> scores = logits;

        // 1) Tekrar cezası (HF sırası: önce ceza, sonra sıcaklık)
        if (sampling.repetition_penalty != 1.0f) {
            const float rp = sampling.repetition_penalty;
            for (int id : seen_list) {
                float& v = scores[static_cast<size_t>(id)];
                v = v > 0.0f ? v / rp : v * rp;
            }
        }

        int next = -1;
        if (greedy) {
            next = static_cast<int>(std::max_element(scores.begin(), scores.end()) - scores.begin());
        } else {
            // 2) Sıcaklık
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < V; i++) scores[static_cast<size_t>(i)] /= temp;

            // 3) Top-k
            if (sampling.top_k > 0 && sampling.top_k < V) {
                const int k = sampling.top_k;
                top_idx.resize(V);
                for (int i = 0; i < V; i++) top_idx[static_cast<size_t>(i)] = i;
                std::partial_sort(top_idx.begin(), top_idx.begin() + k, top_idx.end(),
                                  [&](int a, int b) {
                                      return scores[static_cast<size_t>(a)] > scores[static_cast<size_t>(b)];
                                  });
                const float thresh = scores[static_cast<size_t>(top_idx[static_cast<size_t>(k - 1)])];
                for (int i = 0; i < V; i++) {
                    if (scores[static_cast<size_t>(i)] < thresh) {
                        scores[static_cast<size_t>(i)] = -INFINITY;
                    }
                }
            }

            // 4) Top-p (nucleus)
            if (sampling.top_p >= 0.0f && sampling.top_p < 1.0f) {
                // HF TopPLogitsWarper ile birebir aynı: kümülatif toplam
                // NORMALİZE OLASILIKLAR üzerinden alınır, ham exp(logit)
                // üzerinden değil. Aksi halde nucleus tek token'a düşer.
                float max_s = -INFINITY;
                for (int i = 0; i < V; i++) {
                    max_s = std::max(max_s, scores[static_cast<size_t>(i)]);
                }
                std::vector<float> expv(static_cast<size_t>(V), 0.0f);
                double denom = 0.0;
                for (int i = 0; i < V; i++) {
                    if (!std::isfinite(scores[static_cast<size_t>(i)])) continue;
                    expv[static_cast<size_t>(i)] = std::exp(scores[static_cast<size_t>(i)] - max_s);
                    denom += expv[static_cast<size_t>(i)];
                }

                top_idx.resize(V);
                for (int i = 0; i < V; i++) top_idx[static_cast<size_t>(i)] = i;
                std::sort(top_idx.begin(), top_idx.end(), [&](int a, int b) {
                    return scores[static_cast<size_t>(a)] > scores[static_cast<size_t>(b)];
                });

                // HF TopPLogitsWarper birebir:
                //   remove_sorted = (cum - probs) > top_p   ->  cum[i-1] > top_p
                //   ardından maske BİR POZİSYON kaydırılır  ->  rm[i] = rm_orig[i-1]
                // Yani rm[i] = cum[i-2] > top_p (i >= 2), ve rm[0] = rm[1] = False:
                // en olası İKİ token her zaman korunur. nucleus, kümülatif toplamı
                // >= top_p olan en küçük kümedir.
                double cum_m2 = 0.0;
                double cum_m1 = 0.0;
                for (int rank = 0; rank < V; rank++) {
                    if (rank >= 2 && cum_m2 > sampling.top_p) {
                        for (int r = rank; r < V; r++) {
                            scores[static_cast<size_t>(top_idx[static_cast<size_t>(r)])] = -INFINITY;
                        }
                        break;
                    }
                    const int id = top_idx[static_cast<size_t>(rank)];
                    cum_m2 = cum_m1;
                    cum_m1 += denom > 0.0 ? expv[static_cast<size_t>(id)] / denom : 0.0;
                }
            }

            // 5) Softmax + ters CDF örneklemesi
            softmax(scores.data(), V);
            const float r = dist(rng);
            float cum = 0.0f;
            next = V - 1;
            for (int i = 0; i < V; i++) {
                cum += scores[static_cast<size_t>(i)];
                if (r < cum) {
                    next = i;
                    break;
                }
            }
            if (!std::isfinite(scores[static_cast<size_t>(next)]) || next < 0) {
                next = static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
            }
        }

        if (next < 0) next = 0;
        if (sampling.eos_token_id >= 0 && next == sampling.eos_token_id) break;

        out_ids.push_back(next);
        if (!seen[static_cast<size_t>(next)]) {
            seen[static_cast<size_t>(next)] = 1;
            seen_list.push_back(next);
        }
        if (callback) callback(next, user_data);

        // Son adımda bir ileri geçiş daha yapmaya gerek yok
        if (step + 1 >= max_new_tokens) break;

        std::vector<int> next_tok(1, next);
        if (!forward(next_tok, &cache, pos, logits, &err)) return false;
        pos++;
    }

    return true;
}

} // namespace cofeu

// --- Python (ctypes) için C API ---

extern "C" {

typedef void (*cofeu_token_cb)(int token_id, void* user_data);

static void set_error(char* out, int cap, const std::string& msg) {
    if (out && cap > 0) {
        std::snprintf(out, static_cast<size_t>(cap), "%s", msg.c_str());
    }
}

// Modeli cofeu.bin'den yükle. Başarılıysa handle döner.
void* cofeu_model_load(const char* path, char* error_out, int error_buf) {
    if (!path) {
        set_error(error_out, error_buf, "yol nullptr");
        return nullptr;
    }
    // C++ istisnaları ABI sınırından geçmemeli
    try {
        auto* model = new cofeu::Transformer();
        std::string err;
        if (!model->load(path, &err)) {
            delete model;
            set_error(error_out, error_buf, err);
            return nullptr;
        }
        return static_cast<void*>(model);
    } catch (const std::exception& e) {
        set_error(error_out, error_buf, std::string("yükleme hatası: ") + e.what());
        return nullptr;
    } catch (...) {
        set_error(error_out, error_buf, "bilinmeyen yükleme hatası");
        return nullptr;
    }
}

void cofeu_model_free(void* handle) {
    delete static_cast<cofeu::Transformer*>(handle);
}

int cofeu_model_vocab_size(void* handle) {
    if (!handle) return 0;
    return static_cast<cofeu::Transformer*>(handle)->config().vocab_size;
}

int cofeu_model_block_size(void* handle) {
    if (!handle) return 0;
    return static_cast<cofeu::Transformer*>(handle)->config().block_size;
}

int cofeu_model_max_position(void* handle) {
    if (!handle) return 0;
    return static_cast<cofeu::Transformer*>(handle)->maxPosition();
}

// Prompt token id'lerinden metin üretir. out_ids'e prompt + üretilen
// token'lar yazılır, toplam token sayısı döner.
// cb verilirse her üretilen token için anında çağrılır.
// Hata: -1 döner, error_out doldurulur.
int cofeu_model_generate(void* handle, const int* prompt_ids, int prompt_len,
                         int max_new_tokens, float temperature, unsigned seed,
                         int top_k, float top_p, float repetition_penalty, int eos_token_id,
                         cofeu_token_cb cb, void* user_data,
                         int* out_ids, int max_out,
                         char* error_out, int error_buf) {
    if (!handle) {
        set_error(error_out, error_buf, "model handle nullptr");
        return -1;
    }
    if (!out_ids || max_out <= 0) {
        set_error(error_out, error_buf, "çıktı buffer'ı geçersiz");
        return -1;
    }
    if (prompt_len < 0 || (!prompt_ids && prompt_len > 0)) {
        set_error(error_out, error_buf, "prompt dizisi geçersiz");
        return -1;
    }

    try {
        auto* model = static_cast<cofeu::Transformer*>(handle);
        std::vector<int> prompt;
        if (prompt_len > 0) prompt.assign(prompt_ids, prompt_ids + prompt_len);

        cofeu::SamplingConfig sc;
        sc.temperature = temperature;
        sc.seed = seed;
        sc.top_k = top_k;
        sc.top_p = top_p;
        sc.repetition_penalty = repetition_penalty;
        sc.eos_token_id = eos_token_id;

        std::vector<int> result;
        if (!model->generate(prompt, max_new_tokens, sc, result, cb, user_data)) {
            set_error(error_out, error_buf, "üretim başarısız");
            return -1;
        }

        int n = static_cast<int>(result.size());
        if (n > max_out) n = max_out;
        for (int i = 0; i < n; i++) out_ids[i] = result[i];
        set_error(error_out, error_buf, "");
        return n;
    } catch (const std::exception& e) {
        set_error(error_out, error_buf, std::string("üretim hatası: ") + e.what());
        return -1;
    } catch (...) {
        set_error(error_out, error_buf, "bilinmeyen üretim hatası");
        return -1;
    }
}

// Model + tokenizer uyumluluğunu denetler (vocab boyutu eşleşmeli).
int cofeu_model_check_vocab(void* handle, int tokenizer_vocab_size, char* error_out, int error_buf) {
    if (!handle) {
        set_error(error_out, error_buf, "model handle nullptr");
        return 0;
    }
    const int model_vocab = static_cast<cofeu::Transformer*>(handle)->config().vocab_size;
    if (tokenizer_vocab_size != model_vocab) {
        set_error(error_out, error_buf,
                  "vocab uyuşmazlığı: tokenizer " + std::to_string(tokenizer_vocab_size) +
                      " token, model " + std::to_string(model_vocab) + " bekliyor. "
                      "Modeli export.py ile yeniden dışa aktarın.");
        return 0;
    }
    set_error(error_out, error_buf, "");
    return 1;
}

// Hata ayıklama / doğrulama: verilen token dizisinin SON pozisyonuna ait
// logitleri yazar. PyTorch ile karşılaştırmak için kullanılır.
// Dönüş: yazılan float sayısı veya -1 (hata_out doldurulur).
int cofeu_model_logits(void* handle, const int* token_ids, int n_tokens,
                       float* out_logits, int max_out,
                       char* error_out, int error_buf) {
    if (!handle) {
        set_error(error_out, error_buf, "model handle nullptr");
        return -1;
    }
    if (n_tokens <= 0 || (!token_ids && n_tokens > 0)) {
        set_error(error_out, error_buf, "token dizisi geçersiz");
        return -1;
    }
    if (!out_logits || max_out <= 0) {
        set_error(error_out, error_buf, "çıktı buffer'ı geçersiz");
        return -1;
    }
    try {
        auto* model = static_cast<cofeu::Transformer*>(handle);
        std::vector<int> ids(token_ids, token_ids + n_tokens);
        std::string err;
        std::vector<float> logits;
        if (!model->forwardPublic(ids, &logits, &err)) {
            set_error(error_out, error_buf, err.empty() ? "ileri geçiş başarısız" : err);
            return -1;
        }
        const int want = static_cast<int>(logits.size());
        const int n = want < max_out ? want : max_out;
        for (int i = 0; i < n; i++) out_logits[i] = logits[static_cast<size_t>(i)];
        if (n < want) {
            set_error(error_out, error_buf,
                      "çıktı buffer'ı küçük: " + std::to_string(want) + " gerekli");
            return -1;
        }
        set_error(error_out, error_buf, "");
        return n;
    } catch (const std::exception& e) {
        set_error(error_out, error_buf, std::string("ileri geçiş hatası: ") + e.what());
        return -1;
    } catch (...) {
        set_error(error_out, error_buf, "bilinmeyen ileri geçiş hatası");
        return -1;
    }
}

const char* cofeu_version(void) {
    return "cofeu 2.0 (rope)";
}

} // extern "C"