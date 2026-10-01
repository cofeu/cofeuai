#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tensor.hpp"

namespace cofeu {

// Binary format tanımı — python/export.py ile birebir aynı olmalıdır.
//   [char4 "COFE"][int32 version][int32 x6 config][float32 x N ağırlık]
constexpr char kModelMagic[4] = {'C', 'O', 'F', 'E'};
constexpr int kModelVersion = 2;
constexpr size_t kModelHeaderBytes = 4 + 4 + 6 * 4;  // 32 bayt

struct ModelConfig {
    int vocab_size = 0;
    int n_embd = 0;
    int n_head = 0;
    int n_layer = 0;
    int block_size = 0;   // bağlam penceresi (sliding window)
    int rope_base = 10000;
};

// Üretim sırasında örnekleme parametreleri.
struct SamplingConfig {
    float temperature = 1.0f;
    int top_k = 0;              // 0 = kapalı
    float top_p = -1.0f;        // <0 = kapalı
    float repetition_penalty = 1.0f;
    int eos_token_id = -1;
    unsigned seed = 0;          // 0 = rastgele
};

// C++ ile saf transformer inference motoru.
// PyTorch bağımlılığı yoktur; ağırlıklar export.py ile üretilen
// cofeu.bin dosyasından yüklenir.
//
// Pozisyon kodlaması RoPE'dir (öğrenilmiş mutlak pozisyon tablosu yok),
// bu yüzden bağlam penceresi kaydırıldığında pozisyonlar tutarlı kalır ve
// pencere taşması diye bir durum oluşmaz.
class Transformer {
public:
    // Üretim sırasında üretilen token'ları anında bildirir.
    using TokenCallback = void (*)(int token_id, void* user_data);

    bool load(const std::string& path, std::string* error = nullptr);

    // Verilen token id'lerinden devam ederek max_new_tokens kadar üretir.
    // Geçersiz token id veya boş prompt için false döner, error doldurulur.
    bool generate(const std::vector<int>& prompt_ids,
                  int max_new_tokens,
                  const SamplingConfig& sampling,
                  std::vector<int>& out_ids,
                  TokenCallback callback = nullptr,
                  void* user_data = nullptr);

    const ModelConfig& config() const { return config_; }

    int headDim() const { return head_dim_; }
    int maxPosition() const { return static_cast<int>(rope_cos_.size() / head_dim_); }

    // Doğrulama/debug amaçlı: token dizisinin son pozisyon logitlerini üretir.
    // KV cache kullanmaz (her çağrı bağımsız).
    bool forwardPublic(const std::vector<int>& token_ids,
                       std::vector<float>* out_logits,
                       std::string* error);

private:
    // KV cache, halka tampon (ring buffer) olarak saklanır.
    // Düzen: (cap, n_embd) — satır = mutlak pozisyon, sütun = kanal.
    // Bu düzen hem QK^T nokta çarpımı hem de P·V toplamı için satır
    // ardışık erişim sağlar (her ikisi de kanal sabit, konum değişken).
    //
    // RoPE fazı K'ye yazılmadan önce uygulandığı için en eski girişleri
    // düşürmek pozisyon tutarlılığını bozmaz.
    struct KVCache {
        Tensor k;      // (cap, n_embd)
        Tensor v;      // (cap, n_embd)
        size_t head = 0;  // en eski girişin slot'u
        size_t len = 0;   // geçerli giriş sayısı
        size_t cap = 0;
    };

    struct Workspace {
        Tensor x, ln1, ln2, q, k, v, attn, proj, h, mlp;
        std::vector<float> row, row_out, scores;
    };

    // idx dizisinin logitlerini döndürür. pos_base, idx[0]'ın mutlak pozisyonudur.
    bool forward(const std::vector<int>& idx,
                 std::vector<KVCache>* cache,
                 int pos_base,
                 std::vector<float>& out_logits,
                 std::string* error);

    // --- Yardımcılar (sıcak yol; başlıkta tanımlı) ---
    void linear(const Tensor& x, const Tensor& W, Tensor& out);
    void layer_norm_batch(const Tensor& src, Tensor& dst, const std::vector<float>& w,
                          const std::vector<float>& b, size_t T, int C);
    void layer_norm_row(const Tensor& src, Tensor& dst, const std::vector<float>& w,
                        const std::vector<float>& b, size_t row, int C);
    void apply_rope(Tensor& x, int pos_base, size_t T, int C, int hd, int half,
                    const std::vector<float>& cos_t, const std::vector<float>& sin_t);
    void attention_cached(Tensor& out, const Tensor& q, const KVCache& kv,
                          size_t T, int C, int hd, size_t n_keys_base);
    void attention_prefill(Tensor& out, const Tensor& q, const Tensor& k, const Tensor& v,
                           size_t T, int C, int hd);

    void init_rope();
    void init_workspace(size_t max_batch);
    void reset_cache(std::vector<KVCache>& cache, size_t cap) const;

    ModelConfig config_;
    int head_dim_ = 0;
    int rope_max_pos_ = 0;

    // Ağırlıklar
    Tensor token_emb_;   // (vocab_size, n_embd)

    struct Block {
        std::vector<float> ln1_w, ln1_b;
        Tensor q_w, k_w, v_w, proj_w;  // (n_embd, n_embd)
        std::vector<float> ln2_w, ln2_b;
        Tensor fc1_w, fc2_w;            // (4*n_embd, n_embd), (n_embd, 4*n_embd)
        std::vector<float> fc1_b, fc2_b;
    };
    std::vector<Block> blocks_;

    std::vector<float> lnf_w, lnf_b;
    Tensor head_w_;  // (vocab_size, n_embd)

    // RoPE tabloları: pozisyon x head_dim/2
    std::vector<float> rope_cos_, rope_sin_;

    // Token başına yeniden ayırma yapmamak için paylaşılan buffer'lar
    Workspace ws_;
    std::vector<float> logits_;
    std::vector<float> probs_;
};

} // namespace cofeu