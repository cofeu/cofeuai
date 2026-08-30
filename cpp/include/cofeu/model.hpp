#pragma once

#include <string>
#include <vector>

#include "tensor.hpp"

namespace cofeu {

struct ModelConfig {
    int vocab_size = 0;
    int n_embd = 0;
    int n_head = 0;
    int n_layer = 0;
    int block_size = 0;
};

// C++ ile saf transformer inference motoru.
// PyTorch bağımlılığı yoktur; ağırlıklar export.py ile üretilen
// cofeu.bin dosyasından yüklenir.
class Transformer {
public:
    bool load(const std::string& path);

    // Verilen token id'lerinden devam ederek max_new_tokens kadar üretir.
    std::vector<int> generate(const std::vector<int>& prompt_ids,
                              int max_new_tokens,
                              float temperature,
                              unsigned seed = 42);

    const ModelConfig& config() const { return config_; }

private:
    // KV cache: her blok için (K, V) matrisleri. K/V: (n_head*head_dim, seq_len)
    // satır-major olarak (n_embd, seq_len) şeklinde saklanır.
    struct KVCache {
        Tensor k;  // (n_embd, seq_len)
        Tensor v;  // (n_embd, seq_len)
    };

    // İleri geçiş: son pozisyonun logitlerini döndürür.
    // cache null ise tüm diziyi işler; değilse sadece yeni token'ı işler.
    std::vector<float> forward(const std::vector<int>& idx,
                               std::vector<KVCache>* cache = nullptr);

    ModelConfig config_;

    // Ağırlıklar
    Tensor token_emb_;   // (vocab_size, n_embd)
    Tensor pos_emb_;     // (block_size, n_embd)

    struct Block {
        // LayerNorm 1
        std::vector<float> ln1_w, ln1_b;
        // Attention
        Tensor q_w, k_w, v_w, proj_w;  // (n_embd, n_embd)
        // LayerNorm 2
        std::vector<float> ln2_w, ln2_b;
        // MLP
        Tensor fc1_w, fc2_w;           // (4*n_embd, n_embd) ve (n_embd, 4*n_embd)
        std::vector<float> fc1_b, fc2_b;
    };
    std::vector<Block> blocks_;

    std::vector<float> lnf_w, lnf_b;
    Tensor head_w_;  // (vocab_size, n_embd)
};

} // namespace cofeu