#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace cofeu {

// Byte-Pair Encoding (BPE) tokenizer (Python'daki BPETokenizer ile uyumlu).
//
// canonical BPE uygulanır: en düşük rank'lı çift, bağlı liste + öncelik
// kuyruğu ile birleştirilir. Bu, python/tokenizer.py::_bpe ile aynı
// algoritmadır ve her metin için aynı token id'lerini üretir.
class Tokenizer {
public:
    // vocab.json dosyasından yükler (merges + vocab formatı).
    bool load(const std::string& path, std::string* error = nullptr);

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids, bool skip_special_tokens = true) const;

    // Gömme tablosunun indekslenmesi gereken boyut (max id + 1).
    int vocabSize() const { return static_cast<int>(itos_.size()); }

    int eosId() const { return eos_id_; }
    int unkId() const { return unk_id_; }

    // En büyük token id'si + 1 olarak hesaplanan gerçek vocab boyutu.
    int tokenCount() const { return static_cast<int>(stoi_.size()); }

private:
    std::vector<std::pair<std::string, std::string>> merges_;
    // Birleştirme sırası (küçük = önce uygulanır): (id_a << 32 | id_b) -> rank
    std::unordered_map<uint64_t, int> merge_rank_;
    // rank -> birleşmiş token'ın vocab id'si
    std::vector<int> merged_id_;
    std::unordered_map<std::string, int> stoi_;
    std::vector<std::string> itos_;
    std::vector<bool> is_special_;
    int eos_id_ = -1;
    int unk_id_ = -1;
};

} // namespace cofeu