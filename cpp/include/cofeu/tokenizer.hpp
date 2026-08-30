#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace cofeu {

// Byte-Pair Encoding (BPE) tokenizer (Python'daki BPETokenizer ile uyumlu).
class Tokenizer {
public:
    // vocab.json dosyasından yükler (merges + vocab formatı).
    bool load(const std::string& path);

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;

    int vocabSize() const { return static_cast<int>(itos_.size()); }

private:
    std::vector<std::pair<std::string, std::string>> merges_;
    std::unordered_map<std::string, int> stoi_;
    std::vector<std::string> itos_;
};

} // namespace cofeu