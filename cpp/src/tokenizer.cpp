#include "cofeu/tokenizer.hpp"

#include <cctype>
#include <cstring>
#include <fstream>
#include <sstream>

namespace cofeu {

namespace {

// UTF-8 karakter uzunluğunu döndürür.
size_t utf8_len(unsigned char c) {
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

// JSON string'i ayrıştırır (kaçış dizileri desteklenir).
// pos, açılış tırnağından sonraki konumda olmalıdır.
std::string parse_json_string(const std::string& s, size_t& pos) {
    std::string out;
    while (pos < s.size() && s[pos] != '"') {
        if (s[pos] == '\\' && pos + 1 < s.size()) {
            pos++;
            char esc = s[pos];
            if (esc == 'n') out += '\n';
            else if (esc == 't') out += '\t';
            else if (esc == 'r') out += '\r';
            else if (esc == 'u') {
                // \uXXXX (UTF-16) — basitlik için ham olarak ekle
                out += "\\u";
                for (int i = 0; i < 4 && pos + 1 < s.size(); i++) {
                    pos++;
                    out += s[pos];
                }
            } else {
                out += esc;
            }
        } else {
            out += s[pos];
        }
        pos++;
    }
    return out;
}

} // namespace

bool Tokenizer::load(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return false;

    std::stringstream ss;
    ss << f.rdbuf();
    std::string content = ss.str();

    merges_.clear();
    stoi_.clear();
    itos_.clear();

    // "merges" dizisini bul
    size_t merges_pos = content.find("\"merges\"");
    if (merges_pos != std::string::npos) {
        size_t arr_start = content.find('[', merges_pos);
        size_t arr_end = content.find(']', arr_start);
        if (arr_start != std::string::npos && arr_end != std::string::npos) {
            size_t pos = arr_start + 1;
            while (pos < arr_end) {
                size_t q = content.find('"', pos);
                if (q == std::string::npos || q >= arr_end) break;
                pos = q + 1;
                std::string merge = parse_json_string(content, pos);
                pos++;  // kapanış tırnağı
                // "a b" -> (a, b)
                size_t sp = merge.find(' ');
                if (sp != std::string::npos) {
                    merges_.emplace_back(merge.substr(0, sp), merge.substr(sp + 1));
                }
                pos = content.find(',', pos);
                if (pos == std::string::npos) break;
                pos++;
            }
        }
    }

    // "vocab" nesnesini bul
    size_t vocab_pos = content.find("\"vocab\"");
    if (vocab_pos != std::string::npos) {
        size_t obj_start = content.find('{', vocab_pos);
        size_t obj_end = content.find('}', obj_start);
        if (obj_start != std::string::npos && obj_end != std::string::npos) {
            size_t pos = obj_start + 1;
            int max_id = -1;
            while (pos < obj_end) {
                size_t q = content.find('"', pos);
                if (q == std::string::npos || q >= obj_end) break;
                pos = q + 1;
                std::string key = parse_json_string(content, pos);
                pos++;  // kapanış tırnağı

                size_t colon = content.find(':', pos);
                if (colon == std::string::npos || colon >= obj_end) break;
                size_t val_start = content.find_first_of("0123456789", colon);
                if (val_start == std::string::npos || val_start >= obj_end) break;
                size_t val_end = val_start;
                while (val_end < obj_end && isdigit(content[val_end])) val_end++;

                int id = std::stoi(content.substr(val_start, val_end - val_start));
                stoi_[key] = id;
                if (id > max_id) max_id = id;

                pos = content.find(',', val_end);
                if (pos == std::string::npos) break;
                pos++;
            }

            itos_.resize(max_id + 1);
            for (const auto& [k, v] : stoi_) {
                itos_[v] = k;
            }
        }
    }

    return !stoi_.empty();
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    // Metni UTF-8 karakterlerine böl
    std::vector<std::string> tokens;
    size_t i = 0;
    while (i < text.size()) {
        size_t len = utf8_len(static_cast<unsigned char>(text[i]));
        tokens.push_back(text.substr(i, len));
        i += len;
    }

    // BPE birleştirmelerini uygula
    for (const auto& [a, b] : merges_) {
        std::vector<std::string> new_tokens;
        size_t j = 0;
        while (j < tokens.size()) {
            if (j + 1 < tokens.size() && tokens[j] == a && tokens[j + 1] == b) {
                new_tokens.push_back(a + b);
                j += 2;
            } else {
                new_tokens.push_back(tokens[j]);
                j += 1;
            }
        }
        tokens = std::move(new_tokens);
    }

    std::vector<int> ids;
    ids.reserve(tokens.size());
    for (const auto& t : tokens) {
        auto it = stoi_.find(t);
        if (it != stoi_.end()) {
            ids.push_back(it->second);
        }
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) {
        if (id >= 0 && id < static_cast<int>(itos_.size())) {
            out += itos_[id];
        }
    }
    return out;
}

} // namespace cofeu

// --- Python (ctypes) için C API ---

extern "C" {

// Tokenizer oluştur ve vocab.json'dan yükle. Başarılıysa handle döner, değilse nullptr.
void* cofeu_tokenizer_load(const char* path) {
    auto* tok = new cofeu::Tokenizer();
    if (!tok->load(path)) {
        delete tok;
        return nullptr;
    }
    return static_cast<void*>(tok);
}

void cofeu_tokenizer_free(void* handle) {
    delete static_cast<cofeu::Tokenizer*>(handle);
}

int cofeu_tokenizer_vocab_size(void* handle) {
    return static_cast<cofeu::Tokenizer*>(handle)->vocabSize();
}

// Metni encode eder. ids çıktı buffer'ına yazılır, token sayısını döndürür.
int cofeu_tokenizer_encode(void* handle, const char* text, int* ids, int max_ids) {
    auto* tok = static_cast<cofeu::Tokenizer*>(handle);
    auto result = tok->encode(std::string(text));
    int n = static_cast<int>(result.size());
    if (n > max_ids) n = max_ids;
    for (int i = 0; i < n; i++) ids[i] = result[i];
    return n;
}

// Token id'lerini metne çevirir. Çıktıyı out buffer'ına yazar.
void cofeu_tokenizer_decode(void* handle, const int* ids, int n, char* out, int max_out) {
    auto* tok = static_cast<cofeu::Tokenizer*>(handle);
    std::vector<int> vec(ids, ids + n);
    std::string text = tok->decode(vec);
    int len = static_cast<int>(text.size());
    if (len > max_out - 1) len = max_out - 1;
    std::memcpy(out, text.data(), len);
    out[len] = '\0';
}

} // extern "C"