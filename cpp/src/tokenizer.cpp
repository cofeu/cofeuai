#include "cofeu/tokenizer.hpp"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <queue>
#include <sstream>
#include <utility>

namespace cofeu {

namespace {

// ---------------------------------------------------------------------------
// Minimal JSON ayrıştırıcı
//
// vocab.json içindeki \`}\` karakteri bir token string'inin içinde geçebiliyor
// (Python ensure_ascii=False ile yazıyor) ve \`"merges"\` gibi anahtarları
// ham substring aramasıyla bulmak yanlış hedef verebiliyor. Bu yüzden gerçek
// bir JSON ayrıştırıcı kullanıyoruz.
// ---------------------------------------------------------------------------

struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string str;
    std::vector<JsonValue> arr;
    std::vector<std::pair<std::string, JsonValue>> obj;

    const JsonValue* find(const std::string& key) const {
        if (type != Type::Object) return nullptr;
        for (const auto& kv : obj) {
            if (kv.first == key) return &kv.second;
        }
        return nullptr;
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& s) : s_(s) {}

    bool parse(JsonValue& out) {
        skip_ws();
        if (!parse_value(out)) return false;
        return true;
    }

    const std::string& error() const { return error_; }

private:
    const std::string& s_;
    size_t pos_ = 0;
    std::string error_;

    bool fail(const std::string& msg) {
        if (error_.empty()) {
            error_ = msg + " (offset " + std::to_string(pos_) + ")";
        }
        return false;
    }

    void skip_ws() {
        while (pos_ < s_.size()) {
            char c = s_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                pos_++;
            } else {
                break;
            }
        }
    }

    bool parse_value(JsonValue& out) {
        if (pos_ >= s_.size()) return fail("JSON bitti");
        switch (s_[pos_]) {
            case '{': return parse_object(out);
            case '[': return parse_array(out);
            case '"':
                out.type = JsonValue::Type::String;
                return parse_string(out.str);
            case 't':
                if (s_.compare(pos_, 4, "true") == 0) {
                    out.type = JsonValue::Type::Bool;
                    out.boolean = true;
                    pos_ += 4;
                    return true;
                }
                return fail("geçersiz literal");
            case 'f':
                if (s_.compare(pos_, 5, "false") == 0) {
                    out.type = JsonValue::Type::Bool;
                    out.boolean = false;
                    pos_ += 5;
                    return true;
                }
                return fail("geçersiz literal");
            case 'n':
                if (s_.compare(pos_, 4, "null") == 0) {
                    out.type = JsonValue::Type::Null;
                    pos_ += 4;
                    return true;
                }
                return fail("geçersiz literal");
            default: return parse_number(out);
        }
    }

    bool parse_number(JsonValue& out) {
        size_t start = pos_;
        if (pos_ < s_.size() && (s_[pos_] == '-' || s_[pos_] == '+')) pos_++;
        bool any = false;
        while (pos_ < s_.size() && (isdigit(static_cast<unsigned char>(s_[pos_])) || s_[pos_] == '.' ||
                                   s_[pos_] == 'e' || s_[pos_] == 'E' || s_[pos_] == '-' || s_[pos_] == '+')) {
            any = true;
            pos_++;
        }
        if (!any) return fail("sayı bekleniyordu");
        out.type = JsonValue::Type::Number;
        out.number = std::strtod(s_.substr(start, pos_ - start).c_str(), nullptr);
        return true;
    }

    // \uXXXX kaçışını UTF-8'e çevirir (surrogate pair destekli)
    static void append_utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (cp >> 18));
            out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }

    bool parse_hex4(uint32_t& out) {
        if (pos_ + 4 > s_.size()) return fail("kısa \\u kaçışı");
        out = 0;
        for (int i = 0; i < 4; i++) {
            char c = s_[pos_ + i];
            out <<= 4;
            if (c >= '0' && c <= '9') out |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') out |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') out |= static_cast<uint32_t>(c - 'A' + 10);
            else return fail("geçersiz hex");
        }
        pos_ += 4;
        return true;
    }

    bool parse_string(std::string& out) {
        if (pos_ >= s_.size() || s_[pos_] != '"') return fail("string bekleniyordu");
        pos_++;
        out.clear();
        while (pos_ < s_.size()) {
            char c = s_[pos_];
            if (c == '"') {
                pos_++;
                return true;
            }
            if (c == '\\') {
                pos_++;
                if (pos_ >= s_.size()) return fail("kaçış koptu");
                char esc = s_[pos_++];
                switch (esc) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case '/': out += '/'; break;
                    case '\\': out += '\\'; break;
                    case '"': out += '"'; break;
                    case 'u': {
                        uint32_t cp = 0;
                        if (!parse_hex4(cp)) return false;
                        if (cp >= 0xD800 && cp <= 0xDBFF && pos_ + 1 < s_.size() &&
                            s_[pos_] == '\\' && s_[pos_ + 1] == 'u') {
                            size_t save = pos_;
                            pos_ += 2;
                            uint32_t lo = 0;
                            if (!parse_hex4(lo)) return false;
                            if (lo >= 0xDC00 && lo <= 0xDFFF) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            } else {
                                pos_ = save;  // geçerli değil, ham bırak
                            }
                        }
                        append_utf8(out, cp);
                        break;
                    }
                    default: return fail("bilinmeyen kaçış");
                }
                continue;
            }
            out += c;
            pos_++;
        }
        return fail("kapanmamış string");
    }

    bool parse_array(JsonValue& out) {
        out.type = JsonValue::Type::Array;
        pos_++;  // '['
        skip_ws();
        if (pos_ < s_.size() && s_[pos_] == ']') {
            pos_++;
            return true;
        }
        while (pos_ < s_.size()) {
            JsonValue v;
            if (!parse_value(v)) return false;
            out.arr.push_back(std::move(v));
            skip_ws();
            if (pos_ < s_.size() && s_[pos_] == ',') {
                pos_++;
                skip_ws();
                continue;
            }
            if (pos_ < s_.size() && s_[pos_] == ']') {
                pos_++;
                return true;
            }
            return fail("virgül veya ']' bekleniyordu");
        }
        return fail("kapanmamış dizi");
    }

    bool parse_object(JsonValue& out) {
        out.type = JsonValue::Type::Object;
        pos_++;  // '{'
        skip_ws();
        if (pos_ < s_.size() && s_[pos_] == '}') {
            pos_++;
            return true;
        }
        while (pos_ < s_.size()) {
            skip_ws();
            std::string key;
            if (!parse_string(key)) return false;
            skip_ws();
            if (pos_ >= s_.size() || s_[pos_] != ':') return fail("':' bekleniyordu");
            pos_++;
            skip_ws();  // ':' ile değer arasında boşluk olabilir
            JsonValue v;
            if (!parse_value(v)) return false;
            out.obj.emplace_back(std::move(key), std::move(v));
            skip_ws();
            if (pos_ < s_.size() && s_[pos_] == ',') {
                pos_++;
                continue;
            }
            if (pos_ < s_.size() && s_[pos_] == '}') {
                pos_++;
                return true;
            }
            return fail("virgül veya '}' bekleniyordu");
        }
        return fail("kapanmamış nesne");
    }
};

// UTF-8 karakter uzunluğu (baş bayta göre)
inline size_t utf8_len(unsigned char c) {
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

// İki token id'sini tek bir 64-bit anahtara paketler
inline uint64_t pair_key(int32_t a, int32_t b) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) |
           static_cast<uint32_t>(b);
}

} // namespace

bool Tokenizer::load(const std::string& path, std::string* error) {
    merges_.clear();
    merge_rank_.clear();
    stoi_.clear();
    itos_.clear();
    is_special_.clear();
    eos_id_ = -1;
    unk_id_ = -1;

    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        if (error) *error = "dosya açılamadı: " + path;
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string content = ss.str();

    JsonValue root;
    JsonParser parser(content);
    if (!parser.parse(root)) {
        if (error) *error = "vocab.json ayrıştırılamadı: " + parser.error();
        return false;
    }

    // --- vocab ---
    const JsonValue* vocab = root.find("vocab");
    if (!vocab || vocab->type != JsonValue::Type::Object || vocab->obj.empty()) {
        if (error) *error = "\"vocab\" nesnesi bulunamadı veya boş";
        return false;
    }

    int max_id = -1;
    for (const auto& kv : vocab->obj) {
        if (kv.second.type != JsonValue::Type::Number) continue;
        int id = static_cast<int>(kv.second.number);
        if (id < 0) continue;
        // Yinelenen id varsa son yazan kazanır (Python dict davranışı)
        if (stoi_.find(kv.first) == stoi_.end()) {
            stoi_[kv.first] = id;
            if (id > max_id) max_id = id;
        }
    }
    if (stoi_.empty()) {
        if (error) *error = "vocab boş";
        return false;
    }

    itos_.assign(static_cast<size_t>(max_id) + 1, std::string());
    for (const auto& kv : stoi_) itos_[static_cast<size_t>(kv.second)] = kv.first;

    // --- special token'lar ---
    std::vector<std::string> specials;
    if (const JsonValue* st = root.find("special_tokens"); st && st->type == JsonValue::Type::Array) {
        for (const auto& v : st->arr) specials.push_back(v.str);
    }
    if (specials.empty()) {
        specials = {"<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>"};
    }
    is_special_.assign(itos_.size(), false);
    for (const auto& s : specials) {
        auto it = stoi_.find(s);
        if (it != stoi_.end()) is_special_[static_cast<size_t>(it->second)] = true;
        if (s == "<|eos|>") eos_id_ = it != stoi_.end() ? it->second : -1;
        if (s == "<|unk|>") unk_id_ = it != stoi_.end() ? it->second : -1;
    }

    // --- merges ---
    // Kabul edilen biçimler:
    //   v2: [["a", "b"], ...]        (belirsizlik yok — token boşluk içerebilir)
    //   v1: ["a b", ...]             (ilk boşluktan bölünür)
    const JsonValue* merges = root.find("merges");
    if (merges && merges->type == JsonValue::Type::Array) {
        merges_.reserve(merges->arr.size());
        for (const auto& m : merges->arr) {
            std::string a, b;
            if (m.type == JsonValue::Type::Array) {
                if (m.arr.size() != 2) continue;
                a = m.arr[0].str;
                b = m.arr[1].str;
            } else if (m.type == JsonValue::Type::String) {
                size_t sp = m.str.find(' ');
                if (sp == std::string::npos) continue;
                a = m.str.substr(0, sp);
                b = m.str.substr(sp + 1);
            } else {
                continue;
            }
            merges_.emplace_back(a, b);
        }
    }

    // Merge tablosunu hazırla: (id_a, id_b) -> rank, rank -> id_(a+b)
    merge_rank_.reserve(merges_.size() * 2);
    merged_id_.assign(merges_.size(), -1);
    for (size_t rank = 0; rank < merges_.size(); rank++) {
        const auto& [a, b] = merges_[rank];
        auto ia = stoi_.find(a);
        auto ib = stoi_.find(b);
        if (ia == stoi_.end() || ib == stoi_.end()) continue;
        auto im = stoi_.find(a + b);
        if (im == stoi_.end()) continue;  // birleşmiş token vocab'ta yok, atla
        merge_rank_[pair_key(ia->second, ib->second)] = static_cast<int>(rank);
        merged_id_[rank] = im->second;
    }

    if (error) error->clear();
    return true;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    if (text.empty()) return ids;

    const size_t n_chars = text.size();
    ids.reserve(n_chars);

    // 1) Metni UTF-8 karakterlerine böl ve her karakteri vocab id'sine eşle.
    //    Vocab'da olmayan karakterler <|unk|> olur (sessizce düşürülmez).
    const int unknown = unk_id_;
    size_t i = 0;
    while (i < n_chars) {
        size_t len = utf8_len(static_cast<unsigned char>(text[i]));
        if (i + len > n_chars) len = 1;
        auto it = stoi_.find(text.substr(i, len));
        if (it != stoi_.end()) {
            ids.push_back(it->second);
        } else if (unknown >= 0) {
            ids.push_back(unknown);
        }
        i += len;
    }

    const size_t n = ids.size();
    if (n < 2 || merge_rank_.empty()) return ids;

    // 2) Canonical BPE: en düşük rank'lı çift, bağlı liste + öncelik kuyruğu.
    //    python/tokenizer.py::_bpe ile aynı algoritma.
    std::vector<int> nxt(n);
    std::vector<int> prev(n);
    std::vector<char> alive(n, 1);
    for (size_t t = 0; t < n; t++) {
        nxt[t] = static_cast<int>(t) + 1;
        prev[t] = static_cast<int>(t) - 1;
    }

    typedef std::pair<int, int> HeapItem;  // (rank, pozisyon)
    std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<HeapItem>> heap;
    for (size_t t = 0; t + 1 < n; t++) {
        auto it = merge_rank_.find(pair_key(ids[t], ids[t + 1]));
        if (it != merge_rank_.end()) heap.emplace(it->second, static_cast<int>(t));
    }

    while (!heap.empty()) {
        const int rank = heap.top().first;
        const int t = heap.top().second;
        heap.pop();

        if (!alive[t]) continue;
        const int j = nxt[t];
        if (j >= static_cast<int>(n)) continue;
        auto it = merge_rank_.find(pair_key(ids[t], ids[j]));
        if (it == merge_rank_.end() || it->second != rank) continue;  // bayat giriş

        const int m = merged_id_[static_cast<size_t>(rank)];
        if (m < 0) continue;

        ids[t] = m;
        alive[j] = 0;
        const int k = nxt[j];
        nxt[t] = k;
        if (k < static_cast<int>(n)) {
            prev[k] = t;
            auto rk = merge_rank_.find(pair_key(ids[t], ids[k]));
            if (rk != merge_rank_.end()) heap.emplace(rk->second, t);
        }
        if (prev[t] >= 0) {
            auto rp = merge_rank_.find(pair_key(ids[prev[t]], ids[t]));
            if (rp != merge_rank_.end()) heap.emplace(rp->second, prev[t]);
        }
    }

    // 3) Bağlı listeyi sıraya diz. n sentinel (sonlandırıcı) olduğundan
    //    sınır kontrolü idslere DOKUNMADAN yapılmalı; aksi halde ids[n]
    //    taşma okuması oluşur.
    std::vector<int> out;
    out.reserve(n);
    for (int t = 0; t >= 0; ) {
        out.push_back(ids[t]);
        const int nxt_t = nxt[t];
        if (nxt_t >= 0 && static_cast<size_t>(nxt_t) >= n) break;  // son geçerli düğüm
        t = nxt_t;
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids, bool skip_special_tokens) const {
    std::string out;
    out.reserve(ids.size() * 4);
    const int V = static_cast<int>(itos_.size());
    for (int id : ids) {
        if (id < 0 || id >= V) continue;
        if (skip_special_tokens && !is_special_.empty() && is_special_[static_cast<size_t>(id)]) continue;
        out += itos_[static_cast<size_t>(id)];
    }
    return out;
}

} // namespace cofeu

// --- Python (ctypes) için C API ---

extern "C" {

// Tokenizer oluştur ve vocab.json'dan yükle. Başarılıysa handle döner, değilse nullptr.
void* cofeu_tokenizer_load(const char* path, char* error_out, int error_buf) {
    auto* tok = new cofeu::Tokenizer();
    std::string err;
    if (!tok->load(path ? path : "", &err)) {
        delete tok;
        if (error_out && error_buf > 0) {
            std::snprintf(error_out, static_cast<size_t>(error_buf), "%s", err.c_str());
        }
        return nullptr;
    }
    return static_cast<void*>(tok);
}

void cofeu_tokenizer_free(void* handle) {
    delete static_cast<cofeu::Tokenizer*>(handle);
}

int cofeu_tokenizer_vocab_size(void* handle) {
    if (!handle) return 0;
    return static_cast<cofeu::Tokenizer*>(handle)->vocabSize();
}

int cofeu_tokenizer_token_count(void* handle) {
    if (!handle) return 0;
    return static_cast<cofeu::Tokenizer*>(handle)->tokenCount();
}

int cofeu_tokenizer_eos_id(void* handle) {
    if (!handle) return -1;
    return static_cast<cofeu::Tokenizer*>(handle)->eosId();
}

// Metni encode eder. ids çıktı buffer'ına yazar, token sayısını döndürür.
int cofeu_tokenizer_encode(void* handle, const char* text, int* ids, int max_ids) {
    if (!handle || !text || !ids || max_ids <= 0) return 0;
    auto result = static_cast<cofeu::Tokenizer*>(handle)->encode(std::string(text));
    int n = static_cast<int>(result.size());
    if (n > max_ids) n = max_ids;  // buffer taşması; çağıran taraf yeterli yer ayırmalı
    for (int i = 0; i < n; i++) ids[i] = result[i];
    return n;
}

// Token id'lerini metne çevirir. Çıktıyı out buffer'ına yazar.
// Dönüş değeri: yazılan bayt sayısı (null terminatör hariç).
int cofeu_tokenizer_decode(void* handle, const int* ids, int n, char* out, int max_out) {
    if (!out || max_out <= 0) return 0;
    out[0] = '\0';
    if (!handle || !ids || n <= 0) return 0;
    std::string text = static_cast<cofeu::Tokenizer*>(handle)->decode(std::vector<int>(ids, ids + n));

    size_t len = text.size();
    if (len >= static_cast<size_t>(max_out)) {
        // Birleşik karakterin ortasından kesmemek için UTF-8 sınırına geri al
        len = static_cast<size_t>(max_out) - 1;
        while (len > 0 && (static_cast<unsigned char>(text[len]) & 0xC0) == 0x80) len--;
    }
    std::memcpy(out, text.data(), len);
    out[len] = '\0';
    return static_cast<int>(len);
}

// Gerekli decode buffer boyutunu döndürür (null terminatör dahil).
int cofeu_tokenizer_decode_size(void* handle, const int* ids, int n) {
    if (!handle || !ids || n <= 0) return 1;
    return static_cast<int>(static_cast<cofeu::Tokenizer*>(handle)
                                ->decode(std::vector<int>(ids, ids + n)).size()) + 1;
}

} // extern "C"