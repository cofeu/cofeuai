#include "cofeu/model.hpp"
#include "cofeu/tokenizer.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

void usage(const char* prog) {
    std::cerr << "Kullanım: " << prog << " <prompt> [max_tokens] [temperature]\n"
              << "\n"
              << "Seçenekler:\n"
              << "  --vocab <yol>     vocab.json yolu (varsayılan: checkpoints/vocab.json)\n"
              << "  --model <yol>     cofeu.bin yolu  (varsayılan: checkpoints/cofeu.bin)\n"
              << "  --top-k <n>       top-k sampling (0 = kapalı)\n"
              << "  --top-p <f>       top-p / nucleus sampling (<0 = kapalı)\n"
              << "  --rep-penalty <f> tekrar cezası (1.0 = kapalı)\n"
              << "  --seed <n>        RNG tohumu (0 = rastgele)\n"
              << "  --prompt-only     üretmeden yalnızca token sayısını yazdır\n"
              << "  -h, --help        bu yardım\n";
}

// std::stoi/stof istisna fırlatıyordu; yakalanmayınca std::terminate ile
// sessizce abort ediyordu. Açık hata mesajı veren güvenli sürüm.
bool parse_int(const char* s, int& out) {
    try {
        char* end = nullptr;
        const long v = std::strtol(s, &end, 10);
        if (!end || end == s || *end != '\0') return false;
        out = static_cast<int>(v);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_float(const char* s, float& out) {
    try {
        char* end = nullptr;
        const double v = std::strtod(s, &end);
        if (end == s || (end && *end != '\0')) return false;
        out = static_cast<float>(v);
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace

int main(int argc, char** argv) {
    std::string prompt;
    std::string vocab_path = "checkpoints/vocab.json";
    std::string model_path = "checkpoints/cofeu.bin";
    int max_tokens = 200;
    float temperature = 0.8f;
    int top_k = 0;
    float top_p = -1.0f;
    float rep_penalty = 1.0f;
    unsigned seed = 0;
    bool prompt_only = false;

    // Argümanları ayrıştır
    std::vector<std::string> positional;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Hata: " << name << " bir değer gerektirir\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else if (a == "--vocab") {
            vocab_path = need("--vocab");
        } else if (a == "--model") {
            model_path = need("--model");
        } else if (a == "--top-k") {
            if (!parse_int(need("--top-k"), top_k)) {
                std::cerr << "Hata: --top-k geçersiz sayı\n";
                return 2;
            }
        } else if (a == "--top-p") {
            if (!parse_float(need("--top-p"), top_p)) {
                std::cerr << "Hata: --top-p geçersiz sayı\n";
                return 2;
            }
        } else if (a == "--rep-penalty") {
            if (!parse_float(need("--rep-penalty"), rep_penalty)) {
                std::cerr << "Hata: --rep-penalty geçersiz sayı\n";
                return 2;
            }
        } else if (a == "--seed") {
            int s = 0;
            if (!parse_int(need("--seed"), s) || s < 0) {
                std::cerr << "Hata: --seed geçersiz sayı\n";
                return 2;
            }
            seed = static_cast<unsigned>(s);
        } else if (a == "--prompt-only") {
            prompt_only = true;
        } else if (!a.empty() && a[0] == '-' && a != "-") {
            std::cerr << "Hata: bilinmeyen seçenek " << a << "\n\n";
            usage(argv[0]);
            return 2;
        } else {
            positional.push_back(a);
        }
    }

    if (positional.empty()) {
        usage(argv[0]);
        return 1;
    }
    prompt = positional[0];
    if (positional.size() >= 2 && !parse_int(positional[1].c_str(), max_tokens)) {
        std::cerr << "Hata: max_tokens geçersiz sayı: " << positional[1] << "\n";
        return 2;
    }
    if (positional.size() >= 3 && !parse_float(positional[2].c_str(), temperature)) {
        std::cerr << "Hata: temperature geçersiz sayı: " << positional[2] << "\n";
        return 2;
    }
    if (max_tokens < 0) {
        std::cerr << "Hata: max_tokens negatif olamaz\n";
        return 2;
    }

    // --- Tokenizer ---
    cofeu::Tokenizer tokenizer;
    std::string err;
    if (!tokenizer.load(vocab_path, &err)) {
        std::cerr << "Hata: vocab yüklenemedi -> " << err << "\n";
        return 1;
    }

    // --- Model ---
    cofeu::Transformer model;
    if (!model.load(model_path, &err)) {
        std::cerr << "Hata: model yüklenemedi -> " << err << "\n";
        return 1;
    }

    // Vocab/model uyumsuzluğu, tokenizer'ın ürettiği id'lerin gömme
    // tablosunun dışına çıkmasına (ve bellek dışı okumaya) yol açıyordu.
    if (tokenizer.vocabSize() != model.config().vocab_size) {
        std::cerr << "Hata: vocab uyuşmazlığı! Tokenizer " << tokenizer.vocabSize()
                  << " token, model " << model.config().vocab_size
                  << " bekliyor.\n"
                  << "      'python export.py' ile modeli yeniden dışa aktarın.\n";
        return 1;
    }

    std::cerr << "Model: vocab=" << model.config().vocab_size
              << " embd=" << model.config().n_embd
              << " head=" << model.config().n_head
              << " layer=" << model.config().n_layer
              << " pencere=" << model.config().block_size;
#ifdef _OPENMP
    std::cerr << " threads=" << omp_get_max_threads();
#endif
    std::cerr << "\n";

    const auto prompt_ids = tokenizer.encode(prompt);
    if (prompt_ids.empty()) {
        std::cerr << "Hata: prompt vocab'da hiçbir karakter içermiyor.\n";
        return 1;
    }

    if (prompt_only) {
        std::cout << prompt_ids.size() << "\n";
        return 0;
    }

    cofeu::SamplingConfig sampling;
    sampling.temperature = temperature;
    sampling.top_k = top_k;
    sampling.top_p = top_p;
    sampling.repetition_penalty = rep_penalty;
    sampling.eos_token_id = tokenizer.eosId();
    sampling.seed = seed;

    std::vector<int> out_ids;
    if (!model.generate(prompt_ids, max_tokens, sampling, out_ids)) {
        std::cerr << "Hata: üretim başarısız\n";
        return 1;
    }

    // generate() prompt + üretilen token'ların tamamını döndürür; sadece
    // üretilen kısmı yazdırıyoruz (aksi halde prompt tekrar basılıyordu).
    const size_t n_prompt = prompt_ids.size();
    std::cout << tokenizer.decode(std::vector<int>(out_ids.begin() + n_prompt, out_ids.end()))
              << std::endl;
    return 0;
}