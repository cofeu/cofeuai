#include "cofeu/model.hpp"
#include "cofeu/tokenizer.hpp"

#include <iostream>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Kullanım: " << argv[0] << " <prompt> [max_tokens] [temperature]\n";
        return 1;
    }

    std::string prompt = argv[1];
    int max_tokens = (argc >= 3) ? std::stoi(argv[2]) : 200;
    float temperature = (argc >= 4) ? std::stof(argv[3]) : 0.8f;

    cofeu::Tokenizer tokenizer;
    if (!tokenizer.load("vocab.json")) {
        std::cerr << "vocab.json yüklenemedi. Önce eğitim + export yapın.\n";
        return 1;
    }

    cofeu::Transformer model;
    if (!model.load("cofeu.bin")) {
        std::cerr << "cofeu.bin yüklenemedi. Önce eğitim + export yapın.\n";
        return 1;
    }

    std::cout << "Model yüklendi: vocab=" << model.config().vocab_size
              << ", embd=" << model.config().n_embd
              << ", layer=" << model.config().n_layer;
#ifdef _OPENMP
    std::cout << ", OpenMP threads=" << omp_get_max_threads();
#endif
    std::cout << "\n\n";

    auto prompt_ids = tokenizer.encode(prompt);
    auto out_ids = model.generate(prompt_ids, max_tokens, temperature);

    std::cout << tokenizer.decode(out_ids) << "\n";
    return 0;
}