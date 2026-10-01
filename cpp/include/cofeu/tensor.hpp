#pragma once

#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace cofeu {

// Basit 2D tensor (row-major). C++ inference için yeterli.
struct Tensor {
    std::vector<float> data;
    size_t rows = 0;
    size_t cols = 0;

    Tensor() = default;
    Tensor(size_t r, size_t c) : data(r * c, 0.0f), rows(r), cols(c) {}

    // --- Erişim ---
    //
    // at() bilinçli olarak kontrolsuzdur: sıcak döngülerde (her token için
    // ~30 çağrı) sınır testi ölçülebilir bir maliyet getirir. Dışarıdan
    // gelen indeksler (token id, pozisyon) model katmanında doğrulanır;
    // at_bounds() bunları doğrulamak için kullanılır.
    float& at(size_t i, size_t j) { return data[i * cols + j]; }
    const float& at(size_t i, size_t j) const { return data[i * cols + j]; }

    // Ham pointer erişimi (iç döngülerde at()'in indeks çarpımından kaçınmak için)
    float* row(size_t i) { return data.data() + i * cols; }
    const float* row(size_t i) const { return data.data() + i * cols; }
    const float* data_ptr() const { return data.data(); }
    float* data_ptr() { return data.data(); }

    bool in_bounds(size_t i, size_t j) const { return i < rows && j < cols; }

    // Sınır denetimli erişim — dış kaynaklı indeksler için.
    float& at_bounds(size_t i, size_t j) {
        if (!in_bounds(i, j)) {
            throw std::out_of_range("Tensor::at_bounds: indeks ("
                                    + std::to_string(i) + ", " + std::to_string(j)
                                    + ") sınırların dışında (satır " + std::to_string(rows)
                                    + ", sütun " + std::to_string(cols) + ")");
        }
        return data[i * cols + j];
    }

    void resize(size_t r, size_t c) {
        rows = r;
        cols = c;
        data.assign(r * c, 0.0f);
    }

    // Boyutu değiştirir ama içeriği sıfırlamaz — üzerine tamamen yazılacak
    // geçici buffer'lar için (her forward'da sıfırlamak boşa iş).
    void resize_uninit(size_t r, size_t c) {
        const size_t need = r * c;
        if (data.size() != need) data.resize(need);
        rows = r;
        cols = c;
    }

    void clear() {
        data.clear();
        rows = 0;
        cols = 0;
    }
};

} // namespace cofeu