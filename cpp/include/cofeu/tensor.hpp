#pragma once

#include <vector>
#include <cstddef>

namespace cofeu {

// Basit 2D tensor (row-major). C++ inference için yeterli.
struct Tensor {
    std::vector<float> data;
    size_t rows = 0;
    size_t cols = 0;

    Tensor() = default;
    Tensor(size_t r, size_t c) : data(r * c, 0.0f), rows(r), cols(c) {}

    float& at(size_t i, size_t j) { return data[i * cols + j]; }
    const float& at(size_t i, size_t j) const { return data[i * cols + j]; }

    void resize(size_t r, size_t c) {
        rows = r;
        cols = c;
        data.assign(r * c, 0.0f);
    }
};

} // namespace cofeu