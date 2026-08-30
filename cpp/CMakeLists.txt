cmake_minimum_required(VERSION 3.10)
project(CofeuAI CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# OpenMP ile paralelleştirme
find_package(OpenMP REQUIRED)

# Paylaşılan kütüphane (Python'dan ctypes ile çağrılır)
add_library(cofeu SHARED
    src/model.cpp
    src/tokenizer.cpp
)

target_include_directories(cofeu PRIVATE include)
target_link_libraries(cofeu PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(cofeu PRIVATE -O3 -march=native -funroll-loops)

add_executable(cofeu_infer
    src/main.cpp
    src/model.cpp
    src/tokenizer.cpp
)

target_include_directories(cofeu_infer PRIVATE include)

# OpenMP bağla
target_link_libraries(cofeu_infer PRIVATE OpenMP::OpenMP_CXX)

# Optimizasyon
target_compile_options(cofeu_infer PRIVATE -O3 -march=native -funroll-loops)