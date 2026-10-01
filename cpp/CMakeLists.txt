cmake_minimum_required(VERSION 3.16)
project(CofeuAI VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT CMAKE_CXX_FLAGS_RELEASE)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

# OpenMP ile paralelleştirme
find_package(OpenMP)

set(COFEU_WARNINGS -Wall -Wextra -Wpedantic)
set(COFEU_OPT -funroll-loops -fno-math-errno)

# --- Paylaşılan kütüphane (Python'dan ctypes ile çağrılır) ---
#
# -march=native bilerek KULLANILMAZ: bu kütüphane makineye özgü derlenip
# başka bir bilgisayarda SIGILL üretiyordu. Bunun yerine çalışma zamanında
# CPU özelliklerini ölçen, makineden bağımsız bir temel seviye kullanıyoruz.
# İsterseniz -DCMAKE_CXX_FLAGS="-march=x86-64-v3" ile kendi makinenize
# göre sabitleyebilirsiniz.
set(COFEU_BASE_ARCH -march=x86-64-v2 -mtune=native)

add_library(cofeu SHARED
    src/model.cpp
    src/tokenizer.cpp
)

target_include_directories(cofeu
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(cofeu PRIVATE ${COFEU_WARNINGS} ${COFEU_OPT} ${COFEU_BASE_ARCH})
set_target_properties(cofeu PROPERTIES
    VERSION   ${PROJECT_VERSION}
    SOVERSION 2
    POSITION_INDEPENDENT_CODE ON
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(cofeu PUBLIC OpenMP::OpenMP_CXX)
endif()

# --- CLI ---
# model.cpp/tokenizer.cpp yeniden derlenmiyor, kütüphaneye bağlanıyor:
# sembol kopyasını ve iki hedefin ayrışmasını önlüyor.
add_executable(cofeu_infer src/main.cpp)
target_link_libraries(cofeu_infer PRIVATE cofeu)
target_compile_options(cofeu_infer PRIVATE ${COFEU_WARNINGS})

# Uyarıları hatadan ayır: temiz derleme hedefi
add_custom_target(strict
    COMMAND ${CMAKE_CXX_COMPILER} ${COFEU_WARNINGS} ${COFEU_OPT} ${COFEU_BASE_ARCH}
            -std=c++17 -fsyntax-only src/model.cpp src/tokenizer.cpp src/main.cpp
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Sıkı uyarı kontrolü"
    VERBATIM
)

install(TARGETS cofeu cofeu_infer
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)
install(DIRECTORY include/cofeu DESTINATION include)