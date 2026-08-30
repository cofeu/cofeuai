#!/usr/bin/env bash
# CofeuAI: eğitim -> export -> C++ build -> test
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== 1. Eğitim ==="
cd python
python train.py
cd ..

echo ""
echo "=== 2. Model Dışa Aktarma ==="
cd python
python export.py
cd ..

echo ""
echo "=== 3. C++ Build ==="
cd cpp
mkdir -p build
cd build
cmake .. > /dev/null
make
cd ../..

echo ""
echo "=== 4. C++ Inference Testi ==="
# vocab.json ve cofeu.bin'i build dizinine kopyala
cp checkpoints/vocab.json cpp/build/
cp checkpoints/cofeu.bin cpp/build/
cd cpp/build
./cofeu_infer "Bir zamanlar" 200 0.8
cd ../..

echo ""
echo "Tamamlandı!"