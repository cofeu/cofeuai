# CofeuAI

Kendi LLM'imizi sıfırdan geliştirdiğimiz proje. Python ile eğitim, C++ ile hızlı inference, FastAPI ile web arayüzü.

## Özellikler

- 🧠 **Sıfırdan Transformer** — decoder-only (GPT benzeri) mimari, PyTorch ile eğitim
- 🔤 **BPE Tokenizer** — Byte-Pair Encoding ile alt-kelime tokenizasyonu (Python + C++)
- ⚡ **KV Cache** — otoregresif üretimde hızlandırma (Python + C++)
- 🚀 **C++ Inference** — PyTorch'suz, saf C++ ileri geçiş
- 🌐 **Web Arayüzü** — FastAPI + tarayıcıda sohbet

## Mimari

```
CofeuAI/
├── README.md
├── requirements.txt
├── data/
│   └── sample_corpus.txt        # Eğitim verisi (Türkçe hikayeler)
├── python/
│   ├── tokenizer.py             # BPE tokenizer
│   ├── model.py                 # Transformer modeli (PyTorch, KV cache)
│   ├── train.py                 # Eğitim scripti
│   ├── generate.py              # Metin üretimi (Python)
│   ├── export.py                # Modeli C++ için dışa aktarma
│   ├── make_corpus.py           # Daha büyük corpus oluşturma
│   └── server.py                # FastAPI web sunucusu
├── cpp/
│   ├── CMakeLists.txt
│   ├── include/cofeu/
│   │   ├── tensor.hpp           # Basit tensor yapısı
│   │   ├── model.hpp            # Transformer inference (KV cache)
│   │   └── tokenizer.hpp        # BPE tokenizer
│   └── src/
│       ├── main.cpp             # CLI inference aracı
│       ├── model.cpp
│       └── tokenizer.cpp
└── scripts/
    └── run_all.sh               # Eğitim + export + C++ build
```

## Kurulum

```bash
# Python bağımlılıkları
pip install -r requirements.txt

# C++ build (CMake gerekli)
cd cpp && mkdir -p build && cd build && cmake .. && make
```

## Kullanım

### 1. Eğitim
```bash
cd python
python train.py --max-iters 2000 --vocab-size 256
```

### 2. Python ile üretim
```bash
cd python
python generate.py "Bir zamanlar"
```

### 3. C++ ile üretim (hızlı inference)
```bash
cd cpp/build
./cofeu_infer "Bir zamanlar" 200 0.8
```

### 4. Web arayüzü
```bash
cd python
python server.py
# Tarayıcıda: http://localhost:8000
```

## Nasıl Çalışır?

1. **BPE Tokenizer**: Metni alt-kelime parçalarına böler (karakter seviyesinden daha verimli).
2. **Transformer**: Sıfırdan yazılmış decoder-only transformer (causal attention, LayerNorm, GELU MLP).
3. **Eğitim**: `data/sample_corpus.txt` üzerinde next-token prediction (AdamW).
4. **KV Cache**: Üretim sırasında geçmiş K/V değerlerini cache'leyerek hızlandırır.
5. **Export**: Eğitilen ağırlıklar düz bir binary dosyaya yazılır.
6. **C++ Inference**: Ağırlıkları yükleyip saf C++ ile ileri geçiş yapar (PyTorch'suz, hızlı).

## Sonraki Adımlar

- Daha büyük ve çeşitli Türkçe corpus (Wikipedia, kitap, haber)
- Daha büyük model (`n_embd`, `n_layer`, `n_head` artırma)
- C++ OpenMP ile matris çarpımı paralelleştirme
- int8 quantization ile daha küçük ve hızlı model
- Learning rate scheduler ve gradient clipping