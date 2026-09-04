# CofeuAI

Kendi LLM'imizi sıfırdan geliştirdiğimiz proje. Python ile eğitim, C++ ile hızlı inference, FastAPI ile web arayüzü.

## Özellikler

- 🧠 **Sıfırdan Transformer** — decoder-only (GPT benzeri) mimari, PyTorch ile eğitim
- 🔤 **BPE Tokenizer** — Byte-Pair Encoding ile alt-kelime tokenizasyonu (Python + C++)
- ⚡ **KV Cache** — otoregresif üretimde hızlandırma (Python + C++)
- 🚀 **C++ Inference** — PyTorch'suz, saf C++ ileri geçiş
- 🌐 **Web Arayüzü** — FastAPI + tarayıcıda sohbet (streaming destekli)
- 🎯 **Gelişmiş Sampling** — Top-k, Top-p (nucleus), repetition penalty
- 📊 **TensorBoard** — Eğitim takibi için görsel logging
- 💾 **Checkpoint** — Periyodik kayıt, resume, en iyi model takibi
- 🛡️ **NaN Koruması** — Attention'da güvenli softmax

## Mimari

```
CofeuAI/
├── README.md
├── requirements.txt
├── data/
│   ├── sample_corpus.txt        # Kısa örnek corpus
│   └── turkish_corpus.txt       # Wikipedia'dan Türkçe corpus
├── python/
│   ├── tokenizer.py             # BPE tokenizer (special token'lar dahil)
│   ├── model.py                 # Transformer modeli (PyTorch, KV cache, streaming)
│   ├── train.py                 # Eğitim scripti (checkpoint, resume, TensorBoard)
│   ├── generate.py              # Metin üretimi (streaming, top-k/top-p)
│   ├── export.py                # Modeli C++ için dışa aktarma
│   ├── make_corpus.py           # Daha büyük corpus oluşturma
│   ├── download_dataset.py      # Wikipedia'dan Türkçe dataset indirme
│   ├── cpp_bridge.py            # C++ kütüphanesi bridge
│   └── server.py                # FastAPI web sunucusu (SSE streaming)
├── cpp/
│   ├── CMakeLists.txt
│   ├── include/cofeu/
│   │   ├── tensor.hpp
│   │   ├── model.hpp
│   │   └── tokenizer.hpp
│   └── src/
│       ├── main.cpp
│       ├── model.cpp
│       └── tokenizer.cpp
└── scripts/
    └── run_all.sh
```

## Kurulum

```bash
# Python bağımlılıkları
pip install -r requirements.txt

# C++ build (CMake gerekli)
cd cpp && mkdir -p build && cd build && cmake .. && make
```

## Kullanım

### 1. Dataset İndirme
```bash
cd python
python download_dataset.py --max-chars 5000000
```

### 2. Eğitim
```bash
cd python
python train.py --max-iters 5000 --vocab-size 2048

# Resume ile devam et
python train.py --resume --max-iters 10000

# TensorBoard ile takip
python train.py --max-iters 5000
tensorboard --logdir checkpoints/tb_logs
```

### 3. Python ile üretim
```bash
cd python

# Normal üretim
python generate.py "Bir zamanlar"

# Streaming üretim
python generate.py "Bir zamanlar" --stream

# Sampling parametreleri ile
python generate.py "Bir zamanlar" --top-k 40 --top-p 0.9 --repetition-penalty 1.2
```

### 4. C++ ile üretim (hızlı inference)
```bash
cd cpp/build
./cofeu_infer "Bir zamanlar" 200 0.8
```

### 5. Web arayüzü
```bash
cd python
python server.py
# Tarayıcıda: http://localhost:8000
```

### 6. API Kullanımı
```bash
# Normal üretim
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Bir zamanlar", "max_tokens": 200, "temperature": 0.8}'

# Streaming üretim (SSE)
curl -N -X POST http://localhost:8000/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Bir zamanlar", "max_tokens": 200, "temperature": 0.8}'
```

## Special Token'lar

| Token | ID | Açıklama |
|-------|-----|----------|
| `<\|bos\|>` | 0 | Beginning of Sequence |
| `<\|eos\|>` | 1 | End of Sequence (stop token) |
| `<\|pad\|>` | 2 | Padding |
| `<\|unk\|>` | 3 | Bilinmeyen token |

## Eğitim Metrikleri

Eğitim sırasında şu metrikler loglanır:
- **Train/Val Loss**: Cross-entropy kaybı
- **Perplexity**: Loss'un üstel değeri
- **Token Accuracy**: Doğru tahmin oranı
- **Gradient Norm**: Gradyan büyüklüğü
- **Learning Rate**: Anlık lr değeri

## Sampling Stratejileri

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `temperature` | 0.8 | Düşük=eißsiz, Yüksek=yaratıcı |
| `top_k` | Kapalı | En yüksek k token'dan örneklem |
| `top_p` | Kapalı | Nucleus sampling (olasılık eşiği) |
| `repetition_penalty` | 1.0 | Tekrar eden token'lara ceza |

## Nasıl Çalışır?

1. **BPE Tokenizer**: Metni alt-kelime parçalarına böler + special token'lar
2. **Transformer**: Sıfırdan yazılmış decoder-only transformer (NaN korumalı attention)
3. **Eğitim**: Next-token prediction (AdamW + cosine LR + warmup + gradient clipping)
4. **KV Cache**: Üretim sırasında geçmiş K/V değerlerini cache'leyerek hızlandırır
5. **Streaming**: Token-by-token üretim (web arayüzünde anlık görünür)
6. **Export**: Eğitilen ağırlıklar düz bir binary dosyaya yazılır
7. **C++ Inference**: Ağırlıkları yükleyip saf C++ ile ileri geçiş yapar

## Sonraki Adımlar

- Daha büyük ve çeşitli Türkçe corpus (Wikipedia, kitap, haber)
- Daha büyük model (`n_embd`, `n_layer`, `n_head` artırma)
- C++ OpenMP ile matris çarpımı paralelleştirme
- int8 quantization ile daha küçük ve hızlı model
- Flash Attention entegrasyonu
- Multi-GPU (DDP) desteği
- RoPE / ALiBi position encoding
- Pre-norm (GPT-3 tarzı LayerNorm placement)
- SwiGLU MLP
- Grouped Query Attention (GQA)
