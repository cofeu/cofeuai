"""CofeuAI Eğitim Scripti.

Örnek corpus üzerinde transformer modelini eğitir ve kaydeder.


"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from model import CofeuTransformer, ModelConfig
from tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "turkish_corpus.txt"
OUT_DIR = ROOT / "checkpoints"


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    """Rastgele bir batch üretir."""
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: CofeuTransformer,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str,
    eval_iters: int = 20,
) -> float:
    """Ortalama kaybı tahmin eder."""
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    parser = argparse.ArgumentParser(description="CofeuAI modelini eğit")
    parser.add_argument("--max-iters", type=int, default=5000, help="Eğitim iterasyon sayısı")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--vocab-size", type=int, default=2048, help="BPE vocab boyutu")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--warmup-iters", type=int, default=500, help="LR warmup iterasyonu")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation adımı")
    parser.add_argument("--amp", action="store_true", help="Mixed precision (float16) eğitim")
    args = parser.parse_args()

    device = (
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Cihaz: {device}")

    # Veriyi yükle
    text = DATA_PATH.read_text(encoding="utf-8")
    print(f"Veri yüklendi: {len(text):,} karakter")
    print("BPE tokenizer eğitiliyor...")
    tokenizer = BPETokenizer.build(text, vocab_size=args.vocab_size)
    print(f"Vocab boyutu: {tokenizer.vocab_size}")

    # Vocab'ı kaydet ve C++ tokenizer ile hızlı encode et
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(OUT_DIR / "vocab.json")
    print("Metin tokenize ediliyor (C++ ile)...")
    try:
        from cpp_bridge import CppTokenizer
        cpp_tok = CppTokenizer(OUT_DIR / "vocab.json")
        ids = cpp_tok.encode(text)
        data = torch.tensor(ids, dtype=torch.long)
        print(f"  C++ tokenizer kullanıldı")
    except Exception as e:
        print(f"  C++ tokenizer kullanılamadı ({e}), Python'a düşülüyor...")
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"Toplam token: {len(data):,}")

    # Train/val ayrımı
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    config = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = CofeuTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parametre sayısı: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    # Mixed precision (AMP) — RTX 3060'ta float16 ile ~2x hız
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device == "cuda"))
    if args.amp and device == "cuda":
        print("Mixed precision (AMP) etkin")

    # Cosine learning rate scheduler (warmup ile)
    def get_lr(it: int) -> float:
        if it < args.warmup_iters:
            return args.lr * (it + 1) / args.warmup_iters
        if it > args.max_iters:
            return 0.0
        decay_ratio = (it - args.warmup_iters) / (args.max_iters - args.warmup_iters)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    start = time.time()

    for it in range(args.max_iters):
        # Learning rate güncelle
        lr = get_lr(it)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if it % args.eval_interval == 0:
            train_loss = estimate_loss(model, train_data, config.block_size, args.batch_size, device)
            val_loss = estimate_loss(model, val_data, config.block_size, args.batch_size, device)
            elapsed = time.time() - start
            print(
                f"iter {it:5d} | lr {lr:.6f} | train loss {train_loss:.4f} | "
                f"val loss {val_loss:.4f} | {elapsed:.1f}s"
            )

        # Gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        for micro in range(args.grad_accum):
            xb, yb = get_batch(train_data, config.block_size, args.batch_size, device)
            if args.amp and device == "cuda":
                with torch.amp.autocast("cuda"):
                    _, loss = model(xb, yb)
                loss = loss / args.grad_accum
                scaler.scale(loss).backward()
            else:
                _, loss = model(xb, yb)
                loss = loss / args.grad_accum
                loss.backward()

        # Gradient clipping (foreach=False: bazı CUDA sürümlerinde foreach=True
        # "illegal instruction" hatasına neden olabiliyor)
        if args.amp and device == "cuda":
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, foreach=False)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, foreach=False)
            optimizer.step()

    # Son kayıp
    final_loss = estimate_loss(model, val_data, config.block_size, args.batch_size, device)
    print(f"\nEğitim tamamlandı. Final val loss: {final_loss:.4f}")

    # Kaydet
    ckpt = {
        "config": config,
        "model_state": model.state_dict(),
        "vocab": tokenizer.stoi,
        "merges": [f"{a} {b}" for a, b in tokenizer.merges],
        "final_loss": final_loss,
    }
    torch.save(ckpt, OUT_DIR / "cofeu.pt")
    tokenizer.save(OUT_DIR / "vocab.json")
    print(f"Model kaydedildi: {OUT_DIR / 'cofeu.pt'}")
    print(f"Vocab kaydedildi: {OUT_DIR / 'vocab.json'}")


if __name__ == "__main__":
    main()