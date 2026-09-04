"""CofeuAI Eğitim Scripti.

Örnek corpus üzerinde transformer modelini eğitir ve kaydeder.

Özellikler:
  - Checkpoint kaydetme ve resume
  - Validation metrics (loss, perplexity, token accuracy)
  - TensorBoard logging
  - Auto AMP (mixed precision) detection
  - Gradient clipping ve warmup
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import torch

from model import CofeuTransformer, ModelConfig
from tokenizer import BPETokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cofeu.train")

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
) -> dict:
    """Ortalama kaybı ve ek metrikleri hesaplar."""
    model.eval()
    losses = torch.zeros(eval_iters)
    correct_tokens = 0
    total_tokens = 0

    for k in range(eval_iters):
        x, y = get_batch(data, block_size, batch_size, device)
        logits, loss = model(x, y)
        losses[k] = loss.item()

        # Token accuracy
        predictions = logits.argmax(dim=-1)
        mask = y != -100
        correct_tokens += ((predictions == y) & mask).sum().item()
        total_tokens += mask.sum().item()

    model.train()

    avg_loss = losses.mean().item()
    perplexity = math.exp(min(avg_loss, 20))  # exp(clamp) ile overflow koruması
    accuracy = correct_tokens / max(total_tokens, 1)

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "accuracy": accuracy,
    }


def save_checkpoint(
    model: CofeuTransformer,
    optimizer: torch.optim.Optimizer,
    config: ModelConfig,
    tokenizer: BPETokenizer,
    iteration: int,
    val_loss: float,
    train_metrics: dict,
    path: Path,
):
    """Checkpoint kaydeder."""
    ckpt = {
        "config": config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
        "val_loss": val_loss,
        "train_metrics": train_metrics,
        "vocab": tokenizer.stoi,
        "merges": [f"{a} {b}" for a, b in tokenizer.merges],
        "special_tokens": tokenizer.special_tokens,
    }
    torch.save(ckpt, path)
    logger.info("Checkpoint kaydedildi: %s (iter %d)", path, iteration)


def load_checkpoint(path: Path, model: CofeuTransformer, optimizer: torch.optim.Optimizer):
    """Checkpoint yükler, resume için."""
    if not path.exists():
        return 0, {}

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if "optimizer_state" in ckpt and optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    iteration = ckpt.get("iteration", 0)
    meta = {
        "val_loss": ckpt.get("val_loss", float("inf")),
        "train_metrics": ckpt.get("train_metrics", {}),
    }
    logger.info("Checkpoint yüklendi: %s (iter %d)", path, iteration)
    return iteration, meta


def setup_tensorboard(log_dir: Path):
    """TensorBoard writer'ı kurar."""
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        logger.info("TensorBoard log dizini: %s", log_dir)
        return writer
    except ImportError:
        logger.warning("TensorBoard yüklü değil. pip install tensorboard ile kurabilirsiniz.")
        return None


def detect_amp(device: str, force_amp: bool = False) -> bool:
    """AMP kullanımı için otomatik tespit."""
    if force_amp:
        if device == "cuda" and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            # Volta ve üzeri (SM >= 7.0) float16 destekler
            if capability[0] >= 7:
                logger.info("AMP zorlandı (SM %d.%d)", capability[0], capability[1])
                return True
            else:
                logger.warning("GPU SM %d.%d - float16 desteklenmiyor, AMP atlanıyor", capability[0], capability[1])
                return False
        else:
            logger.warning("AMP sadece CUDA cihazlarda çalışır")
            return False

    # Otomatik tespit
    if device == "cuda" and torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 7:
            logger.info("AMP otomatik etkinleştirildi (SM %d.%d)", capability[0], capability[1])
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="CofeuAI modelini eğit")
    parser.add_argument("--max-iters", type=int, default=5000, help="Eğitim iterasyon sayısı")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=1000, help="Checkpoint kayıt aralığı")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--vocab-size", type=int, default=2048, help="BPE vocab boyutu")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--warmup-iters", type=int, default=500, help="LR warmup iterasyonu")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation adımı")
    parser.add_argument("--amp", action="store_true", help="Mixed precision (float16) eğitim")
    parser.add_argument("--no-amp", action="store_true", help="AMP'yi devre dışı bırak")
    parser.add_argument("--resume", action="store_true", help="Son checkpoint'tan devam et")
    parser.add_argument("--tb-log-dir", type=str, default=None, help="TensorBoard log dizini")
    parser.add_argument("--eval-iters", type=int, default=20, help="Eval iterasyon sayısı")
    args = parser.parse_args()

    device = (
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Cihaz: %s", device)

    # Veriyi yükle
    if not DATA_PATH.exists():
        logger.error("Veri dosyası bulunamadı: %s", DATA_PATH)
        logger.info("Önce corpus oluşturun: python make_corpus.py")
        return

    text = DATA_PATH.read_text(encoding="utf-8")
    logger.info("Veri yüklendi: %s karakter", f"{len(text):,}")

    logger.info("BPE tokenizer eğitiliyor...")
    tokenizer = BPETokenizer.build(text, vocab_size=args.vocab_size)
    logger.info("Vocab boyutu: %d", tokenizer.vocab_size)
    logger.info("Special token'lar: %s", tokenizer.special_tokens)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(OUT_DIR / "vocab.json")

    logger.info("Metin tokenize ediliyor...")
    try:
        from cpp_bridge import CppTokenizer
        cpp_tok = CppTokenizer(OUT_DIR / "vocab.json")
        ids = cpp_tok.encode(text)
        data = torch.tensor(ids, dtype=torch.long)
        logger.info("C++ tokenizer kullanıldı")
    except Exception as e:
        logger.warning("C++ tokenizer kullanılamadı (%s), Python'a düşülüyor", e)
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    logger.info("Toplam token: %s", f"{len(data):,}")

    # Train/val ayrımı (son %10 validation)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    logger.info("Train: %s token, Val: %s token", f"{len(train_data):,}", f"{len(val_data):,}")

    # Config
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        eos_token_id=tokenizer.eos_id,
    )
    model = CofeuTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Parametre sayısı: %s (%.2fM)", f"{n_params:,}", n_params / 1e6)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    # AMP
    use_amp = detect_amp(device, force_amp=args.amp)
    if args.no_amp:
        use_amp = False
        logger.info("AMP manuel olarak devre dışı bırakıldı")

    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device == "cuda"))

    # Resume
    start_iter = 0
    if args.resume:
        start_iter, meta = load_checkpoint(OUT_DIR / "cofeu_latest.pt", model, optimizer)
        logger.info("İterasyon %d'den devam ediliyor", start_iter)

    # TensorBoard
    tb_log_dir = Path(args.tb_log_dir) if args.tb_log_dir else OUT_DIR / "tb_logs"
    writer = setup_tensorboard(tb_log_dir)

    # Cosine learning rate scheduler
    def get_lr(it: int) -> float:
        if it < args.warmup_iters:
            return args.lr * (it + 1) / args.warmup_iters
        if it > args.max_iters:
            return 0.0
        decay_ratio = (it - args.warmup_iters) / (args.max_iters - args.warmup_iters)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    # Metrics log dosyası
    metrics_path = OUT_DIR / "training_metrics.jsonl"

    start_time = time.time()
    best_val_loss = float("inf")

    logger.info("Eğitim başlıyor: %d iterasyon, lr=%.1e, batch=%d", args.max_iters, args.lr, args.batch_size)
    logger.info("Eval her %d iterasyonda, checkpoint her %d'de", args.eval_interval, args.save_interval)

    for it in range(start_iter, args.max_iters):
        lr = get_lr(it)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Eval
        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            train_metrics = estimate_loss(model, train_data, config.block_size, args.batch_size, device, args.eval_iters)
            val_metrics = estimate_loss(model, val_data, config.block_size, args.batch_size, device, args.eval_iters)
            elapsed = time.time() - start_time
            iters_per_sec = (it - start_iter + 1) / max(elapsed, 1e-6)

            logger.info(
                "iter %5d | lr %.6f | train loss %.4f (ppl %.2f, acc %.2f%%) | "
                "val loss %.4f (ppl %.2f, acc %.2f%%) | %.1fs | %.1f it/s",
                it, lr,
                train_metrics["loss"], train_metrics["perplexity"], train_metrics["accuracy"] * 100,
                val_metrics["loss"], val_metrics["perplexity"], val_metrics["accuracy"] * 100,
                elapsed, iters_per_sec,
            )

            # TensorBoard
            if writer is not None:
                writer.add_scalar("train/loss", train_metrics["loss"], it)
                writer.add_scalar("train/perplexity", train_metrics["perplexity"], it)
                writer.add_scalar("train/accuracy", train_metrics["accuracy"], it)
                writer.add_scalar("val/loss", val_metrics["loss"], it)
                writer.add_scalar("val/perplexity", val_metrics["perplexity"], it)
                writer.add_scalar("val/accuracy", val_metrics["accuracy"], it)
                writer.add_scalar("lr", lr, it)

            # JSONL log
            log_entry = {
                "iter": it,
                "lr": lr,
                "train_loss": train_metrics["loss"],
                "train_ppl": train_metrics["perplexity"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_ppl": val_metrics["perplexity"],
                "val_acc": val_metrics["accuracy"],
                "elapsed": elapsed,
            }
            with open(metrics_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            # En iyi model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(model, optimizer, config, tokenizer, it, val_metrics["loss"], train_metrics, OUT_DIR / "cofeu_best.pt")

        # Gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        for micro in range(args.grad_accum):
            xb, yb = get_batch(train_data, config.block_size, args.batch_size, device)
            if use_amp and device == "cuda":
                with torch.amp.autocast("cuda"):
                    _, loss = model(xb, yb)
                loss = loss / args.grad_accum
                scaler.scale(loss).backward()
            else:
                _, loss = model(xb, yb)
                loss = loss / args.grad_accum
                loss.backward()

        # Gradient clipping
        if use_amp and device == "cuda":
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, foreach=False)
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, foreach=False)
            optimizer.step()

        # Gradient norm TensorBoard'a yaz
        if writer is not None and it % 100 == 0:
            writer.add_scalar("train/grad_norm", grad_norm.item(), it)

        # Periyodik checkpoint
        if it > 0 and it % args.save_interval == 0:
            save_checkpoint(model, optimizer, config, tokenizer, it, best_val_loss, {}, OUT_DIR / "cofeu_latest.pt")

    # Son checkpoint
    final_metrics = estimate_loss(model, val_data, config.block_size, args.batch_size, device, args.eval_iters)
    logger.info(
        "Eğitim tamamlandı. Final val loss: %.4f (ppl %.2f, acc %.2f%%)",
        final_metrics["loss"], final_metrics["perplexity"], final_metrics["accuracy"] * 100,
    )

    save_checkpoint(model, optimizer, config, tokenizer, args.max_iters, final_metrics["loss"], final_metrics, OUT_DIR / "cofeu.pt")
    tokenizer.save(OUT_DIR / "vocab.json")

    if writer is not None:
        writer.close()

    logger.info("Model kaydedildi: %s", OUT_DIR / "cofeu.pt")
    logger.info("Vocab kaydedildi: %s", OUT_DIR / "vocab.json")


if __name__ == "__main__":
    main()
