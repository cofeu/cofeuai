#!/usr/bin/env python3
"""Instruction tuning (SFT) - Alpaca biçimli veri seti ile.

Temel eğitimden (train.py) farkı: tüm token'lar eşit ağırlıkla değil, SADECE
asistan cevabı kayıp hesabına girer. Prompt token'ları -100 ile maskelenir
(PyTorch'in ignore_index'i), böylece model cevabı üretmeyi öğrenir.

Kullanım:
    python sft.py --data alpaca.jsonl --resume --epochs 2
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from model import CofeuTransformer, ModelConfig
from tokenizer import BPETokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cofeu.sft")

ROOT = Path(__file__).resolve().parent.parent
IGNORE = -100

SYSTEM_PROMPT = "Sen yardımcı bir yapay zeka asistanısın."


def build_prompt(instruction: str, inp: str) -> str:
    """Kullanıcı kısmını kurar (modelin göreceği bağlam)."""
    if inp and inp.strip():
        return f"{instruction}\n\n{inp.strip()}"
    return instruction.strip()


def build_text(instruction: str, inp: str, output: str) -> Tuple[str, str]:
    """(prompt, tam_metin) döndürür.

    tam_metin, modelin öğrenmesi gereken tam konuşma. prompt, cevabın
    başladığı yere kadar olan kısım ve maskelenir.

    NOT: Cevap sonundaki <|eos|>, encode sırasında metin olarak değil
    TEK TOKEN olarak eklenmelidir; encode() özel token'ları tanımaz,
    "<|eos|>" yazısını unk + parçalara böler.
    """
    p = build_prompt(instruction, inp)
    prompt_part = (
        f"### Sistem:\n{SYSTEM_PROMPT}\n\n"
        f"### Kullanıcı:\n{p}\n\n"
        f"### Asistan:\n"
    )
    return prompt_part, prompt_part + f"{output.strip()}"


def encode_example(tok, instruction: str, inp: str, output: str,
                   block_size: int) -> Optional[Tuple[List[int], List[int]]]:
    """Bir örneği (ids, labels) çevirir. Block'a sığmıyorsa None.

    Cevabı EOS ile kapatır: model böylece "dur" işaretini öğrenir. Ham metin
    eğitiminde EOS hiç görülmediği için bu kritik.
    """
    prompt_part, full = build_text(instruction, inp, output)
    full_ids = tok.encode(full)
    if tok.eos_id >= 0:
        full_ids = full_ids + [tok.eos_id]
    if len(full_ids) > block_size:
        return None
    prompt_len = len(tok.encode(prompt_part))
    labels = list(full_ids)
    for i in range(min(prompt_len, len(labels))):
        labels[i] = IGNORE
    return full_ids, labels


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({
                "instruction": obj.get("instruction", ""),
                "input": obj.get("input", ""),
                "output": obj.get("output", ""),
            })
    return rows


def make_batches(samples, batch_size: int, block_size: int, rng: random.Random):
    """samples: (ids, labels) listesi -> (x, y) tensor batch'leri."""
    idxs = list(range(len(samples)))
    rng.shuffle(idxs)
    for b in range(0, len(idxs), batch_size):
        chunk = [samples[i] for i in idxs[b:b + batch_size]]
        pad = tok_pad_id_holder[0]
        maxlen = max(len(ids) for ids, _ in chunk)
        maxlen = min(maxlen, block_size)
        x = torch.full((len(chunk), maxlen), pad, dtype=torch.long)
        y = torch.full((len(chunk), maxlen), IGNORE, dtype=torch.long)
        for j, (ids, labels) in enumerate(chunk):
            n = min(len(ids), maxlen)
            x[j, :n] = torch.tensor(ids[:n], dtype=torch.long)
            y[j, :n] = torch.tensor(labels[:n], dtype=torch.long)
        yield x, y


tok_pad_id_holder = [2]


def evaluate(model, samples, batch_size: int, block_size: int, device: str,
             max_batches: int = 20) -> float:
    model.eval()
    rng = random.Random(1234)
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for bi, (x, y) in enumerate(make_batches(samples, batch_size, block_size, rng)):
            if bi >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            _, loss = model(x, targets=y)
            n = (y != IGNORE).sum().item()
            if n > 0:
                total_loss += loss.item() * n
                total_n += n
    model.train()
    return total_loss / max(total_n, 1)


def main():
    ap = argparse.ArgumentParser(description="Instruction tuning (SFT)")
    ap.add_argument("--data", type=str, required=True, help="JSONL veri dosyası")
    ap.add_argument("--out-dir", type=str, default=str(ROOT / "checkpoints" / "sft"))
    ap.add_argument("--resume", type=str, default=str(ROOT / "checkpoints" / "cofeu.pt"),
                    help="Başlangıç checkpoint'i (temel eğitim çıktısı)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4, help="SFT için daha düşük lr")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--max-iters", type=int, default=0, help="0 = epoch sayısı kadar")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--log-interval", type=int, default=25)
    ap.add_argument("--patience", type=int, default=3,
                    help="Val iyilesmezse kac eval sonra durulur (0 = devam et)")
    ap.add_argument("--eval-interval", type=int, default=200)
    ap.add_argument("--save-interval", type=int, default=400)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = tümü")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Cihaz: %s", device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer: mevcut vocab'ı kullan (yeniden eğitme)
    vocab_path = ROOT / "checkpoints" / "vocab.json"
    logger.info("Vocab yükleniyor: %s", vocab_path)
    tok = BPETokenizer.load(vocab_path)
    tok_pad_id_holder[0] = tok.pad_id
    logger.info("Vocab: %d token, eos=%d bos=%d", tok.vocab_size, tok.eos_id, tok.bos_id)

    # Veri
    rows = load_jsonl(Path(args.data))
    if args.max_samples:
        rows = rows[:args.max_samples]
    logger.info("Ham örnek: %d", len(rows))

    resume_path = Path(args.resume)
    if not resume_path.exists():
        logger.error("Checkpoint bulunamadı: %s", resume_path)
        return
    ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
    state_key = "model_state" if "model_state" in ckpt else "model"
    cfg = ModelConfig(**ckpt["config"].__dict__) if isinstance(ckpt["config"], ModelConfig) \
        else ModelConfig(**ckpt["config"])
    block_size = cfg.block_size
    logger.info("Model config: vocab=%d embd=%d layer=%d block=%d",
                cfg.vocab_size, cfg.n_embd, cfg.n_layer, block_size)

    # Encode
    samples = []
    skipped = 0
    for r in rows:
        enc = encode_example(tok, r["instruction"], r["input"], r["output"], block_size)
        if enc is None:
            skipped += 1
            continue
        samples.append(enc)
    logger.info("Kullanılabilir: %d (block=%d'a sığmayan: %d)", len(samples), block_size, skipped)

    if not samples:
        logger.error("Kullanılabilir örnek yok!")
        return

    # Model
    model = CofeuTransformer(cfg).to(device)
    model.load_state_dict(ckpt[state_key])
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model yüklendi: %.2fM parametre", n_params / 1e6)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    steps_per_epoch = math.ceil(len(samples) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    if args.max_iters:
        total_steps = min(total_steps, args.max_iters)
    warmup = min(100, total_steps // 10)

    def get_lr(it: int) -> float:
        if it < warmup:
            return args.lr * (it + 1) / max(warmup, 1)
        prog = (it - warmup) / max(total_steps - warmup, 1)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    # Val: son %5
    n_val = max(len(samples) // 20, 1)
    val_samples = samples[-n_val:]
    train_samples = samples[:-n_val]
    logger.info("SFT train: %d, val: %d", len(train_samples), len(val_samples))

    logger.info("Eğitim başlıyor: %d adım (%d epoch x %d adım)", total_steps, args.epochs, steps_per_epoch)
    step = 0
    t0 = time.time()
    best_val = float("inf")
    evals_since_improve = 0
    stopped_early = False

    for epoch in range(args.epochs):
        for x, y in make_batches(train_samples, args.batch_size, block_size, rng):
            if step >= total_steps or stopped_early:
                break
            for g in opt.param_groups:
                g["lr"] = get_lr(step)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, targets=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_interval == 0:
                elapsed = time.time() - t0
                ppl = math.exp(min(loss.item(), 20))
                logger.info(
                    "step %5d/%d | loss %.4f | ppl %7.2f | lr %.2e | %.2f tok/s",
                    step, total_steps, loss.item(), ppl, get_lr(step),
                    step * x.numel() / elapsed,
                )
            if step % args.eval_interval == 0:
                vl = evaluate(model, val_samples, args.batch_size, block_size, device)
                logger.info("  >> VAL loss %.4f (ppl %.2f)", vl, math.exp(min(vl, 20)))
                if vl < best_val:
                    best_val = vl
                    evals_since_improve = 0
                    torch.save({
                        "model_state": model.state_dict(),
                        "config": cfg,
                        "iteration": step,
                        "val_loss": vl,
                        "sft": True,
                    }, out_dir / "cofeu_sft.pt")
                    logger.info("  >> kaydedildi: cofeu_sft.pt")
                else:
                    # Val arttı -> ezberleme başladı. En iyi model elimizde,
                    # daha fazla eğitim sadece bozar.
                    evals_since_improve += 1
                    logger.info("  >> val arttı (%d/%d), best: %.4f",
                                evals_since_improve, args.patience, best_val)
                    if args.patience and evals_since_improve >= args.patience:
                        logger.info("ERKEN DURDURMA: val %d kez iyilesmedi. "
                                    "En iyi model kaydedildi.", args.patience)
                        stopped_early = True
                        break
            if step % args.save_interval == 0:
                torch.save({
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "iteration": step,
                    "sft": True,
                }, out_dir / "cofeu_sft_latest.pt")
        if step >= total_steps or stopped_early:
            break

    vl = evaluate(model, val_samples, args.batch_size, block_size, device)
    logger.info("SON val loss %.4f (ppl %.2f)", vl, math.exp(min(vl, 20)))

    torch.save({"model_state": model.state_dict(), "config": cfg, "iteration": step,
                "val_loss": vl, "sft": True}, out_dir / "cofeu_sft.pt")
    logger.info("Bitti -> %s", out_dir / "cofeu_sft.pt")


if __name__ == "__main__":
    main()
