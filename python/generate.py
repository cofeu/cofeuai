"""CofeuAI Metin Üretimi.

Eğitilmiş modeli yükleyip metin üretir. C++ inference motoru varsa onu
kullanır (çok daha hızlı), yoksa Python'a düşer.

Özellikler:
  - Top-k, top-p, repetition penalty sampling
  - Token-by-token streaming üretim
  - EOS (stop token) desteği
  - Structured logging
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Generator

import torch

from model import CofeuTransformer, ModelConfig
from tokenizer import BPETokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cofeu.generate")

ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = ROOT / "checkpoints" / "cofeu.pt"
VOCAB_PATH = ROOT / "checkpoints" / "vocab.json"
BIN_PATH = ROOT / "checkpoints" / "cofeu.bin"


def generate_cpp(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    stream: bool = False,
) -> str | None:
    """C++ inference motoru ile üretim."""
    try:
        from cpp_bridge import CppModel, CppTokenizer

        tok = CppTokenizer(VOCAB_PATH)
        model = CppModel(BIN_PATH)
        prompt_ids = tok.encode(prompt)
        out_ids = model.generate(prompt_ids, max_tokens, temperature)
        return tok.decode(out_ids)
    except Exception as e:
        logger.warning("C++ kullanılamadı: %s", e)
        return None


def generate_python(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    stream: bool = False,
) -> str | Generator[str, None, None]:
    """Python (PyTorch) ile üretim."""
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Model bulunamadı: {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    config: ModelConfig = ckpt["config"]
    tokenizer = BPETokenizer(
        merges=ckpt.get("merges", []),
        vocab=ckpt["vocab"],
        special_tokens=ckpt.get("special_tokens"),
    )

    model = CofeuTransformer(config)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    prompt_ids = tokenizer.encode(prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long)

    eos_id = tokenizer.eos_id if tokenizer.eos_id >= 0 else config.eos_token_id

    if stream:
        return _stream_python(model, tokenizer, idx, max_tokens, temperature, top_k, top_p, repetition_penalty, eos_id)
    else:
        out = model.generate(
            idx, max_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
            eos_token_id=eos_id,
        )
        return tokenizer.decode(out[0].tolist())


def _stream_python(
    model: CofeuTransformer,
    tokenizer: BPETokenizer,
    idx: torch.Tensor,
    max_tokens: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float,
    eos_id: int,
) -> Generator[str, None, None]:
    """Token-by-token streaming üretim."""
    for token_tensor in model.generate_stream(
        idx, max_tokens, temperature=temperature,
        top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
        eos_token_id=eos_id,
    ):
        token_id = token_tensor.item()
        decoded = tokenizer.decode([token_id])
        yield decoded


def main():
    parser = argparse.ArgumentParser(description="CofeuAI ile metin üret")
    parser.add_argument("prompt", type=str, help="Başlangıç metni")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling (varsayılan: tümü)")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p (nucleus) sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Tekrar cezası (1.0=kapalı)")
    parser.add_argument("--stream", action="store_true", help="Token-by-token streaming çıktı")
    parser.add_argument("--python", action="store_true", help="Python ile üret (C++ yerine)")
    args = parser.parse_args()

    if not CKPT_PATH.exists():
        logger.error("Model bulunamadı: %s", CKPT_PATH)
        logger.info("Önce eğitim yapın: python train.py")
        return

    logger.info(
        "Üretim: prompt='%s', max_tokens=%d, temp=%.2f, top_k=%s, top_p=%s, rep_penalty=%.2f",
        args.prompt, args.max_tokens, args.temperature,
        args.top_k, args.top_p, args.repetition_penalty,
    )

    start_time = time.time()

    if not args.python and BIN_PATH.exists():
        text = generate_cpp(
            args.prompt, args.max_tokens, args.temperature,
            args.top_k, args.top_p, args.repetition_penalty,
        )
        if text is not None:
            elapsed = time.time() - start_time
            logger.info("C++ ile üretim tamamlandı (%.2fs)", elapsed)
            print(text)
            return

    if args.stream:
        logger.info("Streaming üretim başlatılıyor...")
        gen = generate_python(
            args.prompt, args.max_tokens, args.temperature,
            args.top_k, args.top_p, args.repetition_penalty,
            stream=True,
        )
        token_count = 0
        try:
            for chunk in gen:
                print(chunk, end="", flush=True)
                token_count += 1
            print()  # Yeni satır
        except KeyboardInterrupt:
            print("\nÜretim kesildi.")
        elapsed = time.time() - start_time
        logger.info("Streaming üretim tamamlandı: %d token (%.2fs)", token_count, elapsed)
    else:
        text = generate_python(
            args.prompt, args.max_tokens, args.temperature,
            args.top_k, args.top_p, args.repetition_penalty,
        )
        elapsed = time.time() - start_time
        logger.info("Üretim tamamlandı (%.2fs)", elapsed)
        print(text)


if __name__ == "__main__":
    main()
