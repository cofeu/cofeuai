"""CofeuAI Metin Üretimi.

Eğitilmiş modeli yükleyip metin üretir. C++ inference motoru varsa onu
kullanır (çok daha hızlı), yoksa PyTorch'a düşer.

Özellikler:
  - Top-k, top-p, repetition penalty sampling
  - Token-by-token streaming (her iki motor için de)
  - EOS (stop token) desteği
  - --seed ile tekrarlanabilir çıktı
  - Structured logging

Kullanım:
    cd python
    ../.venv/bin/python generate.py "Bir zamanlar" --stream
    ../.venv/bin/python generate.py "Merhaba" --temperature 0.7 --top-k 40 --top-p 0.95
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Iterator, Optional

import runtime
from runtime import Backend, ModelLoadError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cofeu.generate")


def _stream_text(
    backend: Backend,
    prompt_ids: list[int],
    max_tokens: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    repetition_penalty: float,
    seed: int,
) -> Iterator[str]:
    """Token id'lerini çözerek parça parça metin üretir."""
    eos = backend.tokenizer.eos_id
    for token_id in runtime.stream_ids(
        backend, prompt_ids, max_tokens,
        temperature=temperature, top_k=top_k, top_p=top_p,
        repetition_penalty=repetition_penalty, eos_token_id=eos, seed=seed,
    ):
        yield backend.tokenizer.decode([token_id])


def main() -> int:
    parser = argparse.ArgumentParser(description="CofeuAI ile metin üret")
    parser.add_argument("prompt", type=str, help="Başlangıç metni")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="0 -> greedy, >0 -> örnekleme")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Top-k sampling (varsayılan: tümü)")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Top-p (nucleus) sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Tekrar cezası (1.0=kapalı)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Rastgelelik tohumu (0 = gerçek rastgele)")
    parser.add_argument("--stream", action="store_true",
                        help="Token-by-token streaming çıktı")
    parser.add_argument("--python", action="store_true",
                        help="PyTorch ile üret (C++ motoru yerine)")
    parser.add_argument("--cuda", action="store_true",
                        help="PyTorch yolunda GPU kullan")
    parser.add_argument("--chat", action="store_true",
                        help="SFT modeli: konuşma şablonuna sarar, "
                             "Asistan kısmını yazdırmaz")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Model yolu (SFT için: checkpoints/sft/cofeu_sft.pt)")
    args = parser.parse_args()

    if args.max_tokens < 1:
        logger.error("--max-tokens pozitif olmalı")
        return 2

    try:
        backend = runtime.load_backend(
            prefer_cpp=not args.python,
            device="cuda" if args.cuda else "cpu",
            ckpt_path=Path(args.ckpt) if args.ckpt else None,
        )
    except ModelLoadError as e:
        logger.error("%s", e)
        return 1

    raw_prompt = args.prompt
    if args.chat:
        from sft import build_prompt, SYSTEM_PROMPT
        args.prompt = (
            f"### Sistem:\n{SYSTEM_PROMPT}\n\n"
            f"### Kullanıcı:\n{build_prompt(raw_prompt, '')}\n\n"
            f"### Asistan:\n"
        )

    with backend:
        prompt_ids = backend.tokenizer.encode(args.prompt)
        if not prompt_ids:
            logger.error("Prompt tokenize edilemedi (boş veya bilinmeyen karakterler)")
            return 1

        logger.info(
            "Motor: %s | prompt %d token | max_tokens=%d temp=%.2f top_k=%s "
            "top_p=%s rep_penalty=%.2f seed=%d",
            backend.name, len(prompt_ids), args.max_tokens, args.temperature,
            args.top_k, args.top_p, args.repetition_penalty, args.seed,
        )
        if backend.uses_cpp:
            logger.info("  V=%d block=%d max_pos=%d",
                        backend.model.vocab_size,
                        backend.model.block_size,
                        backend.model.max_position)

        start = time.time()

        if args.stream:
            count = 0
            try:
                for chunk in _stream_text(
                    backend, prompt_ids, args.max_tokens, args.temperature,
                    args.top_k, args.top_p, args.repetition_penalty, args.seed,
                ):
                    print(chunk, end="", flush=True)
                    count += 1
                print()
            except KeyboardInterrupt:
                print("\n(kesildi)")
            elapsed = time.time() - start
            logger.info("Streaming tamamlandı: %d token / %.2fs (%.1f tok/s)",
                        count, elapsed, count / max(elapsed, 1e-9))
        else:
            eos = backend.tokenizer.eos_id
            ids = runtime.generate_ids(
                backend, prompt_ids, args.max_tokens,
                temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                repetition_penalty=args.repetition_penalty, eos_token_id=eos,
                seed=args.seed,
            )
            elapsed = time.time() - start
            logger.info("Üretim tamamlandı: %d token / %.2fs (%.1f tok/s)",
                        len(ids), elapsed, len(ids) / max(elapsed, 1e-9))
            # Yalnızca üretilen kısmı göster (prompt zaten ekranda).
            print(backend.tokenizer.decode(ids), end="")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
