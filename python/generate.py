"""CofeuAI Metin Üretimi.

Eğitilmiş modeli yükleyip metin üretir. C++ inference motoru varsa onu
kullanır (çok daha hızlı), yoksa Python'a düşer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import CofeuTransformer
from tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = ROOT / "checkpoints" / "cofeu.pt"
VOCAB_PATH = ROOT / "checkpoints" / "vocab.json"
BIN_PATH = ROOT / "checkpoints" / "cofeu.bin"


def generate_cpp(prompt: str, max_tokens: int, temperature: float) -> str | None:
    """C++ inference motoru ile üretim (hızlı)."""
    try:
        from cpp_bridge import CppModel, CppTokenizer

        tok = CppTokenizer(VOCAB_PATH)
        model = CppModel(BIN_PATH)
        prompt_ids = tok.encode(prompt)
        out_ids = model.generate(prompt_ids, max_tokens, temperature)
        return tok.decode(out_ids)
    except Exception as e:
        print(f"(C++ kullanılamadı: {e})")
        return None


def generate_python(prompt: str, max_tokens: int, temperature: float) -> str:
    """Python (PyTorch) ile üretim."""
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    tokenizer = BPETokenizer(merges=ckpt.get("merges", []), vocab=ckpt["vocab"])

    model = CofeuTransformer(config)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    prompt_ids = tokenizer.encode(prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long)

    out = model.generate(idx, max_tokens, temperature=temperature)
    return tokenizer.decode(out[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="CofeuAI ile metin üret")
    parser.add_argument("prompt", type=str, help="Başlangıç metni")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--python", action="store_true", help="Python ile üret (C++ yerine)")
    args = parser.parse_args()

    if not CKPT_PATH.exists():
        print(f"Model bulunamadı: {CKPT_PATH}")
        print("Önce eğitim yapın: python train.py")
        return

    if not args.python and BIN_PATH.exists():
        text = generate_cpp(args.prompt, args.max_tokens, args.temperature)
        if text is not None:
            print(text)
            return

    text = generate_python(args.prompt, args.max_tokens, args.temperature)
    print(text)


if __name__ == "__main__":
    main()