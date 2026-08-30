"""CofeuAI Model Dışa Aktarma.

Eğitilmiş PyTorch modelini, C++ inference motorunun okuyabileceği
düz bir binary formata dönüştürür.

Format:
  [int32] vocab_size
  [int32] n_embd
  [int32] n_head
  [int32] n_layer
  [int32] block_size
  [float32 x N] ağırlıklar (sıralı)
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = ROOT / "checkpoints" / "cofeu.pt"
OUT_PATH = ROOT / "checkpoints" / "cofeu.bin"


def write_tensor(f, t: torch.Tensor):
    """Bir tensor'u float32 olarak yazar."""
    arr = t.detach().cpu().numpy().astype(np.float32)
    f.write(arr.tobytes())


def main():
    if not CKPT_PATH.exists():
        print(f"Model bulunamadı: {CKPT_PATH}")
        print("Önce eğitim yapın: python train.py")
        return

    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    state = ckpt["model_state"]

    with open(OUT_PATH, "wb") as f:
        # Header
        f.write(struct.pack("iiiii", config.vocab_size, config.n_embd,
                            config.n_head, config.n_layer, config.block_size))

        # Token embedding
        write_tensor(f, state["token_emb.weight"])
        # Positional embedding
        write_tensor(f, state["pos_emb.weight"])

        # Her blok için
        for i in range(config.n_layer):
            prefix = f"blocks.{i}."
            # LayerNorm 1 (weight, bias)
            write_tensor(f, state[prefix + "ln1.weight"])
            write_tensor(f, state[prefix + "ln1.bias"])
            # Attention
            write_tensor(f, state[prefix + "attn.query.weight"])
            write_tensor(f, state[prefix + "attn.key.weight"])
            write_tensor(f, state[prefix + "attn.value.weight"])
            write_tensor(f, state[prefix + "attn.proj.weight"])
            # LayerNorm 2
            write_tensor(f, state[prefix + "ln2.weight"])
            write_tensor(f, state[prefix + "ln2.bias"])
            # MLP
            write_tensor(f, state[prefix + "mlp.fc1.weight"])
            write_tensor(f, state[prefix + "mlp.fc1.bias"])
            write_tensor(f, state[prefix + "mlp.fc2.weight"])
            write_tensor(f, state[prefix + "mlp.fc2.bias"])

        # Final LayerNorm
        write_tensor(f, state["ln_f.weight"])
        write_tensor(f, state["ln_f.bias"])
        # Head
        write_tensor(f, state["head.weight"])

    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Model dışa aktarıldı: {OUT_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()