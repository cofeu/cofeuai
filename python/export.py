"""CofeuAI Model Dışa Aktarma.

Eğitilmiş PyTorch modelini, C++ inference motorunun okuyabileceği
düz bir binary formata dönüştürür.

Format (cpp/include/cofeu/model.hpp ile birebir aynı olmalıdır):

    [char[4] "COFE"]          imza
    [int32]   version         = 2
    [int32]   vocab_size
    [int32]   n_embd
    [int32]   n_head
    [int32]   n_layer
    [int32]   block_size
    [int32]   rope_base
    [float32 x N]             ağırlıklar (aşağıdaki sırada)

Ağırlık sırası:
    token_emb.weight                      (V x C)
    her blok i için:
        ln1.weight, ln1.bias              (C)
        attn.query.weight                 (C x C)
        attn.key.weight                   (C x C)
        attn.value.weight                 (C x C)
        attn.proj.weight                  (C x C)
        ln2.weight, ln2.bias              (C)
        mlp.fc1.weight                    (4C x C)
        mlp.fc1.bias                      (4C)
        mlp.fc2.weight                    (C x 4C)
        mlp.fc2.bias                      (C)
    ln_f.weight, ln_f.bias                (C)
    head.weight                           (V x C)

Not: RoPE'ye geçişle birlikte mutlak konum gömmesi (pos_emb) KALDIRILDI;
onun yerine header'da rope_base taşınır.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = ROOT / "checkpoints" / "cofeu.pt"
DEFAULT_OUT = ROOT / "checkpoints" / "cofeu.bin"

# cpp/include/cofeu/model.hpp
MODEL_MAGIC = b"COFE"
MODEL_VERSION = 2
HEADER_BYTES = 4 + 4 + 6 * 4


def expected_float_count(config: Any) -> int:
    """C++ motorunun beklediği toplam float sayısı."""
    V, C, L = config.vocab_size, config.n_embd, config.n_layer
    per_layer = 12 * C * C + 9 * C
    return 2 * V * C + L * per_layer + 2 * C


def write_tensor(f, name: str, state: dict, rows: int, cols: int) -> None:
    """Bir state_dict girdisini doğrulayıp float32 olarak yazar."""
    if name not in state:
        raise KeyError(f"state_dict içinde '{name}' yok (model mimarisi değişmiş olabilir)")
    t = state[name]
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"'{name}' tensor değil: {type(t).__name__}")
    if tuple(t.shape) != (rows, cols):
        raise ValueError(f"'{name}' şekli {tuple(t.shape)} beklenen {(rows, cols)}")
    if not torch.isfinite(t).all():
        bad = int((~torch.isfinite(t)).sum())
        raise ValueError(f"'{name}' içinde {bad} NaN/Inf var; dışa aktarma iptal")
    f.write(t.detach().to(torch.float32).cpu().numpy().tobytes())


def write_vector(f, name: str, state: dict, n: int) -> None:
    """1-D bias/gain vektörünü yazar."""
    if name not in state:
        raise KeyError(f"state_dict içinde '{name}' yok (model mimarisi değişmiş olabilir)")
    t = state[name]
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"'{name}' tensor değil: {type(t).__name__}")
    if tuple(t.shape) != (n,):
        raise ValueError(f"'{name}' şekli {tuple(t.shape)} beklenen ({n},)")
    if not torch.isfinite(t).all():
        bad = int((~torch.isfinite(t)).sum())
        raise ValueError(f"'{name}' içinde {bad} NaN/Inf var; dışa aktarma iptal")
    f.write(t.detach().to(torch.float32).cpu().numpy().tobytes())


def export(ckpt_path: Path, out_path: Path) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    state = ckpt["model_state"]

    V, C, L = config.vocab_size, config.n_embd, config.n_layer
    rope_base_raw = getattr(config, "rope_base", 10000)

    # Header'da rope_base int32 olarak saklanır (C++ tarafı `read_int` ile
    # okuyor). Config float tuttuğu için burada tam sayıya çeviriyoruz; kesirli
    # bir değeri sessizce yuvarlamak yerine reddediyoruz.
    if abs(rope_base_raw - round(rope_base_raw)) > 1e-9:
        raise ValueError(
            f"rope_base tam sayı olmalı (header int32): {rope_base_raw!r}. "
            "Örn. --rope-base 10000"
        )
    rope_base = int(round(rope_base_raw))

    if V <= 0 or C <= 0 or L <= 0 or config.block_size <= 0:
        raise ValueError("geçersiz model konfigürasyonu")
    if C % config.n_head != 0:
        raise ValueError(f"n_embd ({C}) n_head ({config.n_head}) ile bölünmüyor")
    if (C // config.n_head) % 2 != 0:
        raise ValueError(f"head_dim ({C // config.n_head}) çift olmalı (RoPE)")
    if rope_base <= 1:
        raise ValueError("rope_base 1'den büyük olmalı")

    # Eski (pos_emb'li) mimari yanlışlıkla buraya gelirse sessizce bozuk
    # dosya yazmak yerine net hata ver.
    stale = [k for k in state if "pos_emb" in k]
    if stale:
        raise ValueError(
            "Bu checkpoint mutlak konum gömmesi (pos_emb) içeriyor; C++ motoru "
            "RoPE kullanıyor. RoPE'ye geçişten sonra üretilmiş yeni bir "
            "checkpoint ile (python train.py) tekrar deneyin."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    written = 0
    with open(tmp_path, "wb") as f:
        f.write(MODEL_MAGIC)
        f.write(struct.pack("<i", MODEL_VERSION))
        f.write(struct.pack("<6i", V, C, config.n_head, L, config.block_size, rope_base))

        write_tensor(f, "token_emb.weight", state, V, C)
        written += V * C

        for i in range(L):
            p = f"blocks.{i}."
            write_vector(f, p + "ln1.weight", state, C)
            write_vector(f, p + "ln1.bias", state, C)
            written += 2 * C
            for nm in ("query", "key", "value", "proj"):
                write_tensor(f, f"{p}attn.{nm}.weight", state, C, C)
                written += C * C
            write_vector(f, p + "ln2.weight", state, C)
            write_vector(f, p + "ln2.bias", state, C)
            written += 2 * C
            write_tensor(f, p + "mlp.fc1.weight", state, 4 * C, C)
            write_vector(f, p + "mlp.fc1.bias", state, 4 * C)
            write_tensor(f, p + "mlp.fc2.weight", state, C, 4 * C)
            write_vector(f, p + "mlp.fc2.bias", state, C)
            written += 8 * C * C + 5 * C

        write_vector(f, "ln_f.weight", state, C)
        write_vector(f, "ln_f.bias", state, C)
        written += 2 * C
        write_tensor(f, "head.weight", state, V, C)
        written += V * C

    want = expected_float_count(config)
    if written != want:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"iç sayım tutarsız: yazılan {written}, beklenen {want}")

    got = tmp_path.stat().st_size
    if got != HEADER_BYTES + want * 4:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"dosya boyutu tutarsız: {got}, beklenen {HEADER_BYTES + want * 4}"
        )

    tmp_path.replace(out_path)
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description="PyTorch modelini C++ binary formatına aktar")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT,
                    help=f"girdi checkpoint (varsayılan: {DEFAULT_CKPT})")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"çıktı dosyası (varsayılan: {DEFAULT_OUT})")
    args = ap.parse_args()

    if not args.checkpoint.exists():
        print(f"HATA: checkpoint bulunamadı: {args.checkpoint}", file=sys.stderr)
        print("Önce eğitin: python train.py", file=sys.stderr)
        return 1

    try:
        size = export(args.checkpoint, args.out)
    except (KeyError, ValueError, TypeError, RuntimeError) as e:
        print(f"HATA: {e}", file=sys.stderr)
        return 1

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    c = ckpt["config"]
    print(f"Model dışa aktarıldı: {args.out} ({size / (1024 * 1024):.2f} MB)")
    print(f"  V={c.vocab_size} C={c.n_embd} H={c.n_head} L={c.n_layer} "
          f"block={c.block_size} rope_base={getattr(c, 'rope_base', 10000)}")
    print(f"  {c.n_params() if hasattr(c, 'n_params') else sum(v.numel() for v in ckpt['model_state'].values()):,} parametre")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
