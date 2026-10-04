"""Cofeu SFT veri hazirligi: ShareGPT Turkce + tascib/turkish-instruction.

Iki kaynagi birlestirir:
  1. AhiskaAI/sharegpt-turkish  -> gercek Turkce sohbet (coklu tur)
  2. tascib/turkish-instruction -> kisa soru/cevap (instruction/input/output)

Sohbet verisi cogu zaman >512 token, o yuzden her konusma tek turlu
(instruction, output) ciftlerine duzlestirilir. Boylece egitim sirasinda
sartname tam olarak ne olacak? Cunku inference'da da tek tur soruyoruz.

Cikti semasi sft.py ile ayni: instruction / input / output.

Kullanim:
  python python/prepare_sft_data.py --out /content/sft_data.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

SOURCES = {
    "sharegpt": "https://huggingface.co/datasets/AhiskaAI/sharegpt-turkish/resolve/main/dataset.json",
    "tascib": "https://huggingface.co/datasets/tascib/turkish-instruction/resolve/main/data/turkish_instruction_dataset.jsonl",
}

MIN_CHARS = 2
MAX_OUT_CHARS = 1200      # 512 token'a sigmasina yardimci olur
MAX_INSTR_CHARS = 800


def log(msg: str) -> None:
    print(msg, flush=True)


def download(url: str, dest: Path, timeout: int = 1800) -> Path:
    """Dosyayi indirir; varsa tekrar indirmez."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        log(f"  zaten var: {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    log(f"  indiriliyor: {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    # curl ile: HF yonlendirmelerini iyi takip ediyor ve ilerleme gosteriyor
    rc = subprocess.run(
        ["curl", "-sL", "--fail", "--max-time", str(timeout), "-o", str(tmp), url]
    ).returncode
    if rc != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        raise RuntimeError(f"indirme basarisiz ({url}), curl exit={rc}")
    tmp.rename(dest)
    log(f"  indirildi: {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
    return dest


# ---------------------------------------------------------------- temizleme

_WS = re.compile(r"\s+")


def has_degenerate_repeat(text: str, max_period: int = 8, times: int = 3) -> bool:
    """Kisa bir kelime blogunun arka arkaya tekrarlanip tekrarlamadigi.

    Uretici veride "tıkpı bir tıkpı bir tıkpı" gibi bozukluklar var; modelin
    de urettigi tip bir tekrar dongusunu beslememek icin eliyoruz.

    Tek bir sabit blog boyutu (orn. 3) yetmez: bozukluklar 2 kelimelik
    ("bir tıkpı bir tıkpı bir tıkpı") ya da 5 kelimelik periyotlarda da
    goruluyor. Bu yuzden 2..max_period arasi tum periyotlari tarariz.
    """
    words = text.split()
    n = len(words)
    for period in range(2, max_period + 1):
        need = period * times
        if n < need:
            break
        for i in range(n - need + 1):
            block = words[i:i + period]
            if all(words[i + j * period:i + (j + 1) * period] == block
                   for j in range(1, times)):
                return True
    return False


def clean(text: Optional[str]) -> str:
    if not text:
        return ""
    # U+200D zero-width joiner uretici metinlerde artik kaliyor
    text = text.replace("\u200d", "").strip()
    return _WS.sub(" ", text)


def usable(instr: str, out: str) -> bool:
    if len(instr) < MIN_CHARS or len(out) < MIN_CHARS:
        return False
    if len(instr) > MAX_INSTR_CHARS or len(out) > MAX_OUT_CHARS:
        return False
    if has_degenerate_repeat(out):
        return False
    return True


# ---------------------------------------------------------------- kaynaklar

def iter_sharegpt(path: Path) -> Iterator[Tuple[str, str]]:
    """Her konusmadan (human turu, gpt cevabi) ciftleri uretir.

    Konusma sirasiyla sistem + human/gpt ... seklinde. Bir tur cifti, bir
    onceki human cevabini takip eden ilk gpt mesaji olur.
    """
    with open(path, encoding="utf-8") as f:
        convs = json.load(f)
    for conv in convs:
        if not isinstance(conv, list):
            continue
        pending: Optional[str] = None
        for msg in conv:
            if not isinstance(msg, dict):
                continue
            role, val = msg.get("from"), clean(msg.get("value"))
            if role == "human" and val:
                pending = val                      # soruyu biraktir, cevabi bekle
            elif role == "gpt" and val and pending:
                yield pending, val
                pending = None


def iter_tascib(path: Path) -> Iterator[Tuple[str, str, str]]:
    """(instruction, input, output) uclusu uretir."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            instr = clean(obj.get("instruction"))
            extra = clean(obj.get("input"))
            out = clean(obj.get("output"))
            if instr and out:
                yield instr, extra, out


# ------------------------------------------------------------------ toplama

def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="cikti jsonl")
    ap.add_argument("--cache-dir", default="/content/sft_raw")
    ap.add_argument("--max-sharegpt", type=int, default=60000)
    ap.add_argument("--max-tascib", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--stats-only", action="store_true",
                    help="sadece istatistik yaz, dosya uretme")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cache = Path(args.cache_dir)

    log("=" * 60)
    log("1) ShareGPT Turkce (gercek sohbet)")
    sg_path = download(SOURCES["sharegpt"], cache / "sharegpt-turkish.json")
    log("2) tascib/turkish-instruction (kisa soru/cevap)")
    ts_path = download(SOURCES["tascib"], cache / "turkish_instruction_dataset.jsonl")

    seen = set()
    sharegpt: List[dict] = []
    total = dropped = 0
    for instr, out in iter_sharegpt(sg_path):
        total += 1
        if not usable(instr, out):
            dropped += 1
            continue
        key = (instr[:120], out[:120])
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        sharegpt.append({"instruction": instr, "input": "", "output": out,
                         "source": "sharegpt-tr"})
    log(f"  ham cift {total} | elenen {dropped} | kalan {len(sharegpt)}")

    tascib: List[dict] = []
    total = dropped = 0
    for instr, extra, out in iter_tascib(ts_path):
        total += 1
        if not usable(instr, out):
            dropped += 1
            continue
        key = (instr[:120], out[:120])
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        tascib.append({"instruction": instr, "input": extra, "output": out,
                       "source": "tascib"})
    log(f"  ham satir {total} | elenen {dropped} | kalan {len(tascib)}")

    rng.shuffle(sharegpt)
    rng.shuffle(tascib)
    if args.max_sharegpt:
        sharegpt = sharegpt[:args.max_sharegpt]
    if args.max_tascib:
        tascib = tascib[:args.max_tascib]

    rows = sharegpt + tascib
    rng.shuffle(rows)

    log("")
    log(f"toplam: {len(rows)} (sharegpt {len(sharegpt)} + tascib {len(tascib)})")
    log("=" * 60)
    if args.stats_only:
        for r in rows[:5]:
            log(f"  [{r['source']}] {r['instruction'][:60]} -> {r['output'][:60]}")
        return 0

    n = write_jsonl(Path(args.out), rows)
    log(f"yazildi: {args.out} ({n} satir)")
    return 0


if __name__ == "__main__":
    sys.exit(main())