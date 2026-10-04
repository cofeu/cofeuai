#!/usr/bin/env python3
"""colab_sft_chat.ipynb dosyasini URETIR.

Neden var: notebook, python/sft.py ve python/prepare_sft_data.py dosyalarini
GitHub'dan cekiyor. Push yapilmadan bu dosyalar remote'da ya yok ya eski.
Bunun yerine iki dosyayi base64 olarak notebook'un icine gomuyoruz; boylece
Colab'da push/upload gerektiren HICBIR sey kalmiyor.

Gomulu kopyalar bu dosyadan uretildigi icin repo'daki guncel surumle
otomatik esit kalir. Dosyalardan birini degistirince:
    python3 data/sft/build_chat_notebook.py
sonra notebook'u commit'le.
"""
import base64
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "sft" / "colab_sft_chat.ipynb"

# GitHub'a bagimlilik olmasin diye gomulen dosyalar: repo yolu -> Colab yolu
EMBEDDED = [
    ("python/sft.py", "/content/cofeuai/python/sft.py"),
    ("python/prepare_sft_data.py", "/content/cofeuai/python/prepare_sft_data.py"),
]

CELLS = {
    "e01": """\
# COFEU SFT — Türkçe Sohbet + Bilgi (Colab)
# GPU: T4/A100 | eğitim ~30-45 dk, veri indirme ~5 dk
#
# Veri: AhiskaAI/sharegpt-turkish (gerçek sohbet, 74K tur)
#     + tascib/turkish-instruction (kısa soru/cevap, 324K)
# NOT: sft.py, veri hazırlığını python/prepare_sft_data.py ile yapıyor.
""",

    "e02": """\
# GPU kontrolü: CPU'da eğitim saatlerce sürer, baştan uyaralım.
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "GPU YOK"
!python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
""",

    "e03": """\
import subprocess
r = subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/cofeu/cofeuai.git", "/content/cofeuai"],
                   capture_output=True, text=True)
print(r.stderr[-300:])
if r.returncode != 0:
    raise RuntimeError("repo klonlanamadi (internet/DNS kontrolu)")
print(subprocess.run(["git", "-C", "/content/cofeuai", "log", "--oneline", "-1"],
                     capture_output=True, text=True).stdout)
""",

    "e04": """\
from google.colab import drive
drive.mount('/content/drive')
print("Drive baglandi")
""",

    "e05": """\
import os, shutil, glob, zipfile
from pathlib import Path

REPO = Path('/content/cofeuai')
(REPO / 'checkpoints').mkdir(parents=True, exist_ok=True)
(REPO / 'python').mkdir(parents=True, exist_ok=True)

# train.py modeli uc isimle kaydediyor; hepsini kabul et, tum Drive'i tara.
CANDIDATES = ["cofeu.pt", "cofeu_best.pt", "cofeu_latest.pt"]

def find(names):
    for name in names:
        for pat in (f"/content/drive/MyDrive/{name}",
                    f"/content/drive/MyDrive/cofeuai/checkpoints/{name}",
                    f"/content/drive/MyDrive/cofeuai/{name}",
                    f"/content/drive/MyDrive/checkpoints/{name}"):
            if os.path.exists(pat):
                return Path(pat)
    for name in names:
        hits = glob.glob(f"/content/drive/MyDrive/**/{name}", recursive=True)
        if hits:
            return Path(hits[0])
    return None

src = find(CANDIDATES)
print("=== MODEL ARAMASI ===")
for n in CANDIDATES:
    hits = glob.glob(f"/content/drive/MyDrive/**/{n}", recursive=True)
    print(f"  {n:18s} {'BULUNDU: ' + hits[0] if hits else 'yok'}")
if src is None:
    raise FileNotFoundError(
        "Drive'da model checkpoint'i yok. Yukleyin:\n"
        "  ~/Desktop/CofeuAI/checkpoints/cofeu.pt\n"
        "  -> /content/drive/MyDrive/cofeuai/checkpoints/cofeu.pt")
print(f"secilen: {src}\n")
shutil.copy(src, REPO / 'checkpoints' / 'cofeu.pt')
print(f"cofeu.pt ({os.path.getsize(REPO/'checkpoints'/'cofeu.pt')/1e6:.1f} MB) hazir")

vsrc = find(["vocab.json"])
if vsrc is None:
    raise FileNotFoundError("Drive'da vocab.json yok.")
shutil.copy(vsrc, REPO / 'checkpoints' / 'vocab.json')
print(f"vocab.json <- {vsrc}")
""",

    "e06": """\
# Veri hazirligi. Script repo ile birlikte gelir; internet gerekir
# (ShareGPT 31 MB + tascib 390 MB). Onceden /content/sft_raw'a indirilmis
# olurlarsa tekrar indirmez.
#
# Sohbetleri tek turlu (insan, asistan) ciftlerine duzlestirir; medyan 12
# tur oldugu icin sohbetleri oldugu gibi 512 token'a sigmaz.
import subprocess, sys

cmd = [
    sys.executable, "python/prepare_sft_data.py",
    "--out", "/content/sft_data.jsonl",
    "--cache-dir", "/content/sft_raw",
    "--max-sharegpt", "60000",
    "--max-tascib", "60000",
]
print(" ".join(cmd), "\n")
r = subprocess.run(cmd, cwd="/content/cofeuai")
if r.returncode != 0:
    raise RuntimeError(f"veri hazirligi basarisiz, exit={r.returncode}")
print("\nVERI HAZIR")
""",

    "e07": """\
# lr 3e-5, 1 epoch. Onceki denemeler 5e-5'te ezberlemeye yol acti.
# Ciktiyi Dogrudan Drive'a yaziyoruz; oturum duserse checkpoint kalir.
# sft.py log DOSYASI yazmiyor, o yuzden ciktiyi hem ekrana hem train.log'a
# yaziyoruz.
import subprocess, sys
from pathlib import Path

OUT_DIR = Path('/content/drive/MyDrive/cofeuai/sft_chat')
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = OUT_DIR / 'train.log'

cmd = [
    sys.executable, "python/sft.py",
    "--data", "/content/sft_data.jsonl",
    "--resume", "/content/cofeuai/checkpoints/cofeu.pt",
    "--out-dir", str(OUT_DIR),
    "--epochs", "1",
    "--batch-size", "16", "--lr", "3e-5", "--patience", "3",
    "--eval-interval", "200", "--save-interval", "500",
    "--log-interval", "50", "--device", "cuda",
]
print("calistiriliyor:", " ".join(cmd), "\n")
print(f"log -> {LOG}\n")

with open(LOG, "w", encoding="utf-8", buffering=1) as logf:
    logf.write(f"$ {' '.join(cmd)}\n\n")
    p = subprocess.Popen(cmd, cwd="/content/cofeuai",
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="")
        logf.write(line)
    rc = p.wait()

if rc != 0:
    raise RuntimeError(f"egitim basarisiz, exit={rc}. Log: {LOG}")
print(f"\nEGITIM TAMAM | log: {LOG}")
""",

    "e08": """\
import sys, torch
sys.path.insert(0, '/content/cofeuai/python')
from model import CofeuTransformer, ModelConfig
from tokenizer import BPETokenizer
import sft

CKPT = '/content/drive/MyDrive/cofeuai/sft_chat/cofeu_sft.pt'
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
print("iteration:", ck.get('iteration'), "| val:", ck.get('val_loss'))

cfg = ModelConfig(**ck['config'].__dict__)
m = CofeuTransformer(cfg)
m.load_state_dict(ck['model_state'])
m.eval().cuda()
tok = BPETokenizer.load('/content/cofeuai/checkpoints/vocab.json')

for q in ("Merhaba, nasılsın?",
          "Türkiye'nin başkenti neresidir?",
          "Fotosentez nasıl çalışır?",
          "Bana kısa bir motivasyon cümlesi söyle."):
    p, _ = sft.build_text(q, "", "")
    x = torch.tensor([tok.encode(p)], dtype=torch.long).cuda()
    torch.manual_seed(42)
    out = m.generate(x, 120, temperature=0.7, top_k=40, eos_token_id=tok.eos_id)
    print(f"\n>>> {q}\n{tok.decode(out[0, len(x[0]):].tolist())}")
""",

    "e09": """\
import subprocess
from pathlib import Path
OUT_DIR = Path('/content/drive/MyDrive/cofeuai/sft_chat')

print(subprocess.run(["bash", "-c", f"tail -25 {OUT_DIR}/train.log"],
                     capture_output=True, text=True).stdout)
print("\n--- dosyalar ---")
print(subprocess.run(["bash", "-c", f"ls -la {OUT_DIR}"],
                     capture_output=True, text=True).stdout)

subprocess.run(["zip", "-j", "/content/cofeu_sft_chat.zip", str(OUT_DIR / 'cofeu_sft.pt')])
print("zip: /content/cofeu_sft_chat.zip  (Colab Files sekmesinden indir)")
""",

}

# Hucre sirasi: e03 (klon) sonrasi gomulu dosyalar, cunku /content/cofeuai
# dizini ancak klon sonrasi var.
SEQUENCE = [
    CELLS["e01"],                                   # baslik / yorumlar
    CELLS["e02"],                                   # GPU kontrolu
    CELLS["e03"],                                   # repo klonu
    None,                                           # yer tutucu: sft.py gomulur
    None,                                           # yer tutucu: prepare gomulur
    CELLS["e04"],                                   # Drive mount
    CELLS["e05"],                                   # model + vocab bul
    CELLS["e06"],                                   # veri hazirligi
    CELLS["e07"],                                   # egitim (-> Drive/sft_chat)
    CELLS["e08"],                                   # uretim testi
    CELLS["e09"],                                   # log + zip
]


def embed_cell(relpath: str, dest: str) -> str:
    """Repo dosyasini notebook'a base64 olarak gomer (byte-kesin kopyalama)."""
    data = (REPO / relpath).read_bytes()
    b64 = base64.b64encode(data).decode()
    # tek satirlik string literal'lara sar; hucre boylece tek satirdir
    chunks = "\n".join(f'    "{b64[i:i+100]}"' for i in range(0, len(b64), 100))
    return (
        "import base64, pathlib\n"
        f"# GOMULU KOPYA: {relpath} -> {dest}\n"
        "# Dosya GitHub'a push edilmedi; notebook kendi kopyasini yaziyor.\n"
        f"# Ureten: data/sft/build_chat_notebook.py ({relpath} okunarak)\n"
        "_b64 = (\n" + chunks + "\n)\n"
        "_raw = base64.b64decode(_b64)\n"
        f"_p = pathlib.Path({dest!r})\n"
        "_p.parent.mkdir(parents=True, exist_ok=True)\n"
        "_p.write_bytes(_raw)\n"
        f'print("gomuldu: {relpath} -> " + str(_p) + " (" + str(len(_raw)) + " byte)")\n'
    )


def to_source(text: str) -> list:
    lines = text.rstrip("\n").split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def build():
    srcs = list(SEQUENCE)
    for slot, (rel, dest) in zip((3, 4), EMBEDDED):
        srcs[slot] = embed_cell(rel, dest)

    for i, s in enumerate(srcs):
        assert isinstance(s, str) and s.strip(), f"hucre {i} bos"

    cells = [{"cell_type": "code", "execution_count": None, "metadata": {},
              "outputs": [], "source": to_source(s)} for s in srcs]

    nb = {
        "cells": cells,
        "metadata": {
            "colab": {"name": "cofeu_sft_chat.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"yazildi: {OUT} ({OUT.stat().st_size} byte, {len(cells)} hucre)")
    for rel, _ in EMBEDDED:
        print(f"  gomulu: {rel} ({(REPO / rel).stat().st_size} byte)")


if __name__ == "__main__":
    sys.exit(build())
