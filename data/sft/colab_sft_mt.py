import os, sys, glob, shutil, subprocess
from pathlib import Path

REPO  = Path("/content/cofeuai")
DRIVE = Path("/content/drive/MyDrive/cofeuai")

# GPU
if subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0:
    print("CUDA torch kuruluyor...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch",
                    "--index-url", "https://download.pytorch.org/whl/cu121"],
                   check=False)
else:
    print("GPU yok -> CPU")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy"],
               check=False)

# DRIVE
from google.colab import drive
drive.mount("/content/drive")
print("Drive baglandi")

# REPO
if not (REPO / "python" / "model.py").exists():
    r = subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/cofeu/cofeuai.git", str(REPO)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("klon basarisiz:", r.stderr[:200])
    else:
        print("klonlandi")
else:
    print("repo zaten var")
(REPO / "checkpoints").mkdir(parents=True, exist_ok=True)

# BUL
def bul(adlar):
    for ad in adlar:
        for y in (f"{DRIVE}/checkpoints/{ad}", f"{DRIVE}/{ad}",
                  f"/content/drive/MyDrive/checkpoints/{ad}",
                  f"/content/drive/MyDrive/{ad}"):
            if os.path.exists(y):
                return Path(y)
    for ad in adlar:
        h = glob.glob(f"/content/drive/MyDrive/**/{ad}", recursive=True)
        if h:
            return Path(h[0])
    return None

# vocab + base model
v = bul(["vocab.json"])
if v is None:
    raise FileNotFoundError("vocab.json yok")
shutil.copy(v, REPO / "checkpoints" / "vocab.json")
print("vocab <-", v)

base = bul(["cofeu.pt", "cofeu_best.pt", "cofeu_latest.pt"])
if base is None:
    raise FileNotFoundError("taban checkpoint (cofeu.pt vb) yok")
shutil.copy(base, REPO / "checkpoints" / "cofeu.pt")
print("base  <-", base)

# veri: cofedata.jsonl
data = Path("/content/cofedata.jsonl")
if not data.exists():
    # drive'da ara
    h = glob.glob("/content/drive/MyDrive/**/cofedata.jsonl", recursive=True)
    if h:
        data = Path(h[0])
        shutil.copy(data, "/content/cofedata.jsonl")
    else:
        raise FileNotFoundError("cofedata.jsonl bulunamadi (lokalde data/sft/cofedata.jsonl -> Drive'a kopyala)")
else:
    pass

if not Path("/content/cofedata.jsonl").exists() and data.exists():
    shutil.copy(data, "/content/cofedata.jsonl")
print("data  <-", "/content/cofedata.jsonl")

# eğitim
out = DRIVE / "sft_mt"
out.mkdir(parents=True, exist_ok=True)
cmd = [
    sys.executable, "python/sft_mt.py",
    "--data", "/content/cofedata.jsonl",
    "--resume", str(REPO / "checkpoints" / "cofeu.pt"),
    "--out-dir", str(out),
    "--epochs", "2",
    "--batch-size", "16",
    "--lr", "3e-5",
    "--block-size", "512",
    "--eval-interval", "25",
    "--save-interval", "50",
    "--log-interval", "10",
    "--device", "cuda",
]
print("CALISTIRILIYOR:", " ".join(cmd))
subprocess.run(cmd, cwd=str(REPO))
print("BITTI. Log:", out / "train.log")
