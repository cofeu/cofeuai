import os, sys, glob, shutil, subprocess
from pathlib import Path

REPO  = Path("/content/cofeuai")
DRIVE = "/content/drive/MyDrive/cofeuai"

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

v = bul(["vocab.json"])
if v is None:
    raise FileNotFoundError("vocab.json bulunamadi")
shutil.copy(v, REPO / "checkpoints" / "vocab.json")
print("vocab <-", v)

ck = bul(["cofeu_sft.pt"])
if ck is None:
    raise FileNotFoundError("cofeu_sft.pt bulunamadi")
shutil.copy(ck, REPO / "cofeu_sft.pt")
print("ckpt  <-", ck)

# PROMPT
SYSTEM_PROMPT = "Sen yardımcı bir yapay zeka asistanısın."
def build_text(instruction, inp="", output=""):
    p = instruction.strip()
    if inp and inp.strip():
        p = f"{p}\n\n{inp.strip()}"
    part = f"### Sistem:\n{SYSTEM_PROMPT}\n\n### Kullanıcı:\n{p}\n\n### Asistan:\n"
    return part, part + output.strip()

# TEST
import torch
sys.path.insert(0, str(REPO / "python"))
from model import CofeuTransformer, ModelConfig
from tokenizer import BPETokenizer

c = torch.load(REPO / "cofeu_sft.pt", map_location="cpu", weights_only=False)
print("iteration:", c.get("iteration"), "| val:", c.get("val_loss"))
cfg = ModelConfig(**c["config"].__dict__)
m = CofeuTransformer(cfg)
m.load_state_dict(c["model_state"])
dev = "cuda" if torch.cuda.is_available() else "cpu"
m.eval().to(dev)
print("cihaz:", dev, "| param:", f"{sum(p.numel() for p in m.parameters())/1e6:.2f}M")
tok = BPETokenizer.load(str(REPO / "checkpoints" / "vocab.json"))

def uret(soru, temp):
    pr, _ = build_text(soru)
    x = torch.tensor([tok.encode(pr)], dtype=torch.long).to(dev)
    torch.manual_seed(42)
    o = m.generate(x, 150, temperature=temp, top_k=40, eos_token_id=tok.eos_id)
    y = o[0, len(x[0]):].tolist()
    return tok.decode([t for t in y if t != tok.eos_id]), len(y), tok.eos_id in y

for et, q in [("selamlama","Merhaba, nasılsın?"),
              ("bilgi","Türkiye'nin başkenti neresidir?"),
              ("uzunluk","Renkleri yalnızca üç kelimeyle yaz."),
              ("bilinmeyen","Kaptan John Smith'in doğum tarihi nedir?"),
              ("cozum","Bir elma 5 TL, portakal 7 TL. 3 elma 2 portakal kaç TL?"),
              ("uzun","Fotosentez nedir?")]:
    print("\n" + "="*72)
    print(f"[{et}] {q}")
    for tp, ad in ((0.0,"greedy"), (0.7,"ornek")):
        mtn, n, eos = uret(q, tp)
        print(f"  {ad:6s} ({n:3d} tok, EOS={'evet' if eos else 'YOK'})")
        print(f"         {mtn[:400]}")
