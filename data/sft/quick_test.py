#!/usr/bin/env python3
# Colab'da TEK HUCRE olarak calistir: kurulum + uretim testi.
#
# Neden ayri dosya: sohbet (push basarisiz) GitHub'daki surum
# guncel degil. Bu script yalnizca model.py + tokenizer.py icin clone'a
# guvenir; prompt sablonu egitimde kullanilanla BIREBIR ayni olacak
# sekilde icine yazilmistir (asagida SYSTEM_PROMPT / build_text).
#
# Kullanim: Colab'da yeni hucre ac, bu dosyanin tamamini yapistir, calistir.
# Sadece model.py + tokenizer.py + vocab.json gerekiyor; sft.py gerekmiyor
# (prompt sablonu icine yazildi).
import os
import sys
import glob
import shutil
import subprocess
from pathlib import Path

REPO = Path("/content/cofeuai")
DRIVE = "/content/drive/MyDrive/cofeuai"


# ---------------------------------------------------------------- 1. bagimlilik
def pip_kur():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy"],
                   check=False)


# ---------------------------------------------------------------- 2. GPU (istege bagli)
def gpu_kur():
    """T4 varsa CUDA torch kur. Yoksa CPU'da calisir (29M model icin yeter).

    BURADA import torch YAPILMAZ. Yapilirsa CPU surumu sys.modules'a
    cache'lenir ve asagidaki `import torch` yine onu getirir; kurulan
    CUDA surumu kullanilmaz. Sadece nvidia-smiye bakip pip calistiririz.
    """
    if subprocess.run(["nvidia-smi"], capture_output=True).returncode != 0:
        print("GPU yok -> CPU. Uretim testi birkac dakika surer.")
        return
    print("CUDA torch kuruluyor (bir dakika surebilir)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch", "--index-url",
                    "https://download.pytorch.org/whl/cu121"], check=False)
    print("Kuruldu.")


# ---------------------------------------------------------------- 3. repo
def repo_kur():
    if (REPO / "python" / "model.py").exists():
        print("repo zaten var")
        return
    r = subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/cofeu/cofeuai.git", str(REPO)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            "git clone basarisiz. Repo zaten /content/cofeuai altindaysa "
            "bu adimi atlayip calistir.\n" + (r.stderr or "")[:400])
    print("repo klonlandi")


# ---------------------------------------------------------------- 4. dosya bul
def bul(adlar):
    """Drive'da bir dosyayi ara. Oncelik: bilinen yollar, sonra genel tarama."""
    for ad in adlar:
        for yol in (f"{DRIVE}/checkpoints/{ad}",
                    f"{DRIVE}/{ad}",
                    f"/content/drive/MyDrive/checkpoints/{ad}",
                    f"/content/drive/MyDrive/{ad}"):
            if os.path.exists(yol):
                return Path(yol)
    for ad in adlar:
        hit = glob.glob(f"/content/drive/MyDrive/**/{ad}", recursive=True)
        if hit:
            return Path(hit[0])
    return None


def hazirla():
    (REPO / "checkpoints").mkdir(parents=True, exist_ok=True)

    vsrc = bul(["vocab.json"])
    if vsrc is None:
        raise FileNotFoundError(
            "Drive'da vocab.json bulunamadi. Lokalde:\n"
            "  ~/Desktop/CofeuAI/checkpoints/vocab.json\n"
            "  -> MyDrive/cofeuai/checkpoints/vocab.json")
    shutil.copy(vsrc, REPO / "checkpoints" / "vocab.json")
    print(f"vocab.json   <- {vsrc}")

    ck = bul(["cofeu_sft.pt"])
    if ck is None:
        print("\nSFT checkpoint BULUNAMADI (cofeu_sft.pt).")
        print("Drive'da ne var:")
        for p in glob.glob(f"{DRIVE}/**/*", recursive=True):
            if os.path.isfile(p):
                print("  ", p)
        raise FileNotFoundError(
            "Egitim bitmemis ya da runtime kapandi. Egitim bitmeden "
            "checkpoint yazilmaz (ilk kayit iteration 50'de olur).")
    shutil.copy(ck, REPO / "cofeu_sft.pt")
    print(f"cofeu_sft.pt <- {ck}")


# ------------------------------------------- 5. prompt sablonu (EGITIMLE AYNI)
# Bu iki deger python/sft.py icindekilerle birebir ayni OLMALI. Farkli
# olursa model yanlis formatta cevap verir ve test anlamsizlasir.
SYSTEM_PROMPT = "Sen yardımcı bir yapay zeka asistanısın."


def build_text(instruction: str, inp: str = "", output: str = ""):
    p = instruction.strip()
    if inp and inp.strip():
        p = f"{p}\n\n{inp.strip()}"
    prompt_part = (
        f"### Sistem:\n{SYSTEM_PROMPT}\n\n"
        f"### Kullanıcı:\n{p}\n\n"
        f"### Asistan:\n"
    )
    return prompt_part, prompt_part + f"{output.strip()}"


# ---------------------------------------------------------------- 6. test
SORULAR = [
    ("selamlama",  "Merhaba, nasılsın?"),
    ("bilgi",      "Türkiye'nin başkenti neresidir?"),
    ("uzunluk",    "Renkleri yalnızca üç kelimeyle yaz."),
    ("bilinmeyen", "Kaptan John Smith'in doğum tarihi nedir?"),
    ("cozum",      "Bir elma 5 TL, portakal 7 TL. 3 elma 2 portakal kaç TL?"),
    ("uzun",       "Fotosentez nedir?"),
]


def test(ckpt_yolu):
    import torch
    sys.path.insert(0, str(REPO / "python"))
    from model import CofeuTransformer, ModelConfig
    from tokenizer import BPETokenizer

    ck = torch.load(ckpt_yolu, map_location="cpu", weights_only=False)
    print("iteration:", ck.get("iteration"), "| val:", ck.get("val_loss"))

    cfg = ModelConfig(**ck["config"].__dict__)
    m = CofeuTransformer(cfg)
    m.load_state_dict(ck["model_state"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m.eval().to(dev)
    print("cihaz:", dev, "| parametre:",
          f"{sum(p.numel() for p in m.parameters())/1e6:.2f}M")

    tok = BPETokenizer.load(str(REPO / "checkpoints" / "vocab.json"))

    def uret(soru, temp):
        p, _ = build_text(soru)
        x = torch.tensor([tok.encode(p)], dtype=torch.long).to(dev)
        torch.manual_seed(42)
        out = m.generate(x, 150, temperature=temp, top_k=40,
                         eos_token_id=tok.eos_id)
        yeni = out[0, len(x[0]):].tolist()
        return (tok.decode([t for t in yeni if t != tok.eos_id]),
                len(yeni), tok.eos_id in yeni)

    for etiket, q in SORULAR:
        print(f"\n{'='*72}\n[{etiket}] {q}")
        for temp, ad in ((0.0, "greedy"), (0.7, "ornek")):
            metin, n, eos = uret(q, temp)
            print(f"  {ad:6s} ({n:3d} tok, EOS={'evet' if eos else 'YOK'})")
            print(f"         {metin[:400]}")


# ---------------------------------------------------------------- main
if __name__ == "__main__" or True:
    from google.colab import drive
    drive.mount("/content/drive")
    print("Drive baglandi\n")

    pip_kur()
    gpu_kur()
    repo_kur()
    hazirla()
    print()
    test(REPO / "cofeu_sft.pt")