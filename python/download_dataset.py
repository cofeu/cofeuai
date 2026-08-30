"""Gerçek Türkçe dataset indirir.

HuggingFace'ten Türkçe metin datasetleri indirir ve eğitim için
birleştirilmiş bir corpus dosyası oluşturur.

Kaynaklar:
- Türkçe Wikipedia (wikimedia/wikipedia, tr subset)
- Türkçe haber metinleri

Kullanım:
    python download_dataset.py --max-chars 5000000
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_PATH = DATA_DIR / "turkish_corpus.txt"

# HuggingFace datasets-server API ile Türkçe Wikipedia'dan metin çekme
# Bu API, dataset'i indirmeden satır satır erişim sağlar.
HF_API = "https://datasets-server.huggingface.co/rows"


def clean_text(text: str) -> str:
    """Metni temizler: fazla boşluk, özel karakterler vb."""
    # HTML etiketlerini kaldır
    text = re.sub(r"<[^>]+>", "", text)
    # Fazla boşlukları tek boşluğa indir
    text = re.sub(r"\s+", " ", text)
    # Baş/son boşluk
    text = text.strip()
    return text


def download_wikipedia(max_chars: int) -> str:
    """Türkçe Wikipedia'dan metin indirir (datasets-server API ile)."""
    dataset = "wikimedia/wikipedia"
    config = "20231101.tr"
    split = "train"

    texts = []
    total_chars = 0
    offset = 0
    batch_size = 100

    print(f"Wikipedia ({config}) indiriliyor...")

    while total_chars < max_chars:
        url = f"{HF_API}?dataset={dataset}&config={config}&split={split}&offset={offset}&length={batch_size}"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  Hata (offset={offset}): {e}")
            break

        rows = data.get("rows", [])
        if not rows:
            break

        for row in rows:
            text = row.get("row", {}).get("text", "")
            text = clean_text(text)
            if len(text) < 50:  # çok kısa metinleri atla
                continue
            texts.append(text)
            total_chars += len(text)
            if total_chars >= max_chars:
                break

        offset += batch_size
        if offset % 1000 == 0:
            print(f"  {total_chars:,} karakter indirildi...")

    print(f"  Toplam: {total_chars:,} karakter, {len(texts)} makale")
    return "\n\n".join(texts)


def download_news(max_chars: int) -> str:
    """Türkçe haber metinleri indirir (alternatif kaynak)."""
    # Bu fonksiyon, Wikipedia yeterli olmazsa kullanılabilir.
    # Şimdilik boş döner.
    return ""


def main():
    parser = argparse.ArgumentParser(description="Türkçe dataset indir")
    parser.add_argument("--max-chars", type=int, default=3000000,
                        help="İndirilecek maksimum karakter sayısı")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Wikipedia'dan indir
    wiki_text = download_wikipedia(args.max_chars)

    if not wiki_text:
        print("Wikipedia indirilemedi. Örnek corpus kullanılacak.")
        return

    # Mevcut örnek corpus ile birleştir
    sample_path = DATA_DIR / "sample_corpus.txt"
    sample_text = ""
    if sample_path.exists():
        sample_text = sample_path.read_text(encoding="utf-8")

    combined = wiki_text
    if sample_text:
        combined = sample_text + "\n\n" + wiki_text

    OUT_PATH.write_text(combined, encoding="utf-8")
    print(f"\nCorpus kaydedildi: {OUT_PATH}")
    print(f"Toplam karakter: {len(combined):,}")
    print(f"Yaklaşık kelime: {len(combined.split()):,}")


if __name__ == "__main__":
    main()