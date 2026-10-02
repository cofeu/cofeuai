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
import time
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


def download_wikipedia(
    max_chars: int,
    start_offset: int = 0,
    max_retries: int = 8,
    base_delay: float = 5.0,
    max_delay: float = 120.0,
) -> str:
    """Türkçe Wikipedia'dan metin indirir (datasets-server API ile).

    429 (rate limit) geçicidir: aynı offset üstel geri çekilmeyle yeniden
    denenir. `Retry-After` başlığı varsa o süre beklenir. Üst üste
    max_retries denemesi başarısız olursa o noktadaki veri kaybedilir ama
    daha önce indirilenler korunur.

    start_offset: Daha önce indirilmiş satırları atlamak için. Yarım kalan
    bir indirmeyi sürdürürken önceki offset'i verip yeni metni ayrı dosyaya
    yazmalısın (aksi halde veri kaybolur).
    """
    dataset = "wikimedia/wikipedia"
    config = "20231101.tr"
    split = "train"

    texts = []
    total_chars = 0
    offset = start_offset
    batch_size = 100
    batches_done = 0

    print(f"Wikipedia ({config}) indiriliyor..." + (f" offset={start_offset}'dan" if start_offset else ""))

    while total_chars < max_chars:
        url = f"{HF_API}?dataset={dataset}&config={config}&split={split}&offset={offset}&length={batch_size}"
        data = None
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, timeout=60)
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else min(
                        base_delay * (2 ** attempt), max_delay
                    )
                    print(
                        f"  429 alındı (offset={offset}), {wait:.0f} sn bekleniyor "
                        f"(deneme {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                wait = min(base_delay * (2 ** attempt), max_delay)
                print(
                    f"  Hata (offset={offset}): {e} — {wait:.0f} sn sonra "
                    f"tekrar denenecek ({attempt + 1}/{max_retries})"
                )
                time.sleep(wait)

        if data is None:
            print(f"  {max_retries} deneme başarısız (offset={offset}). Burada duruluyor.")
            break

        rows = data.get("rows", [])
        if not rows:
            print("  Daha fazla satır yok, indirme tamamlandı.")
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
        batches_done += 1
        if batches_done % 5 == 0:
            # Rate limit'e takılmamak için düşük trafik
            time.sleep(1.0)

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
    parser.add_argument("--out", type=str, default=None,
                        help=f"Çıktı dosyası (varsayılan: {OUT_PATH})")
    parser.add_argument("--start-offset", type=int, default=0,
                        help="Atlanacak satır sayısı (yarım indirmeyi sürdürmek için)")
    parser.add_argument("--append", action="store_true",
                        help="Mevcut çıktı dosyasının sonuna ekle (şablonsuz ham veri)")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else OUT_PATH

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Wikipedia'dan indir
    wiki_text = download_wikipedia(args.max_chars, start_offset=args.start_offset)

    if not wiki_text:
        print("Wikipedia indirilemedi. Örnek corpus kullanılacak.")
        return

    if args.append:
        # Ham devamlılık modu: sadece yeni metni ekle, sample karıştırma.
        mode = "a" if out_path.exists() else "w"
        with open(out_path, mode, encoding="utf-8") as f:
            f.write(wiki_text + "\n\n")
        print(f"\nEklendi -> {out_path}")
        print(f"Toplam karakter: {out_path.stat().st_size:,}")
        return

    # Mevcut örnek corpus ile birleştir
    sample_path = DATA_DIR / "sample_corpus.txt"
    sample_text = ""
    if sample_path.exists():
        sample_text = sample_path.read_text(encoding="utf-8")

    combined = wiki_text
    if sample_text:
        combined = sample_text + "\n\n" + wiki_text

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(combined, encoding="utf-8")
    print(f"\nCorpus kaydedildi: {out_path}")
    print(f"Toplam karakter: {len(combined):,}")
    print(f"Yaklaşık kelime: {len(combined.split()):,}")


if __name__ == "__main__":
    main()