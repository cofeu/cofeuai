#!/usr/bin/env python3
"""Kendi urettigimiz SFT verisini egitimden ONCE dogrular.

Neden: 40 dakikalik Colab kosusunu veri hatasiyla harcamak yerine 3
saniyede yakalayalim. Gecerli veri olmayan bilesenler:
  - sema (turns / conversations)
  - tur sayisi ve bos alanlar
  - cevap uzunlugu dagilimi (modelin kisa mi uzun mu konusacagini
    BU belirler; konusma bicimi buradan gelir)
  - tekillestirme
  - encoding + 512 token blok siniri

Kullanim:
    python validate_sft_data.py veri.jsonl
    python validate_sft_data.py veri.jsonl --min-turns 2 --report
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

IGNORE = -100
SYSTEM_PROMPT = "Sen yardımcı bir yapay zeka asistanısın."

HUMAN = {"human", "user", "kullanıcı", "kullanici"}
GPT = {"gpt", "assistant", "asistan", "bot"}

LEN_BANDS = ((0, 80, "kisa"), (80, 300, "orta"),
             (300, 800, "uzun"), (800, 10**9, "cok uzun"))

# Veri icermesi gereken konu tipleri. Elle uretilen veride en sik
# eksik olan sey budur: insanlar 3000 selamlama yazip gerisini unutur.
#
# SIRASI ONEMLI: ilk eslesen kategori kazanir. Ozel kaliplar once
# gelir, kisa_qa en sonda artik kalip olarak durur. Aksi halde
# kisa_qa (r"\?$") butun sorulara eslesip gercek konu tiplerini
# gizler ve kapsama raporu anlamsiz hale gelir.
COVERAGE = {
    "selamlama":  r"^(merhaba|selam|selamlar|günaydın|gunaydin|iyi günler|iyi akşamlar|"
                  r"nasılsın|nasılsiniz|ne haber|iyi haftalar)",
    "reddetme":   r"(bilmiyorum|bilemiyorum|emin değilim|emin degilim|veremem|"
                  r"değerlendiremem|hesaplayamam|öneremem|öneremez|tavsiye veremem|"
                  r"güncel veri yok|emin olamadım)",
    "adim_adim":  r"(nasıl (yapılır|yaparım|yapabilirim|olur|hazırlanır|pişirilir|kurulur|"
                  r"kullanılır|alınır|ayarlanır|çıkarılır|kaçırılır)|"
                  r"nasil (yapilir|yaparim|yapabilirim|olur|hazirlanir|pisirilir|kurulur|"
                  r"kullanilir|alinir|ayarlanir|cikarilir|kacirilir)|adım adım|adim adim|"
                  r"reçete|tarif|ipuçları|kurulum|montaj|ne yapmalıyım|ne yapmam lazım)",
    "sohbet":     r"(hissediyorum|keyfim|üzgün|mutlu|yorgun|bezgin|sıkıldım|moralim|stres|"
                  r"stresli|yapamıyorum|başaramıyorum|özgüven|yalnız|yalnızım|kaygı|"
                  r"moralim bozuk|içimden|canım)",
    "aciklama":   r"(açıkla|açikla|anlat|neden|nasıl çalışır|nasil calisir|fark|"
                  r"nedir|ne demek|ne işe yarar|niye)",
    "bilgi":      r"(tarih|tarihi|kim(ler)?|kaç|nerede|hangi|ne kadar|ne zaman|kaçta)",
}

# Artik kalip: hicbiri eslesmediyse ve ilk kullanici turu soru mi?
KISA_QA = r"\?$"


def parse_record(rec: dict):
    """Kaydi (turler, kaynak_adi) olarak dondurur; degilse None.

    Iki sema kabul edilir:
      {"turns": [[user, assistant], ...]}            -> tercih edilen
      {"conversations": [{"from": "human", ...}]}    -> ShareGPT yerel
    """
    if isinstance(rec.get("turns"), list):
        out = []
        for t in rec["turns"]:
            if not (isinstance(t, (list, tuple)) and len(t) == 2):
                return None
            u, a = t
            if not (isinstance(u, str) and isinstance(a, str)):
                return None
            out.append((u, a))
        return out or None

    convs = rec.get("conversations")
    if isinstance(convs, list) and convs:
        out, pending = [], None
        for m in convs:
            if not isinstance(m, dict):
                return None
            frm = str(m.get("from", "")).strip().lower()
            val = m.get("value")
            if not isinstance(val, str):
                return None
            if frm in HUMAN:
                pending = val
            elif frm in GPT and pending is not None:
                out.append((pending, val))
                pending = None
        return out or None

    return None


def degenerate(text: str, max_period: int = 8, times: int = 3) -> bool:
    w = text.split()
    n = len(w)
    for p in range(2, max_period + 1):
        need = p * times
        if n < need:
            break
        for i in range(n - need + 1):
            blk = w[i:i + p]
            if all(w[i + j * p:i + (j + 1) * p] == blk for j in range(1, times)):
                return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--min-turns", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=0, help="0 = sinir yok")
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        print(f"HATA: dosya yok -> {p}")
        return 2

    # tokenizer opsiyonel
    tok = None
    vocab = ROOT / "checkpoints" / "vocab.json"
    if vocab.exists():
        try:
            from tokenizer import BPETokenizer
            tok = BPETokenizer.load(str(vocab))
        except Exception as e:                        # noqa: BLE001
            print(f"uyari: tokenizer yuklenemedi ({e}); token sayimi atlandi")

    hata, uyari = Counter(), Counter()
    seen_keys, lens, n_tur, n_rec, coverage = set(), [], [], 0, Counter()
    bos, kirpilan_serit, too_long = 0, 0, 0

    with p.open(encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                hata["gecersiz JSON"] += 1
                continue
            if not isinstance(rec, dict):
                hata["JSON nesne degil"] += 1
                continue

            turns = parse_record(rec)
            if not turns:
                hata["sema hatasi (turns/conversations yok)"] += 1
                continue

            n_rec += 1
            n_tur.append(len(turns))

            if len(turns) < args.min_turns:
                hata[f"{args.min_turns}tordan az tur"] += 1
                continue
            if args.max_turns and len(turns) > args.max_turns:
                uyari[f"{args.max_turns} turdan fazla (kesilecek)"] += 1

            bad = False
            for u, a in turns:
                if not u.strip() or not a.strip():
                    bos += 1
                    bad = True
                    break
                if len(a) > 4000:
                    too_long += 1
                    uyari["4000+ kr cevap"] += 1
            if bad:
                continue

            # konu tipi kapsami. Yalnizca ilk KULLANICI turu yeterli
            # degildir: "bilmiyorum" gibi reddetme isaretleri ASISTAN
            # cevabinda yasar, o yuzden ilk cevap da taranir.
            ilk_u = turns[0][0]
            ilk_a = turns[0][1]
            konu = f"{ilk_u} {ilk_a}"
            for ad, pat in COVERAGE.items():
                if re.search(pat, konu, re.I):
                    coverage[ad] += 1
                    break
            else:
                if re.search(KISA_QA, ilk_u):
                    coverage["kisa_qa"] += 1

            for _, a in turns:
                key = re.sub(r"\s+", " ", a.strip().lower())[:120]
                if key in seen_keys:
                    hata["tekrar eden cevap"] += 1
                    bad = True
                    break
                seen_keys.add(key)
                lens.append(len(a.strip()))
                # Dejenere tekrar BIR HATADIR, uyari degil. Ilk SFT
                # kosusunda model tam da bu yuzden (tekrar dongusu)
                # bozuk metin uretiyordu; boyle veri egitime sokulmaz.
                if degenerate(a):
                    hata["dejenere tekrar"] += 1
                    bad = True
            if bad:
                continue

            if tok is not None:
                metin = f"### Sistem:\n{SYSTEM_PROMPT}\n\n"
                for u, a in turns:
                    metin += (f"### Kullanıcı:\n{u.strip()}\n\n"
                              f"### Asistan:\n{a.strip()}\n\n")
                if len(tok.encode(metin)) > args.block_size:
                    kirpilan_serit += 1

    print(f"dosya : {p}")
    print(f"satir : {n_rec} gecerli kayit")
    if n_tur:
        from statistics import median
        print(f"tur   : medyan {median(n_tur):.0f}, min {min(n_tur)}, max {max(n_tur)}")

    if lens:
        tot = len(lens)
        print(f"\ncevap uzunlugu ({tot} cevap):")
        for lo, hi, ad in LEN_BANDS:
            k = sum(1 for x in lens if lo <= x < hi)
            bar = "#" * round(30 * k / tot)
            print(f"  {ad:8s} {lo:4d}-{hi if hi < 10**9 else '+':>4} "
                  f"{k:6d}  %{100*k/tot:5.1f} {bar}")
        from statistics import median
        print(f"  medyan {median(lens):.0f} karakter")

    print("\nkonu tipi kapsami (ilk tur + ilk cevap uzerinden):")
    sirali = list(COVERAGE.items()) + [("kisa_qa", KISA_QA)]
    for ad, _ in sirali:
        k = coverage[ad]
        pct = 100 * k / n_rec if n_rec else 0
        print(f"  {ad:12s} {k:6d}  %{pct:5.1f} "
              f"{'#' * round(20 * pct / 100)}")

    if hata:
        print("\nHATALAR:")
        for k, v in hata.most_common():
            print(f"  {v:6d}  {k}")
    if uyari:
        print("\nUYARILAR:")
        for k, v in uyari.most_common():
            print(f"  {v:6d}  {k}")
    if kirpilan_serit:
        print(f"\n  {kirpilan_serit} kayit {args.block_size} token'a sigmiyor "
              f"-> egitimde ATLANACAK")
    if bos:
        print(f"\n  {bos} turde bos alan")

    if args.report:
        print("\n--oneriler--")
        if lens:
            tot = len(lens)
            uzun = sum(1 for x in lens if x >= 300) / tot * 100
            if uzun < 15:
                print(f"  uzun cevap orani %{uzun:.1f} -> model kisa konusacak. "
                      f">=300 kr cevap oranini %25-40'a cikart.")
            kisa = sum(1 for x in lens if x < 80) / tot * 100
            if kisa > 55:
                print(f"  kisa cevap orani %{kisa:.1f} -> 'bir cumle deyip susar' "
                      f"aliskanligi ouseniyor.")
        for ad in COVERAGE:
            if n_rec and coverage[ad] / n_rec < 0.03:
                print(f"  '{ad}' konu tipi %3'ten az -> bu davranis ogrenilmeyebilir.")
        if kirpilan_serit and n_rec:
            print(f"  %{100*kirpilan_serit/n_rec:.0f} kayit blok sinirini "
                  f"asiyor --block-size dusun.")

    if hata:
        print("\nSONUC: BASARISIZ - egitime sokma")
        return 1
    print("\nSONUC: GECERLI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())