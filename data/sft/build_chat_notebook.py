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
import ast
import base64
import json
import pathlib
from pathlib import Path
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "sft" / "colab_sft_chat.ipynb"

# GitHub'a bagimlilik olmasin diye gomulen dosyalar: repo yolu -> Colab yolu
EMBEDDED = [
    ("python/sft.py", "/content/cofeuai/python/sft.py"),
    ("python/prepare_sft_data.py", "/content/cofeuai/python/prepare_sft_data.py"),
]

# Hucre kaynaklari data/sft/chat_cells/*.md dosyalarinda. Boylece
# /tmp'ye bagimli kalmaz, git farki gorunur ve tek tek incelenebilir.
CELLS_DIR = Path(__file__).resolve().parent / "chat_cells"
CELLS = {f"e{i:02d}": (CELLS_DIR / f"e{i:02d}.md").read_text(encoding="utf-8")
         for i in range(1, 10)}

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


def python_only(src: str) -> str:
    """Colab sihirbazlarini (! / %) at, geriye gercek Python kalir.

    Magikler her zaman satirin BASINDA olmalidir; bu yuzden soldaki bosluk
    olsa bile sihirbaz sayilir (Python'da girintili ifade olur).
    """
    return "\n".join(l for l in src.split("\n")
                     if not l.startswith(("!", "%")))


def validate(srcs):
    """Bozuk hucre uretmeden once hepsini sozdiziminden gecir.

    Gerekce: bir kez `\\n` kacisi unutulunca uc hucre sessizce bozuldu ve
    Colab'da SyntaxError olarak patladi. Uretici artik bunu kendisi yakalar.
    """
    for i, s in enumerate(srcs):
        assert isinstance(s, str) and s.strip(), f"hucre {i} bos"
        code = python_only(s)
        code = "\n".join(l for l in code.split("\n")
                         if l.strip() and not l.lstrip().startswith("#"))
        if not code:
            continue  # saf yorum veya saf shell
        try:
            ast.parse(code)
        except SyntaxError as e:
            raise SystemExit(
                f"Hucre {i} bozuk Python ({e}).\n"
                f"Notebook YAZILMADI. Hucreyi duzeltip tekrar calistir."
            ) from e


def build():
    srcs = list(SEQUENCE)
    for slot, (rel, dest) in zip((3, 4), EMBEDDED):
        srcs[slot] = embed_cell(rel, dest)
    validate(srcs)

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
