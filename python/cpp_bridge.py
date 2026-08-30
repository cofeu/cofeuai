"""C++ kütüphanesini Python'dan çağırmak için ctypes wrapper.

C++ tarafı hızlı tokenizasyon ve inference sağlar, Python tarafı ise
eğitim ve yüksek seviye mantığı yönetir.

Kullanım:
    from cpp_bridge import CppTokenizer, CppModel

    tok = CppTokenizer("checkpoints/vocab.json")
    ids = tok.encode("merhaba dünya")

    model = CppModel("checkpoints/cofeu.bin")
    out = model.generate(ids, max_tokens=100)
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
LIB_PATH = ROOT / "cpp" / "build" / "libcofeu.so"


def _load_lib():
    if not LIB_PATH.exists():
        raise RuntimeError(
            f"C++ kütüphanesi bulunamadı: {LIB_PATH}\n"
            "Önce derleyin: cd cpp/build && cmake .. && make"
        )
    return ctypes.CDLL(str(LIB_PATH))


class CppTokenizer:
    """C++ BPE tokenizer (hızlı encode/decode)."""

    def __init__(self, vocab_path: str | Path):
        self._lib = _load_lib()
        self._lib.cofeu_tokenizer_load.argtypes = [ctypes.c_char_p]
        self._lib.cofeu_tokenizer_load.restype = ctypes.c_void_p
        self._lib.cofeu_tokenizer_free.argtypes = [ctypes.c_void_p]
        self._lib.cofeu_tokenizer_encode.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ]
        self._lib.cofeu_tokenizer_encode.restype = ctypes.c_int
        self._lib.cofeu_tokenizer_decode.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.c_char_p, ctypes.c_int,
        ]
        self._lib.cofeu_tokenizer_vocab_size.argtypes = [ctypes.c_void_p]
        self._lib.cofeu_tokenizer_vocab_size.restype = ctypes.c_int

        self._handle = self._lib.cofeu_tokenizer_load(str(vocab_path).encode())
        if not self._handle:
            raise RuntimeError(f"Tokenizer yüklenemedi: {vocab_path}")

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            self._lib.cofeu_tokenizer_free(self._handle)

    def encode(self, text: str) -> List[int]:
        n = len(text) + 1
        ids = (ctypes.c_int * n)()
        count = self._lib.cofeu_tokenizer_encode(
            self._handle, text.encode("utf-8"), ids, n
        )
        return list(ids[:count])

    def decode(self, ids: List[int]) -> str:
        n = len(ids)
        id_arr = (ctypes.c_int * n)(*ids)
        # Çıktı buffer'ı: her token en fazla birkaç byte
        out = ctypes.create_string_buffer(n * 8 + 1)
        self._lib.cofeu_tokenizer_decode(self._handle, id_arr, n, out, n * 8 + 1)
        return out.value.decode("utf-8")

    @property
    def vocab_size(self) -> int:
        return self._lib.cofeu_tokenizer_vocab_size(self._handle)


class CppModel:
    """C++ transformer inference motoru (hızlı üretim)."""

    def __init__(self, model_path: str | Path):
        self._lib = _load_lib()
        self._lib.cofeu_model_load.argtypes = [ctypes.c_char_p]
        self._lib.cofeu_model_load.restype = ctypes.c_void_p
        self._lib.cofeu_model_free.argtypes = [ctypes.c_void_p]
        self._lib.cofeu_model_generate.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.c_int, ctypes.c_float, ctypes.c_uint,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ]
        self._lib.cofeu_model_generate.restype = ctypes.c_int

        self._handle = self._lib.cofeu_model_load(str(model_path).encode())
        if not self._handle:
            raise RuntimeError(f"Model yüklenemedi: {model_path}")

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            self._lib.cofeu_model_free(self._handle)

    def generate(self, prompt_ids: List[int], max_tokens: int = 100,
                 temperature: float = 0.8, seed: int = 42) -> List[int]:
        n = len(prompt_ids)
        prompt_arr = (ctypes.c_int * n)(*prompt_ids)
        out = (ctypes.c_int * (n + max_tokens))()
        count = self._lib.cofeu_model_generate(
            self._handle, prompt_arr, n, max_tokens, temperature, seed,
            out, n + max_tokens
        )
        return list(out[:count])


if __name__ == "__main__":
    # Hızlı test
    import time

    vocab = ROOT / "checkpoints" / "vocab.json"
    model = ROOT / "checkpoints" / "cofeu.bin"

    if vocab.exists():
        tok = CppTokenizer(vocab)
        print("Vocab size:", tok.vocab_size)
        t0 = time.time()
        ids = tok.encode("Bir zamanlar uzak bir köyde")
        t1 = time.time()
        print("Encode:", ids, f"({(t1-t0)*1000:.2f} ms)")
        print("Decode:", tok.decode(ids))

    if model.exists():
        m = CppModel(model)
        t0 = time.time()
        out = m.generate(tok.encode("Bir zamanlar"), max_tokens=50)
        t1 = time.time()
        print(f"Generate 50 token: {(t1-t0)*1000:.2f} ms")
        print("Çıktı:", tok.decode(out))