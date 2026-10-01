"""C++ kütüphanesini Python'dan çağırmak için ctypes wrapper.

C++ tarafı hızlı tokenizasyon ve inference sağlar, Python tarafı ise
eğitim ve yüksek seviye mantığı yönetir.

Kapsanan C API (cpp/src/*.cpp içindeki `extern "C"` blokları):
    cofeu_tokenizer_load/free/encode/decode/vocab_size/token_count/eos_id
    cofeu_model_load/free/generate/vocab_size/block_size/max_position
    cofeu_model_check_vocab
    cofeu_model_logits
    cofeu_version

Kullanım:
    from cpp_bridge import CppTokenizer, CppModel

    tok = CppTokenizer("checkpoints/vocab.json")
    model = CppModel("checkpoints/cofeu.bin")
    model.check_vocab(tok.vocab_size)
    for tok_id in model.stream(tok.encode("merhaba"), max_tokens=100):
        ...
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
LIB_PATH = ROOT / "cpp" / "build" / "libcofeu.so"

_ERROR_BUF = 1024

# C API geri çağrısı: (token_id, void* user_data)
_TOKEN_CB = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_void_p)


def _load_lib() -> ctypes.CDLL:
    if not LIB_PATH.exists():
        raise RuntimeError(
            f"C++ kütüphanesi bulunamadı: {LIB_PATH}\n"
            "Önce derleyin: cd cpp/build && cmake .. && make"
        )
    return ctypes.CDLL(str(LIB_PATH))


def _decode(buf: ctypes.Array) -> str:
    return bytes(buf).split(b"\x00", 1)[0].decode("utf-8", errors="replace")


def _bind(lib: ctypes.CDLL) -> None:
    """Tüm C API imzalarını tanımlar. İmza yanlışsa bellek bozulması olur,
    bu yüzden tek noktada ve eksiksiz bağlanır."""
    c, i, f, v = ctypes.c_char_p, ctypes.c_int, ctypes.c_float, ctypes.c_void_p
    p_i, pi_i = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)

    sig = {
        # --- tokenizer ---
        "cofeu_tokenizer_load": ([c, c, i], v),
        "cofeu_tokenizer_free": ([v], None),
        "cofeu_tokenizer_encode": ([v, c, pi_i, i], i),
        "cofeu_tokenizer_decode": ([v, pi_i, i, c, i], i),
        "cofeu_tokenizer_decode_size": ([v, pi_i, i], i),
        "cofeu_tokenizer_vocab_size": ([v], i),
        "cofeu_tokenizer_token_count": ([v], i),
        "cofeu_tokenizer_eos_id": ([v], i),
        # --- model ---
        "cofeu_model_load": ([c, c, i], v),
        "cofeu_model_free": ([v], None),
        "cofeu_model_generate": (
            [v, pi_i, i, i, f, ctypes.c_uint, i, f, f, i,
             _TOKEN_CB, v, pi_i, i, c, i],
            i,
        ),
        "cofeu_model_vocab_size": ([v], i),
        "cofeu_model_block_size": ([v], i),
        "cofeu_model_max_position": ([v], i),
        "cofeu_model_check_vocab": ([v, i, c, i], i),
        "cofeu_model_logits": ([v, pi_i, i, ctypes.POINTER(f), i, c, i], i),
        "cofeu_version": ([], c),
    }
    for name, (argtypes, restype) in sig.items():
        try:
            fn = getattr(lib, name)
        except AttributeError as e:
            raise RuntimeError(
                f"{LIB_PATH} içinde '{name}' sembolü yok. Kütüphane eski olabilir; "
                "'cd cpp/build && cmake .. && make' ile yeniden derleyin."
            ) from e
        fn.argtypes = argtypes
        fn.restype = restype


_LIB: Optional[ctypes.CDLL] = None


def library() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = _load_lib()
        _bind(_LIB)
    return _LIB


def version() -> str:
    v = library().cofeu_version()
    return _decode(v) if v else "bilinmiyor"


class _Handle:
    """Serbest bırakma (free) mantığını paylaşan temel sınıf."""

    _FREE_FN = ""

    def __init__(self) -> None:
        self._handle: Optional[int] = None
        self._lib = library()

    def _fail(self, msg: str) -> RuntimeError:
        return RuntimeError(msg)

    def close(self) -> None:
        """Kaynağı serbest bırakır. Birden çok kez çağrılabilir."""
        if getattr(self, "_handle", None):
            getattr(self._lib, self._FREE_FN)(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class CppTokenizer(_Handle):
    """C++ BPE tokenizer (hızlı encode/decode)."""

    _FREE_FN = "cofeu_tokenizer_free"

    def __init__(self, vocab_path: str | Path):
        super().__init__()
        err = ctypes.create_string_buffer(_ERROR_BUF)
        self._handle = self._lib.cofeu_tokenizer_load(
            str(vocab_path).encode(), err, _ERROR_BUF
        )
        if not self._handle:
            raise self._fail(f"Tokenizer yüklenemedi ({vocab_path}): {_decode(err)}")

    def encode(self, text: str) -> List[int]:
        """Metni token id'lerine çevirir. BOS/EOS eklenmez."""
        if not self._handle:
            raise self._fail("kapatılmış tokenizer")
        # En kötü durum: her bayt ayrı token (UTF-8 çok baytlı karakterler
        # daha az token üretir) + bir fazlası.
        cap = max(4, len(text.encode("utf-8")) + 2)
        while True:
            ids = (ctypes.c_int * cap)()
            n = self._lib.cofeu_tokenizer_encode(
                self._handle, text.encode("utf-8"), ids, cap
            )
            if n < cap:
                return list(ids[:n])
            cap *= 2  # taşma olabilir: büyütüp yeniden dene

    def decode(self, ids: Sequence[int]) -> str:
        """Token id'lerini metne çevirir."""
        if not self._handle:
            raise self._fail("kapatılmış tokenizer")
        ids = list(ids)
        n = len(ids)
        if n == 0:
            return ""
        arr = (ctypes.c_int * n)(*ids)
        need = self._lib.cofeu_tokenizer_decode_size(self._handle, arr, n)
        cap = max(need, 8)
        out = ctypes.create_string_buffer(cap)
        self._lib.cofeu_tokenizer_decode(self._handle, arr, n, out, cap)
        return _decode(out)

    @property
    def vocab_size(self) -> int:
        """Token tablosunun boyutu (max id + 1)."""
        if not self._handle:
            raise self._fail("kapatılmış tokenizer")
        return self._lib.cofeu_tokenizer_vocab_size(self._handle)

    @property
    def token_count(self) -> int:
        """Tabloda tanımlı gerçek token sayısı."""
        if not self._handle:
            raise self._fail("kapatılmış tokenizer")
        return self._lib.cofeu_tokenizer_token_count(self._handle)

    @property
    def eos_id(self) -> int:
        if not self._handle:
            raise self._fail("kapatılmış tokenizer")
        return self._lib.cofeu_tokenizer_eos_id(self._handle)


class CppModel(_Handle):
    """C++ transformer inference motoru (RoPE + sliding KV cache)."""

    _FREE_FN = "cofeu_model_free"

    def __init__(self, model_path: str | Path):
        super().__init__()
        err = ctypes.create_string_buffer(_ERROR_BUF)
        self._handle = self._lib.cofeu_model_load(
            str(model_path).encode(), err, _ERROR_BUF
        )
        if not self._handle:
            raise self._fail(f"Model yüklenemedi ({model_path}): {_decode(err)}")

    def check_vocab(self, tokenizer_vocab_size: int) -> None:
        """Tokenizer ile model vocab boyutunun eşleştiğini doğrular."""
        if not self._handle:
            raise self._fail("kapatılmış model")
        err = ctypes.create_string_buffer(_ERROR_BUF)
        if not self._lib.cofeu_model_check_vocab(
            self._handle, tokenizer_vocab_size, err, _ERROR_BUF
        ):
            raise self._fail(_decode(err))

    @property
    def vocab_size(self) -> int:
        if not self._handle:
            raise self._fail("kapatılmış model")
        return self._lib.cofeu_model_vocab_size(self._handle)

    @property
    def block_size(self) -> int:
        if not self._handle:
            raise self._fail("kapatılmış model")
        return self._lib.cofeu_model_block_size(self._handle)

    @property
    def max_position(self) -> int:
        """Bu modelin destekleyebileceği en uzun mutlak konum."""
        if not self._handle:
            raise self._fail("kapatılmış model")
        return self._lib.cofeu_model_max_position(self._handle)

    def logits(self, prompt_ids: Sequence[int]) -> List[float]:
        """Son pozisyonun logitlerini döndürür. Parity testleri ve hata ayıklama
        için; üretim yolu bunu kullanmaz."""
        if not self._handle:
            raise self._fail("kapatılmış model")
        prompt = list(prompt_ids)
        if not prompt:
            raise self._fail("prompt boş olamaz")

        n = len(prompt)
        n_out = self.vocab_size
        prompt_arr = (ctypes.c_int * n)(*prompt)
        out = (ctypes.c_float * n_out)()
        err = ctypes.create_string_buffer(_ERROR_BUF)

        rc = self._lib.cofeu_model_logits(
            self._handle, prompt_arr, n, out, n_out, err, _ERROR_BUF
        )
        if rc != n_out:
            raise self._fail(_decode(err.value) or f"logit hesaplanamadı (rc={rc})")
        return list(out)

    def _run(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        temperature: float,
        seed: int,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        eos_token_id: int,
        on_token: Optional[Callable[[int], None]],
    ) -> List[int]:
        if not self._handle:
            raise self._fail("kapatılmış model")
        prompt = list(prompt_ids)
        if not prompt:
            raise self._fail("prompt boş olamaz")
        if max_new_tokens < 0:
            raise self._fail("max_new_tokens negatif olamaz")

        n = len(prompt)
        # En fazla block_size token işlendiği için üretim bu sınırı aşamaz.
        cap = n + max_new_tokens + 1
        out = (ctypes.c_int * cap)()
        prompt_arr = (ctypes.c_int * n)(*prompt)
        err = ctypes.create_string_buffer(_ERROR_BUF)

        box = {"on_token": on_token}

        def _cb(token_id: int, _user) -> None:
            fn = box["on_token"]
            if fn is not None:
                fn(token_id)

        cb = _TOKEN_CB(_cb)
        rc = self._lib.cofeu_model_generate(
            self._handle, prompt_arr, n, max_new_tokens,
            ctypes.c_float(temperature), ctypes.c_uint(seed),
            int(top_k), ctypes.c_float(top_p), ctypes.c_float(repetition_penalty),
            int(eos_token_id),
            cb if on_token is not None else ctypes.cast(None, _TOKEN_CB),
            None, out, cap, err, _ERROR_BUF,
        )
        if rc < 0:
            raise self._fail(_decode(err) or "üretim başarısız")
        return list(out[:rc])

    def generate(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 0,
        top_p: float = -1.0,
        repetition_penalty: float = 1.0,
        eos_token_id: int = -1,
        seed: int = 0,
    ) -> List[int]:
        """Prompt + üretilen token id'leri (liste)."""
        return self._run(prompt_ids, max_new_tokens, temperature, seed,
                         top_k, top_p, repetition_penalty, eos_token_id, None)

    def stream(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 0,
        top_p: float = -1.0,
        repetition_penalty: float = 1.0,
        eos_token_id: int = -1,
        seed: int = 0,
    ) -> Iterator[int]:
        """Üretilen token id'lerini **anında** üretir (prompt dahil değil).

        C++ motoru üretimi senkron olarak çalıştırır ve her token'da geri
        çağrı verir. Bu geri çağrıyı ayrı bir iş parçacığında toplayıp
        kuyrukla ana iş parçacığına aktarıyoruz; böylece çağıran taraf
        (ör. SSE sunucusu) her token'ı beklemeden alabilir.

        Not: C++ iş parçacığından gelen ctypes geri çağrısı GIL'i kendisi
        alır, bu yüzden bu yöntem Python tarafında thread-safety açısından
        güvenlidir. Aynı CppModel örneği eşzamanlı olarak kullanılmamalıdır.
        """
        import queue
        import threading

        prompt = list(prompt_ids)
        if not prompt:
            raise self._fail("prompt boş olamaz")
        if max_new_tokens < 0:
            raise self._fail("max_new_tokens negatif olamaz")
        if not self._handle:
            raise self._fail("kapatılmış model")

        q: "queue.Queue" = queue.Queue(maxsize=64)
        done = object()
        state: dict = {"error": None}

        def _cb(token_id: int, _user) -> None:
            q.put(int(token_id))

        cb = _TOKEN_CB(_cb)

        def _worker() -> None:
            try:
                self._run_with_cb(cb, prompt, max_new_tokens, temperature,
                                  seed, top_k, top_p, repetition_penalty,
                                  eos_token_id)
            except BaseException as e:  # noqa: BLE001 - ana iş parçacığına taşınır
                state["error"] = e
            finally:
                q.put(done)

        worker = threading.Thread(target=_worker, name="cofeu-generate", daemon=True)
        worker.start()

        emitted = 0
        while True:
            item = q.get()
            if item is done:
                break
            emitted += 1
            yield item

        worker.join(timeout=5.0)
        if state["error"] is not None:
            raise state["error"]
        if emitted > max_new_tokens:
            raise self._fail(
                f"beklenmedik token sayısı: {emitted} > {max_new_tokens}"
            )

    def _run_with_cb(
        self, cb, prompt, max_new_tokens, temperature, seed,
        top_k, top_p, repetition_penalty, eos_token_id,
    ) -> List[int]:
        """C++ generate'i verilen callback ile çalıştırır (çıktı buffer'ına da yazar)."""
        n = len(prompt)
        cap = n + max_new_tokens + 1
        out = (ctypes.c_int * cap)()
        prompt_arr = (ctypes.c_int * n)(*prompt)
        err = ctypes.create_string_buffer(_ERROR_BUF)
        rc = self._lib.cofeu_model_generate(
            self._handle, prompt_arr, n, max_new_tokens,
            ctypes.c_float(temperature), ctypes.c_uint(seed),
            int(top_k), ctypes.c_float(top_p), ctypes.c_float(repetition_penalty),
            int(eos_token_id),
            cb, None, out, cap, err, _ERROR_BUF,
        )
        if rc < 0:
            raise self._fail(_decode(err) or "üretim başarısız")
        return list(out[:rc])


if __name__ == "__main__":
    import time

    print("C++ kütüphanesi:", version())

    vocab = ROOT / "checkpoints" / "vocab.json"
    model_bin = ROOT / "checkpoints" / "cofeu.bin"

    if not vocab.exists():
        print(f"HATA: {vocab} bulunamadı")
        raise SystemExit(1)

    with CppTokenizer(vocab) as tok:
        print(f"Tokenizer: vocab={tok.vocab_size} token={tok.token_count} eos={tok.eos_id}")
        t0 = time.time()
        ids = tok.encode("Bir zamanlar uzak bir köyde")
        t1 = time.time()
        print(f"Encode: {ids}  ({(t1 - t0) * 1000:.2f} ms)")
        print(f"Decode: {tok.decode(ids)!r}")

        if not model_bin.exists():
            print(f"Not: {model_bin} yok, üretim testi atlandı.")
            print("Dışa aktarmak için: python export.py")
            raise SystemExit(0)

        with CppModel(model_bin) as m:
            m.check_vocab(tok.vocab_size)
            print(f"Model: V={m.vocab_size} block={m.block_size} "
                  f"max_pos={m.max_position}")

            n = 50
            t0 = time.time()
            out = m.generate(ids, max_new_tokens=n, temperature=0.8,
                             top_k=40, repetition_penalty=1.1,
                             eos_token_id=tok.eos_id, seed=42)
            t1 = time.time()
            print(f"Generate {n} token: {(t1 - t0) * 1000:.2f} ms "
                  f"({n / max(t1 - t0, 1e-9):.1f} tok/s)")
            print(f"Çıktı: {tok.decode(out)!r}")
