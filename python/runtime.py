"""CofeuAI çalışma zamanı: model + tokenizer yükleme.

`server.py`, `generate.py` ve `train.py` aynı şeyi yapmak zorunda:
tokenerı yükle, modeli yükle, C++ motoru varsa onu tercih et, yoksa PyTorch'a
düş. Bu mantığı tek yerde topluyoruz ki üç dosya birbirinden kaysın.

Öncelik sırası:
    1. C++ motoru (`cofeu.bin`) — çok daha hızlı, önerilen yol
    2. PyTorch (`cofeu.pt`) — referans, CPU'da da çalışır

`cofeu.bin` yoksa hata değildir; PyTorch yolu devreye girer. `cofeu.bin` varsa
ama yüklenemiyorsa (bozuk imza, uyuşmaz vocab, eksik sembol) sunucu çökmez:
kullanıcıya bir uyarı loglanır ve PyTorch yoluna düşülür. Kullanılan motor
`Backend.name` ve `/health` yanıtıyla her zaman görülebilir.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

from tokenizer import BPETokenizer

logger = logging.getLogger("cofeu.runtime")

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints"
CKPT_PATH = CKPT_DIR / "cofeu.pt"
VOCAB_PATH = CKPT_DIR / "vocab.json"
BIN_PATH = CKPT_DIR / "cofeu.bin"


class ModelLoadError(RuntimeError):
    """Model/tokenizer yüklenemediğinde kullanıcıya gösterilebilir hata."""

# RoPE'ye geçişten önceki (pos_emb'li) checkpoint'ler bu anahtarları taşır:
#   pos_emb.weight      -> ogrenilmis mutlak konum gommesi
#   blocks.N.attn.mask  -> SDPA oncesi sabit maske tamponu (buffer)
_LEGACY_KEYS = ("pos_emb", "attn.mask")


def _legacy_keys(state: dict) -> List[str]:
    return [k for k in state if any(leg in k for leg in _LEGACY_KEYS)]


def _reject_legacy(ckpt: dict, path: Path) -> None:
    """Eski (RoPE öncesi) checkpoint'i anlaşılır bir mesajla reddet."""
    state = ckpt.get("model_state") or {}
    stale = _legacy_keys(state)
    if stale:
        raise ModelLoadError(
            f"{path} eski formatta ({', '.join(sorted(set(stale))[:3])} içeriyor). "
            "Bu model RoPE mimarisiyle eğitilmiş olamaz.\n"
            "  → Modeli sıfırdan yeniden eğitin:  python train.py\n"
            "  → Ya da PyTorch checkpoint'ını C++ formatına aktarın:  python export.py"
        )
    if "config" not in ckpt or "model_state" not in ckpt:
        raise ModelLoadError(
            f"{path} beklenen checkpoint biçiminde değil "
            "('config' ve 'model_state' anahtarları eksik)."
        )


@dataclass
class Backend:
    """Yüklenmiş model + tokenizer.

    `kind` ya `"cpp"` ya `"torch"` olur; çağıran taraf `name` ve
    `uses_cpp()` ile hangi yolda olduğunu anlar.
    """

    kind: str
    tokenizer: BPETokenizer
    model: object  # CppModel veya CofeuTransformer
    _lock: threading.Lock

    @property
    def uses_cpp(self) -> bool:
        return self.kind == "cpp"

    @property
    def name(self) -> str:
        return "C++" if self.uses_cpp else "PyTorch"

    def lock(self) -> threading.Lock:
        """Üretim sırasında kilitlenecek kilit.

        C++ motoru ve PyTorch modeli thread-safe DEĞİLDİR: KV cache'i ve
        workspace tamponlarını paylaşırlar. SSE sunucusu eşzamanlı istek
        kabul ettiği için her üretim bu kilit üzerinden geçmelidir.
        """
        return self._lock

    def __enter__(self) -> "Backend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        for obj in (self.model, self.tokenizer):
            close = getattr(obj, "close", None)
            if callable(close):
                close()


def _load_torch(ckpt_path: Path, vocab_path: Path) -> tuple[BPETokenizer, object]:
    import torch

    from model import CofeuTransformer, ModelConfig

    if not ckpt_path.exists():
        raise ModelLoadError(f"Model bulunamadı: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _reject_legacy(ckpt, ckpt_path)

    if vocab_path.exists():
        tokenizer = BPETokenizer.load(vocab_path)
    else:
        # vocab.json yoksa checkpoint içindeki kopya kullanılır.
        tokenizer = BPETokenizer(
            merges=ckpt.get("merges", []),
            vocab=ckpt.get("vocab", {}),
            special_tokens=ckpt.get("special_tokens"),
        )

    config: ModelConfig = ckpt["config"]
    model = CofeuTransformer(config)
    try:
        model.load_state_dict(ckpt["model_state"])
    except RuntimeError as e:
        # Şema uyuşmazlığı: yine de anlaşılır mesaj ver.
        raise ModelLoadError(
            f"{ckpt_path} bu modelin mimarisiyle eşleşmiyor.\n  {e}\n"
            "  → Mimariyi kontrol edin (python train.py --n-embd ... --n-layer ...)\n"
            "  → Veya eski checkpoint ise yeniden eğitin:  python train.py"
        ) from e
    model.eval()
    return tokenizer, model


def _load_cpp(bin_path: Path, vocab_path: Path) -> tuple[BPETokenizer, object]:
    from cpp_bridge import CppModel, CppTokenizer

    if not bin_path.exists():
        raise ModelLoadError(f"C++ modeli bulunamadı: {bin_path}")
    if not vocab_path.exists():
        raise ModelLoadError(f"Vocab bulunamadı: {vocab_path}")

    tokenizer = CppTokenizer(vocab_path)
    try:
        model = CppModel(bin_path)
        model.check_vocab(tokenizer.vocab_size)
    except Exception:
        tokenizer.close()
        raise
    return tokenizer, model


def load_backend(
    prefer_cpp: bool = True,
    device: str = "cpu",
    bin_path: Optional[Path] = None,
    ckpt_path: Optional[Path] = None,
    vocab_path: Optional[Path] = None,
) -> Backend:
    """Modeli yükler; C++ motoru yoksa/bozuksa PyTorch'a düşer.

    Args:
        prefer_cpp: C++ motoru denensin mi.
        device: PyTorch yolunda kullanılacak cihaz (ör. "cuda").
        bin_path: C++ model yolu (varsayılan `checkpoints/cofeu.bin`).
        ckpt_path: PyTorch checkpoint yolu (varsayılan `checkpoints/cofeu.pt`).
        vocab_path: BPE vocab yolu (varsayılan `checkpoints/vocab.json`).

    Raises:
        ModelLoadError: Hiçbir yol da yüklenemezse.
    """
    bin_path = Path(bin_path) if bin_path is not None else BIN_PATH
    ckpt_path = Path(ckpt_path) if ckpt_path is not None else CKPT_PATH
    vocab_path = Path(vocab_path) if vocab_path is not None else VOCAB_PATH
    errors: List[str] = []

    if prefer_cpp:
        try:
            tokenizer, model = _load_cpp(bin_path, vocab_path)
            logger.info("C++ inference motoru yüklendi (%s)", bin_path.name)
            return Backend("cpp", tokenizer, model, threading.Lock())
        except Exception as e:  # yoksa / bozuksa / uyuşmaz vocab / eksik sembol
            # Sessizce düşmek yanıltıcı olurdu: kullanıcıya nedenini söyle.
            logger.warning(
                "C++ motoru kullanılamıyor (%s: %s) — PyTorch yoluna düşülüyor.",
                bin_path.name, e,
            )
            errors.append(f"cpp: {e}")

    try:
        import torch

        tokenizer, model = _load_torch(ckpt_path, vocab_path)
        if device != "cpu" and torch.cuda.is_available():
            model = model.to(device)
        elif device != "cpu":
            logger.warning("CUDA yok, CPU kullanılıyor")
            device = "cpu"
        logger.info(
            "PyTorch modeli yüklendi (%s, %.2fM parametre, cihaz=%s)",
            ckpt_path.name,
            sum(p.numel() for p in model.parameters()) / 1e6,
            next(model.parameters()).device,
        )
        return Backend("torch", tokenizer, model, threading.Lock())
    except Exception as e:
        errors.append(f"torch: {e}")

    raise ModelLoadError(
        "Model yüklenemedi:\n  - " + "\n  - ".join(errors)
        + "\n\nEğitim yapın:  python train.py"
    )


# --- Üretim arayüzü: backend'den bağımsız, iki motor için ortak ---


def _seed_torch(seed: int) -> None:
    """PyTorch RNG'sini tohumla.

    `seed=0` "rastgele" demektir (C++ tarafıyla aynı sözleşme). PyTorch'un
    `manual_seed`'i `None` kabul etmez, bu yüzden 0'da hiç dokunmuyoruz.
    """
    if not seed:
        return
    import torch

    torch.manual_seed(seed)


def generate_ids(
    backend: Backend,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    eos_token_id: int = -1,
    seed: int = 0,
) -> List[int]:
    """Üretilen token id'leri (prompt dahil DEĞİL)."""
    with backend.lock():
        if backend.uses_cpp:
            model = backend.model
            out = model.generate(
                prompt_ids, max_new_tokens, temperature,
                top_k if top_k else 0,
                top_p if top_p is not None else -1.0,
                repetition_penalty, eos_token_id, seed,
            )
            return list(out[len(prompt_ids):])

        import torch

        model = backend.model
        idx = torch.tensor([list(prompt_ids)], dtype=torch.long)
        _seed_torch(seed)
        stream = model.generate_stream(
            idx, max_new_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id if eos_token_id >= 0 else None,
        )
        return [int(t.item()) for t in stream]


def stream_ids(
    backend: Backend,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    eos_token_id: int = -1,
    seed: int = 0,
) -> Iterator[int]:
    """Token id'lerini ürettikçe döndürür (prompt dahil değil).

    Kilit tüm üretim boyunca tutulur: C++ motoru senkron çalışır ve token'ları
    geri çağırma ile verir, bu yüzden çağıran taraf yavaş olsa bile motor
    bloke olmamalı — ama aynı model eşzamanlı iki kez kullanılamaz.
    """
    prompt_ids = list(prompt_ids)
    with backend.lock():
        if backend.uses_cpp:
            yield from backend.model.stream(
                prompt_ids, max_new_tokens, temperature,
                top_k if top_k else 0,
                top_p if top_p is not None else -1.0,
                repetition_penalty, eos_token_id, seed,
            )
        else:
            import torch

            model = backend.model
            idx = torch.tensor([prompt_ids], dtype=torch.long)
            _seed_torch(seed)
            for tok in model.generate_stream(
                idx, max_new_tokens, temperature=temperature,
                top_k=top_k, top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=eos_token_id if eos_token_id >= 0 else None,
            ):
                yield int(tok.item())
