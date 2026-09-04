"""CofeuAI Tokenizer.

Byte-Pair Encoding (BPE) tokenizer. Metni alt-kelime parçalarına böler,
böylece karakter seviyesine göre daha uzun bağlam ve daha verimli üretim
sağlar. C++ inference motoruna kolayca taşınabilmesi için basit
tutulmuştur.

Vocab formatı (JSON):
  {
    "merges": ["a b", "ab c", ...],
    "vocab": {"<|bos|>": 0, "<|eos|>": 1, ...},
    "special_tokens": ["<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>"]
  }
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SPECIAL_TOKENS = {
    "<|bos|>": 0,
    "<|eos|>": 1,
    "<|pad|>": 2,
    "<|unk|>": 3,
}


class BPETokenizer:
    """Byte-Pair Encoding tokenizer."""

    def __init__(
        self,
        merges: Optional[List[str]] = None,
        vocab: Optional[Dict[str, int]] = None,
        special_tokens: Optional[List[str]] = None,
    ):
        self.special_tokens: List[str] = special_tokens or list(SPECIAL_TOKENS.keys())
        self.special_ids: Dict[str, int] = {}
        self.special_id_to_token: Dict[int, str] = {}

        self.merges: List[Tuple[str, str]] = []
        if merges:
            for m in merges:
                parts = m.split(" ")
                if len(parts) == 2:
                    self.merges.append((parts[0], parts[1]))

        self.stoi: Dict[str, int] = dict(vocab) if vocab else {}
        self.itos: Dict[int, str] = {i: c for c, i in self.stoi.items()}

        # Special token mapping'lerini güncelle
        for tok in self.special_tokens:
            if tok in self.stoi:
                self.special_ids[tok] = self.stoi[tok]
                self.special_id_to_token[self.stoi[tok]] = tok

    @property
    def bos_id(self) -> int:
        return self.special_ids.get("<|bos|>", -1)

    @property
    def eos_id(self) -> int:
        return self.special_ids.get("<|eos|>", -1)

    @property
    def pad_id(self) -> int:
        return self.special_ids.get("<|pad|>", -1)

    @property
    def unk_id(self) -> int:
        return self.special_ids.get("<|unk|>", -1)

    @classmethod
    def build(cls, text: str, vocab_size: int = 512, max_sample: int = 200000) -> "BPETokenizer":
        """Metinden BPE vocab'ı oluşturur.

        Hız için BPE eğitimini tüm metin yerine bir örneklem üzerinde yapar.
        max_sample: BPE eğitimi için kullanılacak maksimum karakter sayısı.
        """
        sample = text[:max_sample]
        chars = sorted(set(sample))
        vocab: Dict[str, int] = {}
        merges: List[Tuple[str, str]] = []

        tokens = list(sample)
        num_merges = vocab_size - len(chars) - len(SPECIAL_TOKENS)

        for _ in range(num_merges):
            pairs = Counter(zip(tokens, tokens[1:]))
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < 2:
                break

            new_token = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            vocab[new_token] = len(SPECIAL_TOKENS) + len(chars) + len(merges) - 1

            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # Special token'ları en başa ekle, sonra karakterleri, sonra merge'leri
        full_vocab: Dict[str, int] = dict(SPECIAL_TOKENS)
        for i, c in enumerate(chars):
            full_vocab[c] = len(SPECIAL_TOKENS) + i
        for i, (a, b) in enumerate(merges):
            full_vocab[a + b] = len(SPECIAL_TOKENS) + len(chars) + i

        tok = cls(
            merges=[f"{a} {b}" for a, b in merges],
            vocab=full_vocab,
            special_tokens=list(SPECIAL_TOKENS.keys()),
        )
        return tok

    def _is_special(self, token: str) -> bool:
        return token in self.special_ids

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Metni token id listesine çevirir.

        add_special_tokens=True ise başa BOS ekler.
        """
        ranks: Dict[Tuple[str, str], Tuple[int, str]] = {}
        for rank, (a, b) in enumerate(self.merges):
            ranks[(a, b)] = (rank, a + b)

        tokens = list(text)
        result: List[str] = []
        i = 0
        n = len(tokens)
        while i < n:
            best_rank = None
            best_token = None
            best_len = 1
            j = i
            current = tokens[j]
            while j < n - 1:
                pair = (current, tokens[j + 1])
                if pair in ranks:
                    rank, merged = ranks[pair]
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_token = merged
                        best_len = j - i + 2
                    current = merged
                    j += 1
                else:
                    break

            if best_token is not None:
                result.append(best_token)
                i += best_len
            else:
                result.append(tokens[i])
                i += 1

        ids = [self.stoi.get(t, self.unk_id) for t in result if t in self.stoi or t == "<|unk|>"]

        if add_special_tokens and self.bos_id >= 0:
            ids = [self.bos_id] + ids

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Token id listesini metne çevirir."""
        parts = []
        for i in ids:
            if skip_special_tokens and i in self.special_id_to_token:
                continue
            if i in self.itos:
                parts.append(self.itos[i])
        return "".join(parts)

    def decode_with_special(self, ids: List[int]) -> str:
        """Token id listesini metne çevirir (special token'lar dahil)."""
        return self.decode(ids, skip_special_tokens=False)

    def save(self, path: str | Path) -> None:
        """Vocab'ı JSON olarak kaydeder."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "merges": [f"{a} {b}" for a, b in self.merges],
            "vocab": self.stoi,
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Vocab'ı JSON'dan yükler."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            merges=data.get("merges", []),
            vocab=data.get("vocab", {}),
            special_tokens=data.get("special_tokens", list(SPECIAL_TOKENS.keys())),
        )

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)


CharTokenizer = BPETokenizer


if __name__ == "__main__":
    text = "merhaba dünya merhaba dünya merhaba"
    tok = BPETokenizer.build(text, vocab_size=50)
    ids = tok.encode("merhaba dünya", add_special_tokens=True)
    print("Vocab size:", tok.vocab_size)
    print("BOS ID:", tok.bos_id)
    print("EOS ID:", tok.eos_id)
    print("PAD ID:", tok.pad_id)
    print("UNK ID:", tok.unk_id)
    print("Merges:", tok.merges[:10])
    print("Encoded:", ids)
    print("Decoded:", tok.decode(ids))
    print("Decoded (with special):", tok.decode_with_special(ids))
