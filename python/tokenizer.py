"""CofeuAI Tokenizer.

Byte-Pair Encoding (BPE) tokenizer. Metni alt-kelime parçalarına böler,
böylece karakter seviyesine göre daha uzun bağlam ve daha verimli üretim
sağlar. C++ inference motoruna kolayca taşınabilir olması için basit
tutulmuştur.

Vocab formatı (JSON):
  {
    "merges": ["a b", "ab c", ...],   # BPE birleştirme kuralları (sıralı)
    "vocab": {"a": 0, "b": 1, ...}    # token -> id
  }
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


class BPETokenizer:
    """Byte-Pair Encoding tokenizer."""

    def __init__(self, merges: List[str] | None = None, vocab: Dict[str, int] | None = None):
        self.merges: List[Tuple[str, str]] = []
        if merges:
            for m in merges:
                parts = m.split(" ")
                if len(parts) == 2:
                    self.merges.append((parts[0], parts[1]))

        self.stoi: Dict[str, int] = dict(vocab) if vocab else {}
        self.itos: Dict[int, str] = {i: c for c, i in self.stoi.items()}

    # --- Eğitim ---

    @classmethod
    def build(cls, text: str, vocab_size: int = 512, max_sample: int = 200000) -> "BPETokenizer":
        """Metinden BPE vocab'ı oluşturur.

        Hız için BPE eğitimini tüm metin yerine bir örneklem üzerinde yapar.
        max_sample: BPE eğitimi için kullanılacak maksimum karakter sayısı.
        """
        # BPE eğitimi için örneklem al (hız için)
        sample = text[:max_sample]

        # Başlangıç: karakter seviyesi (UTF-8 karakterler)
        chars = sorted(set(sample))
        vocab = {c: i for i, c in enumerate(chars)}
        merges: List[Tuple[str, str]] = []

        # Metni karakter listesine çevir
        tokens = list(sample)

        num_merges = vocab_size - len(chars)
        for _ in range(num_merges):
            # Bitişik çiftleri say
            pairs = Counter(zip(tokens, tokens[1:]))
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < 2:
                break

            # Yeni token
            new_token = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            vocab[new_token] = len(vocab)

            # Metinde birleştir
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

        tok = cls(merges=[f"{a} {b}" for a, b in merges], vocab=vocab)
        return tok

    # --- Encode / Decode ---

    def encode(self, text: str) -> List[int]:
        """Metni token id listesine çevirir.

        GPT-2 tarzı tek geçişli BPE: her merge'e bir rank verilir ve
        tokenizasyon sırasında en düşük rank'lı merge uygulanır. Bu,
        merge'leri sırayla uygulamakla aynı sonucu verir ama çok daha hızlıdır.
        """
        # Merge rank'ları: (a, b) -> (rank, ab)
        ranks = {}
        for rank, (a, b) in enumerate(self.merges):
            ranks[(a, b)] = (rank, a + b)

        tokens = list(text)
        result = []
        i = 0
        n = len(tokens)
        while i < n:
            # Bu pozisyondan başlayarak en düşük rank'lı merge'i bul
            best_rank = None
            best_token = None
            best_len = 1
            # En fazla birkaç adım ileriye bak (merge zincirleri için)
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

        return [self.stoi[t] for t in result if t in self.stoi]

    def decode(self, ids: List[int]) -> str:
        """Token id listesini metne çevirir."""
        return "".join(self.itos[i] for i in ids if i in self.itos)

    # --- I/O ---

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def save(self, path: str | Path) -> None:
        """Vocab'ı JSON olarak kaydeder."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "merges": [f"{a} {b}" for a, b in self.merges],
            "vocab": self.stoi,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Vocab'ı JSON'dan yükler."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(merges=data.get("merges", []), vocab=data.get("vocab", {}))


# Geriye dönük uyumluluk: eski kod CharTokenizer kullanıyor.
CharTokenizer = BPETokenizer


if __name__ == "__main__":
    text = "merhaba dünya merhaba dünya merhaba"
    tok = BPETokenizer.build(text, vocab_size=50)
    ids = tok.encode("merhaba dünya")
    print("Vocab size:", tok.vocab_size)
    print("Merges:", tok.merges[:10])
    print("Encoded:", ids)
    print("Decoded:", tok.decode(ids))