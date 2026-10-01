"""CofeuAI Tokenizer.

Byte-Pair Encoding (BPE) tokenizer. Metni alt-kelime parçalarına böler,
böylece karakter seviyesine göre daha uzun bağlam ve daha verimli üretim
sağlar. C++ inference motoruna kolayca taşınabilmesi için basit
tutulmuştur.

ALGORİTMA (canonical BPE, GPT-2 ile aynı):
  Metin UTF-8 karakterlerine bölünür, ardından merge'ler **rank sırasına
  göre** uygulanır. Her adımda dizide bulunan en düşük rank'lı çift bulunur
  ve tüm geçişleri birleştirilir. Uygulama, sıralı tam taramalar yerine
  bağlı liste + öncelik kuyruğu ile O(n log n) yapılır.

  Bu algoritmanın C++ karşılığı `cpp/src/tokenizer.cpp` içinde birebir
  aynı şekilde uygulanmıştır; iki tokenizer her metin için aynı token
  id'lerini üretir (`python/tokenizer.py` içindeki test bunu doğrular).

Vocab formatı (JSON):
  {
    "version": 2,
    "merges": [["a", "b"], ["ab", "c"], ...],
    "vocab": {"<|bos|>": 0, "<|eos|>": 1, ...},
    "special_tokens": ["<|bos|>", "<|eos|>", "<|pad|>", "<|unk|>"]
  }

  `merges` iki elemanlı string dizileri olarak yazılır; eski tek-string
  formu (`"a b"`) okunabilir durumda tutulur (VOCAB_VERSION=1).
"""

from __future__ import annotations

import json
from collections import Counter
from heapq import heappop, heappush
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


VOCAB_VERSION = 2

SPECIAL_TOKENS = {
    "<|bos|>": 0,
    "<|eos|>": 1,
    "<|pad|>": 2,
    "<|unk|>": 3,
}


def _count_pairs(tokens: Sequence[str]) -> Dict[Tuple[str, str], int]:
    """Baştaki çiftlerin frekans sayımı."""
    return dict(Counter(zip(tokens, tokens[1:])))


def _pair_rank(item) -> tuple:
    """Bir (pair, freq) girdisini sıralama anahtarına çevirir.

    Sıralama: yüksek frekans → uzun token → lexicografik. Üçüncü anahtar
    olmadan eşitlik kırılamaz ve seçim dict ekleme sırasına bağlı kalır.
    """
    (a, b), freq = item
    return (-freq, -(len(a) + len(b)), a, b)


def _train_bpe_merges(
    sample: str, num_merges: int, min_pair_freq: int
) -> List[Tuple[str, str]]:
    """Corpus'tan BPE merge listesi öğrenir.

    Naif yaklaşım her adımda tüm corpus'u yeniden sayar ve yeniden tarar;
    bu O(merges × N) demektir ve milyon karakterlik bir corpus'ta saatler sürer.

    Burada token'lar bir çift yönlü bağlı liste olarak tutulur. Birleştirilen
    çiftte sol düğüm korunur, sağ düğüm silinir — böylece düğüm indeksleri
    **kaymaz**, yani "bu çift nerelerde geçiyor" bilgisi geçersiz kalmadan
    güncellenebilir. Yalnızca birleştirmenin etrafındaki (soldaki ve sağdaki)
    çiftlerin sayımı değişir.

    En sık çiftin tüm geçişleri soldan sağa (deterministik) işlenmek için
    min-heap kullanılır; bayat kayıtlar açılışta atlanır.
    """
    toks = list(sample)
    n = len(toks)
    if n < 2 or num_merges <= 0:
        return []

    nxt = list(range(1, n)) + [-1]
    prv = [-1] + list(range(n - 1))
    alive = bytearray([1]) * n

    counts: Dict[Tuple[str, str], int] = {}
    pos: Dict[Tuple[str, str], List[int]] = {}

    def bump(pair: Tuple[str, str], i: int, delta: int) -> None:
        if delta > 0:
            counts[pair] = counts.get(pair, 0) + 1
            heappush(pos.setdefault(pair, []), i)
        else:
            c = counts.get(pair)
            if c is None:
                return
            if c <= 1:
                counts.pop(pair, None)
            else:
                counts[pair] = c - 1

    for i in range(n - 1):
        bump((toks[i], toks[i + 1]), i, 1)

    merges: List[Tuple[str, str]] = []
    for _ in range(num_merges):
        if not counts:
            break
        # En sık çift. Eşitlikte önce en uzun olan, sonra lexicografik sıra —
        # son anahtar olmadan kural *totel* değildir ve sonuç dict'in eklenme
        # sırasına göre değişirdi (aynı frekansta 4 çift gelebiliyor).
        best = min(counts.items(), key=_pair_rank)[0]
        if counts[best] < min_pair_freq:
            break
        heap = pos.get(best)
        if not heap:
            break
        merges.append(best)
        merged_tok = best[0] + best[1]

        while heap:
            i = heappop(heap)
            if not alive[i]:
                continue
            j = nxt[i]
            if j == -1 or not alive[j]:
                continue
            if toks[i] != best[0] or toks[j] != best[1]:
                continue  # bayat kayıt

            p = prv[i]
            k = nxt[j]

            # Silinen düğümler: i silinmiyor ama toks[i] değişiyor; j tamamen
            # siliniyor. Etkilenen ÜÇ çift de sayımdan düşmeli:
            #   (p, i)  -> soldaki komşuluk
            #   (i, j)  -> best
            #   (j, k)  -> silinen j'nin sağ komşuluğu (yoksa sayım fazla kalır)
            if p != -1:
                bump((toks[p], toks[i]), p, -1)
            bump(best, i, -1)
            if k != -1:
                bump((toks[j], toks[k]), j, -1)

            toks[i] = merged_tok
            alive[j] = 0
            nxt[i] = k
            if k != -1:
                prv[k] = i

            # Yeni komşuluklar
            if p != -1:
                bump((toks[p], toks[i]), p, 1)
            if k != -1:
                bump((toks[i], toks[k]), i, 1)

    return merges


def _parse_merges(raw: Sequence) -> List[Tuple[str, str]]:
    """Merge listesini normalize eder.

    Hâlâ eski (v1) tek-string formunda olan vocab'ları da kabul eder.
    Token'lar boşluk içerebildiği için `["a", "b"]` gösterimi belirsizliği
    ortadan kaldırır.
    """
    merges: List[Tuple[str, str]] = []
    for m in raw:
        if isinstance(m, (list, tuple)):
            if len(m) != 2:
                raise ValueError(f"Geçersiz merge girdisi (2 eleman bekleniyor): {m!r}")
            a, b = m
        else:
            # v1 geriye uyum: "a b" -> ("a", "b"), ilk boşluktan böl
            a, _, b = str(m).partition(" ")
            if not a and not b:
                raise ValueError(f"Geçersiz merge girdisi: {m!r}")
        merges.append((str(a), str(b)))
    return merges


class BPETokenizer:
    """Byte-Pair Encoding tokenizer."""

    def __init__(
        self,
        merges: Optional[Sequence] = None,
        vocab: Optional[Dict[str, int]] = None,
        special_tokens: Optional[List[str]] = None,
    ):
        self.special_tokens: List[str] = special_tokens or list(SPECIAL_TOKENS.keys())
        self.merges: List[Tuple[str, str]] = _parse_merges(merges or [])

        # (a, b) -> rank ; encode() her çağrıda bunu yeniden kurmasın
        self._ranks: Dict[Tuple[str, str], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }

        self.stoi: Dict[str, int] = dict(vocab) if vocab else {}
        self.itos: Dict[int, str] = {i: c for c, i in self.stoi.items()}

        # decode()'de hızlı eleman erişimi için: i -> special mı
        self._special_id_set: set[int] = set()

        for tok in self.special_tokens:
            if tok in self.stoi:
                self._special_id_set.add(self.stoi[tok])

    @property
    def bos_id(self) -> int:
        return self.stoi.get("<|bos|>", -1)

    @property
    def eos_id(self) -> int:
        return self.stoi.get("<|eos|>", -1)

    @property
    def pad_id(self) -> int:
        return self.stoi.get("<|pad|>", -1)

    @property
    def unk_id(self) -> int:
        return self.stoi.get("<|unk|>", -1)

    @classmethod
    def build(
        cls,
        text: str,
        vocab_size: int = 512,
        max_sample: int = 2_000_000,
        min_pair_freq: int = 2,
    ) -> "BPETokenizer":
        """Metinden BPE vocab'ı oluşturur.

        max_sample: BPE eğitimi için kullanılacak maksimum karakter sayısı.
            Corpus'un bütünü tek bir örneklem (sample) üzerinde işlenir, bu
            yüzden varsayılan corpus boyutundan büyük seçilmiştir.

        Not: Sık çift sayımı kademeli (incremental) tutulur ve token'lar bağlı
        liste olarak işlenir; her merge adımında corpus'un tamamı taranmaz.
        Naif O(merges × N) yaklaşımı büyük corpus'ta pratikte kullanılamaz.
        """
        sample = text[:max_sample]
        chars = sorted(set(sample))
        n_special = len(SPECIAL_TOKENS)
        num_merges = vocab_size - len(chars) - n_special

        merges = _train_bpe_merges(sample, num_merges, min_pair_freq)

        # Special token'lar → karakterler → merge'ler
        full_vocab: Dict[str, int] = dict(SPECIAL_TOKENS)
        for i, c in enumerate(chars):
            full_vocab[c] = n_special + i
        for i, (a, b) in enumerate(merges):
            full_vocab[a + b] = n_special + len(chars) + i

        return cls(
            merges=merges,
            vocab=full_vocab,
            special_tokens=list(SPECIAL_TOKENS.keys()),
        )

    def _bpe(self, tokens: List[str]) -> List[str]:
        """Canonical BPE: en düşük rank'lı çifti bağlı listeyle birleştirir.

        Sıralı tam taramalar yerine öncelik kuyruğu kullanır; sonuç aynıdır
        ama O(n log n) yerine O(n^2) değildir.
        """
        n = len(tokens)
        if n < 2 or not self._ranks:
            return tokens

        ranks = self._ranks
        # Bağlı liste: prev[i] = i'nin sol komşusu, nxt[i] = sağ komşusu
        nxt: List[int] = list(range(1, n + 1))
        prev: List[int] = list(range(-1, n - 1))
        alive = bytearray(b"\x01") * n
        heap: List[Tuple[int, int]] = []

        for i in range(n - 1):
            r = ranks.get((tokens[i], tokens[i + 1]))
            if r is not None:
                heappush(heap, (r, i))

        while heap:
            r, i = heappop(heap)
            if not alive[i]:
                continue
            j = nxt[i]
            if j >= n:
                continue
            # Kuyruk bayatlamış olabilir (çift değişmiş / elenmiş)
            if ranks.get((tokens[i], tokens[j])) != r:
                continue

            # i ve j'yi birleştir
            tokens[i] = tokens[i] + tokens[j]
            alive[j] = 0
            k = nxt[j]
            nxt[i] = k
            if k < n:
                prev[k] = i
                rk = ranks.get((tokens[i], tokens[k]))
                if rk is not None:
                    heappush(heap, (rk, i))
            p = prev[i]
            if p >= 0:
                rp = ranks.get((tokens[p], tokens[i]))
                if rp is not None:
                    heappush(heap, (rp, p))

        out: List[str] = []
        i = 0
        while i < n:
            out.append(tokens[i])
            i = nxt[i]
        return out

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Metni token id listesine çevirir.

        Vocab'da olmayan karakterler <|unk|> olarak eşlenir (sessizce düşürülmez).
        add_special_tokens=True ise başa BOS ekler.
        """
        if not text:
            ids: List[int] = []
        else:
            stoi = self.stoi
            unk = self.unk_id
            ids = [
                stoi[t] if t in stoi else unk
                for t in self._bpe(list(text))
            ]
            # unk tanımlı değilse (vocab'da <|unk|> yok) geçersiz id üretilmemeli
            if unk < 0:
                ids = [i for i in ids if i >= 0]

        if add_special_tokens and self.bos_id >= 0:
            ids = [self.bos_id] + ids

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Token id listesini metne çevirir."""
        itos = self.itos
        special = self._special_id_set
        parts: List[str] = []
        for i in ids:
            if skip_special_tokens and i in special:
                continue
            tok = itos.get(i)
            if tok is not None:
                parts.append(tok)
        return "".join(parts)

    def decode_with_special(self, ids: List[int]) -> str:
        """Token id listesini metne çevirir (special token'lar dahil)."""
        return self.decode(ids, skip_special_tokens=False)

    def save(self, path: str | Path) -> None:
        """Vocab'ı JSON olarak kaydeder."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": VOCAB_VERSION,
            "merges": [[a, b] for a, b in self.merges],
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
        """Modelin gömme matrisi için kullanılacak vocab boyutu.

        Token id'leri 0..vocab_size-1 aralığına yayılır; boşluk (hole) varsa
        en büyük id+1 döner, çünkü gömme tablosu id ile indekslenir.
        """
        return max(self.stoi.values(), default=0) + 1

    def validate(self) -> None:
        """Vocab bütünlüğünü doğrular. Hatalıysa ValueError fırlatır."""
        if not self.stoi:
            raise ValueError("Vocab boş")
        max_id = max(self.stoi.values())
        min_id = min(self.stoi.values())
        if min_id != 0:
            raise ValueError(f"Vocab id'leri 0'dan başlamalı, en küçük={min_id}")
        if max_id + 1 != len(self.stoi):
            holes = max_id + 1 - len(self.stoi)
            raise ValueError(f"Vocab'te {holes} boş id var (max_id={max_id}, size={len(self.stoi)})")
        missing = [t for t in SPECIAL_TOKENS if t not in self.stoi]
        if missing:
            raise ValueError(f"Eksik special token'lar: {missing}")
        bad = [(a, b) for a, b in self.merges if a not in self.stoi or (a + b) not in self.stoi]
        if bad:
            raise ValueError(f"Merge girdileri vocab'ta yok (ilk 3): {bad[:3]}")


if __name__ == "__main__":
    import sys
    import time

    text = "merhaba dünya merhaba dünya merhaba merhaba dünya bir zamanlar uzak bir köyde"
    tok = BPETokenizer.build(text, vocab_size=60)
    tok.validate()
    ids = tok.encode("merhaba dünya", add_special_tokens=True)
    print("Vocab size:", tok.vocab_size)
    print("BOS/EOS/PAD/UNK:", tok.bos_id, tok.eos_id, tok.pad_id, tok.unk_id)
    print("Merges:", tok.merges[:10])
    print("Encoded:", ids)
    print("Decoded:", tok.decode(ids))
    print("OOV -> unk:", tok.encode("🦄🦄"))

    # Python <-> C++ parity (libcofeu.so derliyse)
    lib = Path(__file__).resolve().parent.parent / "cpp" / "build" / "libcofeu.so"
    if lib.exists():
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from cpp_bridge import CppTokenizer

        sample = open(sys.argv[1], encoding="utf-8").read()[:200_000] if len(sys.argv) > 1 else text
        t0 = time.time()
        py = tok.encode(sample)
        t_py = time.time() - t0

        cpath = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("vocab.json")
        if not cpath.exists():
            cpath = Path("/tmp/cofeu_vocab_test.json")
            tok.save(cpath)
        ctok = CppTokenizer(cpath)
        t0 = time.time()
        cp = ctok.encode(sample)
        t_cpp = time.time() - t0
        try:
            print(f"Parity: python={len(py)} token ({t_py:.2f}s) / cpp={len(cp)} token ({t_cpp:.2f}s) -> EŞİT={py == cp}")
        finally:
            ctok.close()
            if cpath.name == "cofeu_vocab_test.json":
                cpath.unlink()