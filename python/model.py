"""CofeuAI Transformer Modeli.

Sıfırdan yazılmış, decoder-only (GPT benzeri) bir transformer.
Eğitim için PyTorch kullanılır. Model, C++ inference motoruna
aktarılabilmesi için basit ve anlaşılır tutulmuştur.

Mimari notlar
-------------
* **RoPE** ile konum kodlaması. Öğrenilmiş mutlak konum tablosu yoktur, bu
  yüzden bağlam penceresi (KV cache) kaydırıldığında konumlar tutarlı kalır ve
  `block_size` aşımı diye bir durum oluşmaz. C++ tarafı da aynı düzeni
  (interleaved / GPT-J) kullanır.
* **SDPA** (`scaled_dot_product_attention`) ile attention. Flash-attention
  çekirdeği B×H×T×T matrisi hiç materyalize etmediği için bellek ve zaman
  kazandırır; `mask` tamponu ve elle yazılmış `_safe_softmax` tamamen
  gereksizleşir (SDPA -inf satırları doğru şekilde ele alır).
* **Tam sayı tabanlı KV cache**: yeni token yazılırken eskisi kaydırılır,
  her adımda `torch.cat` ile tüm cache kopyalanmaz.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("cofeu.model")


@dataclass
class ModelConfig:
    """Model hiperparametreleri.

    Varsayılanlar RTX 3060 6GB VRAM'e sığacak şekilde seçilmiştir.
    Eğitimde `n_layer` x `n_embd^2` baskın maliyettir; 8x512 ~28M parametre.
    """

    vocab_size: int = 2048
    n_embd: int = 512
    n_head: int = 8
    n_layer: int = 8
    block_size: int = 256
    dropout: float = 0.1
    eos_token_id: int = 1
    rope_base: int = 10000

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) n_head ({self.n_head}) ile bölünmüyor")
        if (self.n_embd // self.n_head) % 2 != 0:
            raise ValueError("head_dim çift olmalı (RoPE)")
        if min(self.vocab_size, self.n_embd, self.n_head, self.n_layer, self.block_size) <= 0:
            raise ValueError("Model boyutları pozitif olmalı")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def n_params(self) -> int:
        """Parametre sayısı (eşit ağırlık bağlama varsayımı yok)."""
        C, L, V = self.n_embd, self.n_layer, self.vocab_size
        # q,k,v,proj = 4C^2 ; fc1,fc2 = 8C^2 ; ln1/ln2 = 4C ; fc1b+fc2b = 5C
        per_layer = 12 * C * C + 9 * C
        return V * C + L * per_layer + 2 * C + V * C


def build_rope_cache(
    head_dim: int, max_positions: int, base: int, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RoPE için (cos, sin) tabloları: (max_positions, head_dim/2).

    Frekanslar ters üstel: theta_i = base^(-2i/head_dim).
    Tablolar kalıcı tampon değildir; model tarafından önbelleğe alınır.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, device=device, dtype=torch.float32) * 2.0 / head_dim)
    )
    pos = torch.arange(max_positions, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # (T, half)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved (GPT-J) RoPE uygulaması.

    Args:
        x: (B, H, T, head_dim)
        cos, sin: (T, head_dim/2)
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class CausalSelfAttention(nn.Module):
    """Maskeli (causal) self-attention katmanı."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.rope_base = config.rope_base

        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.attn_dropout = config.dropout
        # (cos, sin) tabloları cihaz/dtype ile eşleşmeli; tembel üretilir.
        self._rope_cos: Optional[torch.Tensor] = None
        self._rope_sin: Optional[torch.Tensor] = None

    def _ensure_rope(self, position: int, device: torch.device, dtype: torch.dtype) -> None:
        """(cos, sin) tablosunu gerektiği kadar büyütür ve önbelleğe alır."""
        cached = self._rope_cos
        if cached is not None and cached.shape[0] >= position and cached.device == device:
            if cached.dtype != dtype:
                self._rope_cos = cached.to(dtype)
                self._rope_sin = self._rope_sin.to(dtype)
            return
        grow = max(position, 256)
        if cached is not None:
            grow = max(grow, int(cached.shape[0]) * 2)
        self._rope_cos, self._rope_sin = build_rope_cache(
            self.head_dim, grow, self.rope_base, device, dtype
        )

    def _qkv(self, x: torch.Tensor):
        B, T, C = x.shape
        shape = (B, T, self.n_head, self.head_dim)
        return (
            self.query(x).view(shape).transpose(1, 2),
            self.key(x).view(shape).transpose(1, 2),
            self.value(x).view(shape).transpose(1, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self._qkv(x)

        self._ensure_rope(T, x.device, q.dtype)
        cos = self._rope_cos[:T].view(1, 1, T, -1)
        sin = self._rope_sin[:T].view(1, 1, T, -1)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # is_causal=True: SDPA nedensel maskeyi çekirdeğin içinde uygular ve
        # -inf dolu satır hiç oluşmaz. Eski elle yazılmış _safe_softmax tamamen
        # maskeli satırda (-inf) - (-inf) = NaN üretiyordu.
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

    def forward_with_cache(
        self,
        x: torch.Tensor,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        pos_offset: int = 0,
    ):
        """KV cache destekli attention. (projeksiyon, yeni_k, yeni_v) döner."""
        B, T, C = x.shape
        q, k, v = self._qkv(x)

        self._ensure_rope(pos_offset + T, x.device, q.dtype)
        cos = self._rope_cos[pos_offset : pos_offset + T].view(1, 1, T, -1)
        sin = self._rope_sin[pos_offset : pos_offset + T].view(1, 1, T, -1)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            k_prev, v_prev = cache
            k = torch.cat((k_prev, k), dim=2)
            v = torch.cat((v_prev, v), dim=2)

        # Bu adımda eklenen T token cache'in son T girişi. Sorgu, kendi mutlak
        # konumundan eski olan tüm girişleri görmeli:
        #     anahtar_indeksi <= n_keys_base + t
        # n_keys_base == 0 ise saf nedensel maske -> flash çekirdeği.
        n_keys = k.size(2)
        n_keys_base = n_keys - T

        if n_keys_base == 0:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            q_pos = torch.arange(T, device=x.device).view(T, 1)
            k_pos = torch.arange(n_keys, device=x.device).view(1, n_keys)
            mask = k_pos <= q_pos + n_keys_base
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.view(1, 1, T, n_keys))

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), k, v


class MLP(nn.Module):
    """Feed-forward ağı."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # exact erf (varsayılan) — C++ tarafı da std::erf ile aynı formülü
        # kullanır, böylece iki motor birebir aynı sayıyı üretir.
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    """Tek bir transformer bloğu (attention + MLP + residual)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def forward_with_cache(self, x: torch.Tensor, cache=None, pos_offset: int = 0):
        attn_out, k, v = self.attn.forward_with_cache(self.ln1(x), cache, pos_offset)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, k, v


class CofeuTransformer(nn.Module):
    """Ana model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        # Residual akış ölçeklendirmesi (GPT-2): derinlikle artan varyansı önler
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight") or pn.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        logger.info("Model oluşturuldu: %.2fM parametre", config.n_params() / 1e6)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size {self.config.block_size}")

        x = self.drop(self.token_emb(idx))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def forward_with_cache(self, idx: torch.Tensor, kv_cache: Optional[List] = None, pos_offset: int = 0):
        """KV cache destekli ileri geçiş -> (logits, yeni_cache)."""
        x = self.drop(self.token_emb(idx))
        new_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            prev = kv_cache[i] if kv_cache is not None else None
            x, k, v = block.forward_with_cache(x, prev, pos_offset)
            new_cache.append((k, v))
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, new_cache

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Otoregresif metin üretimi (prompt dahil döner)."""
        toks = []
        for token in self.generate_stream(
            idx, max_new_tokens, temperature, top_k, top_p, repetition_penalty, eos_token_id
        ):
            toks.append(token)
        if not toks:
            # Hiç token üretilmedi (ör. ilk örneklem EOS veya max_new_tokens==0):
            # önceki sürümde torch.cat([]) ValueError fırlatıyordu.
            return idx.clone()
        return torch.cat(toks, dim=1)

    @torch.inference_mode()
    def generate_stream(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> Generator[torch.Tensor, None, None]:
        """Token-by-token streaming üretim.

        Yield edilen her tensor (1, 1) şeklindedir (üretilen token).
        """
        if idx.ndim != 2:
            raise ValueError("idx (B, T) şeklinde olmalı")
        if idx.size(0) == 0:
            raise ValueError("idx en az bir dizi içermeli (B > 0)")
        if idx.size(1) == 0:
            raise ValueError(
                "idx en az bir token içermeli (T > 0); boş prompt için BOS "
                "token'ı elle ekleyin"
            )
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens negatif olamaz")

        was_training = self.training
        self.eval()

        B = idx.size(0)
        # Bellek verimliliği için yalnızca son pozisyonun logiti yeter.
        # Prompt block_size'dan uzunsa yalnızca son pencere işlenir; RoPE
        # mutlak konum kullandığı için pencerenin ilk token'ı konum 0 olur ve
        # üretim tutarlı kalır.
        idx_cond = idx[:, -self.config.block_size:]
        logits, kv_cache = self.forward_with_cache(idx_cond)

        # Pencere: mutlak konumları takip et, cache'i block_size'da tut
        pos = idx_cond.size(1)
        generated = idx_cond  # (B, P)

        try:
          for step in range(max_new_tokens):
            # HF sırası: repetition penalty logitlere uygulanır, SONRA
            # sıcaklığa bölünür.
            logits_next = logits[:, -1, :].clone()

            if repetition_penalty != 1.0:
                for b in range(B):
                    seen = torch.unique(generated[b])
                    vals = logits_next[b, seen]
                    logits_next[b, seen] = torch.where(
                        vals > 0, vals / repetition_penalty, vals * repetition_penalty
                    )

            # HF sırası: penalty -> temperature -> top_k -> top_p
            logits_next = logits_next / max(temperature, 1e-6)

            if top_k is not None and top_k > 0:
                k = min(top_k, logits_next.size(-1))
                kth = torch.topk(logits_next, k, dim=-1).values[:, -1:]
                logits_next = logits_next.masked_fill(logits_next < kth, float("-inf"))

            if top_p is not None and 0.0 < top_p < 1.0:
                # Nucleus, HF TopPLogitsWarper ile birebir aynı:
                #   1) logitler azalan sırada dizilir, kümülatif olasılık alınır
                #   2) (cum - p_i) > top_p olanlar işaretlenir
                #   3) maske BİR KAYDIRILIR (böylece en olası token asla düşmez)
                sorted_logits, sorted_idx = torch.sort(logits_next, descending=True, dim=-1)
                probs_sorted = F.softmax(sorted_logits, dim=-1)
                cum = torch.cumsum(probs_sorted, dim=-1)
                remove_sorted = (cum - probs_sorted) > top_p
                remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
                remove_sorted[..., 0] = False
                remove = remove_sorted.scatter(-1, sorted_idx, remove_sorted)
                logits_next = logits_next.masked_fill(remove, float("-inf"))

            probs = F.softmax(logits_next, dim=-1)
            # NaN/inf güvenliği: olasılıklar toplanmazsa tek token'a düş.
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            row_sum = probs.sum(dim=-1, keepdim=True)
            if bool((~(row_sum > 0)).any()):
                probs = torch.where(
                    ~(row_sum > 0), torch.full_like(probs, 1.0 / probs.size(-1)), probs
                )
            next_token = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None and bool((next_token.view(-1) == eos_token_id).all()):
                break

            yield next_token
            generated = torch.cat((generated, next_token), dim=1)

            # Son adımda bir ileri geçiş daha boşuna maliyet (logits kullanılmıyor)
            if step + 1 < max_new_tokens:
                # Kaydırmalı pencere: yeni token EKLENMEDEN ÖNce eskiyi düşür.
                # block_size - 1'e kırpmazsak cache block_size'a ulaştığında
                # bir sonraki forward block_size + 1 anahtara bakar ve pencere
                # sınırını aşar (C++ halka tamponunun tuttuğu semantikten sapar).
                keep = self.config.block_size - 1
                for i, (k, v) in enumerate(kv_cache):
                    if k.size(2) > keep:
                        # keep == 0 iken "-0" tüm diziyi verir; ":0" ile boşalt.
                        sl = slice(-keep, None) if keep > 0 else slice(0, 0)
                        kv_cache[i] = (k[:, :, sl, :], v[:, :, sl, :])

                logits, kv_cache = self.forward_with_cache(
                    next_token, kv_cache, pos_offset=pos
                )
                pos += 1
        finally:
            # Generator erken kapatılırsa (ör. istemci bağlantıyı kesti) de
            # eğitim modu geri konmalı.
            if was_training:
                self.train()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = ModelConfig(vocab_size=128, n_embd=64, n_head=4, n_layer=2, block_size=32)
    model = CofeuTransformer(config)
    x = torch.randint(0, 128, (2, 16))
    logits, loss = model(x, x)
    print("Logits shape:", logits.shape, "Loss:", loss.item())
    print("Parametre sayısı:", model.num_parameters())

    prompt = torch.randint(0, 128, (1, 4))
    print("\nStreaming üretim testi (cache ile):")
    out = []
    for i, token in enumerate(model.generate_stream(prompt, max_new_tokens=40, temperature=0.8)):
        out.append(token.item())
    print("  tokenlar:", out)
    print("\nblock_size aşımı testi (64 token, pencere 32):")
    out2 = []
    for token in model.generate_stream(prompt, max_new_tokens=64, temperature=0.8):
        out2.append(token.item())
    print("  token sayısı:", len(out2), "(beklenen 64)")