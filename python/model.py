"""CofeuAI Transformer Modeli.

Sıfırdan yazılmış, decoder-only (GPT benzeri) bir transformer.
Eğitim için PyTorch kullanılır. Model, C++ inference motoruna
aktarılabilmesi için basit ve anlaşılır tutulmuştur.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Model hiperparametreleri.

    Varsayılan değerler RTX 3060 6GB VRAM'e sığacak şekilde optimize edilmiştir
    (~26M parametre, float32).
    """

    vocab_size: int = 2048
    n_embd: int = 512          # embedding boyutu
    n_head: int = 8            # attention başlık sayısı
    n_layer: int = 8           # transformer blok sayısı
    block_size: int = 256      # maksimum bağlam uzunluğu
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Maskeli (causal) self-attention katmanı."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        # Causal mask (üst üçgen -inf)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

    def forward_with_cache(self, x: torch.Tensor, cache=None):
        """KV cache destekli attention.

        cache: (k, v) tuple. None ise tüm diziyi işler.
        Döndürür: (output, k, v)
        """
        B, T, C = x.shape

        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if cache is not None:
            k_prev, v_prev = cache
            k = torch.cat((k_prev, k), dim=2)
            v = torch.cat((v_prev, v), dim=2)

        T_total = k.size(2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        # Causal mask: her sorgu pozisyonu, kendi pozisyonuna kadar olan
        # anahtarlara bakabilir. Cache varsa sorgu pozisyonları
        # [pos_offset, pos_offset+T) aralığındadır.
        if cache is not None:
            pos_offset = T_total - T
            mask_rows = self.mask[:, :, pos_offset:pos_offset + T, :T_total]
        else:
            mask_rows = self.mask[:, :, :T, :T_total]
        att = att.masked_fill(mask_rows == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), k, v


class MLP(nn.Module):
    """Feed-forward ağı."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)


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

    def forward_with_cache(self, x: torch.Tensor, cache=None):
        """KV cache destekli blok ileri geçişi."""
        attn_out, k, v = self.attn.forward_with_cache(self.ln1(x), cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, k, v


class CofeuTransformer(nn.Module):
    """Ana model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Ağırlık başlatma
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape

        tok = self.token_emb(idx)
        pos = torch.arange(0, T, device=idx.device)
        pos = self.pos_emb(pos)
        x = self.drop(tok + pos)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0):
        """Otoregresif metin üretimi (KV cache ile hızlandırılmış)."""
        # İlk adım: tüm prompt'u işle, KV cache'i doldur
        idx_cond = idx[:, -self.config.block_size:]
        logits, kv_cache = self.forward_with_cache(idx_cond)

        for _ in range(max_new_tokens):
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

            # Cache block_size'ı aşarsa en eski token'ı düşür
            if kv_cache[0][0].size(2) >= self.config.block_size:
                kv_cache = [
                    (k[:, :, 1:, :], v[:, :, 1:, :]) for k, v in kv_cache
                ]

            # Sadece yeni token'ı işle, cache'i kullan
            logits, kv_cache = self.forward_with_cache(next_token, kv_cache)
        return idx

    @torch.no_grad()
    def forward_with_cache(self, idx: torch.Tensor, kv_cache=None):
        """KV cache destekli ileri geçiş.

        kv_cache: her blok için (K, V) tuple listesi. None ise tüm diziyi işler.
        """
        B, T = idx.shape

        # Pozisyon ofseti (cache varsa devam eden pozisyonlar)
        if kv_cache is None:
            pos_offset = 0
        else:
            pos_offset = kv_cache[0][0].size(2)  # mevcut cache uzunluğu

        tok = self.token_emb(idx)
        pos = torch.arange(pos_offset, pos_offset + T, device=idx.device)
        pos = self.pos_emb(pos)
        x = self.drop(tok + pos)

        new_cache = []
        for i, block in enumerate(self.blocks):
            if kv_cache is None:
                x, k, v = block.forward_with_cache(x)
            else:
                x, k, v = block.forward_with_cache(x, kv_cache[i])
            new_cache.append((k, v))

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, new_cache


if __name__ == "__main__":
    config = ModelConfig(vocab_size=128)
    model = CofeuTransformer(config)
    x = torch.randint(0, 128, (2, 16))
    logits, loss = model(x, x)
    print("Logits shape:", logits.shape)
    print("Loss:", loss.item())
    print("Parametre sayısı:", sum(p.numel() for p in model.parameters()))