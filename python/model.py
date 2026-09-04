"""CofeuAI Transformer Modeli.

Sıfırdan yazılmış, decoder-only (GPT benzeri) bir transformer.
Eğitim için PyTorch kullanılır. Model, C++ inference motoruna
aktarılabilmesi için basit ve anlaşılır tutulmuştur.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Generator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("cofeu.model")


@dataclass
class ModelConfig:
    """Model hiperparametreleri.

    Varsayılan değerler RTX 3060 6GB VRAM'e sığacak şekilde optimize edilmiştir
    (~26M parametre, float32).
    """

    vocab_size: int = 2048
    n_embd: int = 512
    n_head: int = 8
    n_layer: int = 8
    block_size: int = 256
    dropout: float = 0.1
    eos_token_id: int = 1


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

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

    def _safe_softmax(self, att: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """NaN korumalı softmax: -inf ile dolu satırları sıfırlar."""
        att_max = att.max(dim=dim, keepdim=True).values
        att = att - att_max
        exp_att = torch.exp(att)
        sum_exp = exp_att.sum(dim=dim, keepdim=True)
        # Tüm satır -inf ise (masked), sıfıra bölme hatasını engelle
        sum_exp = sum_exp.clamp(min=1e-8)
        return exp_att / sum_exp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self._safe_softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

    def forward_with_cache(self, x: torch.Tensor, cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """KV cache destekli attention."""
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

        if cache is not None:
            pos_offset = T_total - T
            mask_rows = self.mask[:, :, pos_offset:pos_offset + T, :T_total]
        else:
            mask_rows = self.mask[:, :, :T, :T_total]
        att = att.masked_fill(mask_rows == 0, float("-inf"))
        att = self._safe_softmax(att, dim=-1)
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

    def forward_with_cache(self, x: torch.Tensor, cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
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

        self.apply(self._init_weights)
        logger.info(
            "Model oluşturuldu: %.2fM parametre",
            sum(p.numel() for p in self.parameters()) / 1e6,
        )

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
        """Otoregresif metin üretimi."""
        tokens_generated = []
        for token in self.generate_stream(idx, max_new_tokens, temperature, top_k, top_p, repetition_penalty, eos_token_id):
            tokens_generated.append(token)
        return torch.cat(tokens_generated, dim=1)

    @torch.no_grad()
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
        """Token-by-token streaming üretim."""
        self.eval()
        idx_cond = idx[:, -self.config.block_size:]
        logits, kv_cache = self.forward_with_cache(idx_cond)

        generated_tokens = idx_cond[0].tolist()

        for _ in range(max_new_tokens):
            logits_next = logits[:, -1, :] / temperature

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated_tokens):
                    if logits_next[0, token_id] > 0:
                        logits_next[0, token_id] /= repetition_penalty
                    else:
                        logits_next[0, token_id] *= repetition_penalty

            # Top-k sampling
            if top_k is not None:
                values, _ = torch.topk(logits_next, min(top_k, logits_next.size(-1)))
                threshold = values[:, -1].unsqueeze(-1)
                logits_next = torch.where(
                    logits_next < threshold,
                    torch.full_like(logits_next, float("-inf")),
                    logits_next,
                )

            # Top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits_next, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits_next = logits_next.masked_fill(indices_to_remove, float("-inf"))

            probs = F.softmax(logits_next, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()

            # EOS kontrolü
            if eos_token_id is not None and token_id == eos_token_id:
                break

            generated_tokens.append(token_id)
            yield next_token

            # Cache güncelle
            if kv_cache[0][0].size(2) >= self.config.block_size:
                kv_cache = [(k[:, :, 1:, :], v[:, :, 1:, :]) for k, v in kv_cache]

            logits, kv_cache = self.forward_with_cache(next_token, kv_cache)

    @torch.no_grad()
    def forward_with_cache(self, idx: torch.Tensor, kv_cache: Optional[list] = None):
        """KV cache destekli ileri geçiş."""
        B, T = idx.shape

        if kv_cache is None:
            pos_offset = 0
        else:
            pos_offset = kv_cache[0][0].size(2)

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
    logging.basicConfig(level=logging.INFO)
    config = ModelConfig(vocab_size=128)
    model = CofeuTransformer(config)
    x = torch.randint(0, 128, (2, 16))
    logits, loss = model(x, x)
    print("Logits shape:", logits.shape)
    print("Loss:", loss.item())
    print("Parametre sayısı:", sum(p.numel() for p in model.parameters()))

    # Streaming test
    prompt = torch.randint(0, 128, (1, 4))
    print("\nStreaming üretim testi:")
    for i, token in enumerate(model.generate_stream(prompt, max_new_tokens=10, temperature=0.8)):
        print(f"  Token {i}: {token.item()}")
