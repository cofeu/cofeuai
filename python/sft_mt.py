#!/usr/bin/env python3
"""Instruction tuning (SFT) - Çok turlu (multi-turn) sohbet verisi ile."""
from __future__ import annotations
import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

ROOT = Path(__file__).parent.parent
CHECKPOINTS = ROOT / "checkpoints"
SYSTEM_PROMPT = "Sen yardımcı bir yapay zeka asistanısın."
EOS_STR = "<|eos|>"

@dataclass
class TrainingState:
    config: dict
    model_state: dict
    optimizer_state: Optional[dict]
    iteration: int
    best_val_loss: float
    last_val_loss: float
    rng_state: Optional[bytes]


def set_seed(s: int = 1234):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_conversation(turns: List[Tuple[str, str]]) -> Tuple[List[int], List[int]]:
    from tokenizer import BPETokenizer
    tok = BPETokenizer.load(str(CHECKPOINTS / "vocab.json"))
    text = f"### Sistem:\n{SYSTEM_PROMPT}\n\n"
    ids_list = tok.encode(text)
    labels_list = [-100] * len(ids_list)
    for user, assistant in turns:
        utext = f"### Kullanıcı:\n{user.strip()}\n\n### Asistan:\n"
        uids = tok.encode(utext)
        ids_list.extend(uids)
        labels_list.extend([-100] * len(uids))
        atext = f"{assistant.strip()}{EOS_STR}"
        aids = tok.encode(atext)
        ids_list.extend(aids)
        labels_list.extend(aids)
    return ids_list, labels_list

class MultiTurnDataset(Dataset):
    def __init__(self, data_path: Path, block_size: int):
        self.data_path = data_path
        self.block_size = block_size
        self.examples = []
        self._load()

    def _load(self):
        from tokenizer import BPETokenizer
        tok = BPETokenizer.load(str(CHECKPOINTS / "vocab.json"))
        with open(self.data_path, encoding='utf-8') as f:
            for i, ln in enumerate(f, 1):
                if not ln.strip():
                    continue
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                turns = self._parse_turns(rec)
                if not turns:
                    continue
                ids, labels = build_conversation(turns)
                if len(ids) > self.block_size:
                    # pencerelere böl (basit)
                    # ama önce son tam turu korumaya calis
                    pass
                if len(ids) <= self.block_size:
                    self.examples.append((ids, labels))
                else:
                    # böl
                    self.examples.append((ids[:self.block_size], labels[:self.block_size]))
        random.shuffle(self.examples)

    def _parse_turns(self, rec):
        if isinstance(rec.get('turns'), list):
            out = []
            for t in rec['turns']:
                if isinstance(t, (list, tuple)) and len(t) == 2:
                    u, a = t
                    if isinstance(u, str) and isinstance(a, str) and a.strip():
                        out.append((u, a))
            return out or None
        convs = rec.get('conversations')
        if isinstance(convs, list) and convs:
            out, pending = [], None
            for m in convs:
                if not isinstance(m, dict):
                    continue
                frm = m.get('from') or m.get('role') or ''
                val = m.get('value') or m.get('content') or m.get('text') or ''
                if frm in ('human', 'user', 'kullanıcı', 'kullanici'):
                    pending = val
                elif frm in ('gpt', 'assistant', 'asistan', 'bot') and pending is not None:
                    out.append((pending, val))
                    pending = None
            return out or None
        return None

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids, labels = self.examples[idx]
        x = torch.tensor(ids, dtype=torch.long)
        y = torch.tensor(labels, dtype=torch.long)
        return x, y

def pad_collate(batch):
    if len(batch) == 0:
        return None, None
    xs, ys = zip(*batch)
    maxl = max(x.shape[0] for x in xs)
    xpad = torch.full((len(xs), maxl), 0, dtype=torch.long)
    ypad = torch.full((len(xs), maxl), -100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        xpad[i, :x.shape[0]] = x
        ypad[i, :y.shape[0]] = y
    return xpad, ypad

def compute_loss(model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
    return loss


def save_state(path: Path, model, optimizer, state: TrainingState):
    torch.save(
        {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
            'iteration': state.iteration,
            'best_val_loss': state.best_val_loss,
            'last_val_loss': state.last_val_loss,
            'config': state.config,
            'rng_state': torch.get_rng_state().cpu().numpy().tobytes(),
        },
        path,
    )


def load_state(path: Path, model, optimizer):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model_state'])
    if optimizer is not None and ck.get('optimizer_state') is not None:
        optimizer.load_state_dict(ck['optimizer_state'])
    if ck.get('rng_state') is not None:
        try:
            torch.set_rng_state(torch.ByteTensor(np.frombuffer(ck['rng_state'], dtype=np.uint8)))
        except Exception:
            pass
    return ck

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--resume', type=str, required=True)
    ap.add_argument('--out-dir', type=str, required=True)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--block-size', type=int, default=512)
    ap.add_argument('--eval-interval', type=int, default=50)
    ap.add_argument('--save-interval', type=int, default=100)
    ap.add_argument('--log-interval', type=int, default=10)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    set_seed(1234)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'train.log'

    from model import CofeuTransformer, ModelConfig
    from tokenizer import BPETokenizer
    tok = BPETokenizer.load(str(CHECKPOINTS / 'vocab.json'))

    ck = torch.load(args.resume, map_location='cpu', weights_only=False)
    cfg = ModelConfig(**ck['config'].__dict__)
    model = CofeuTransformer(cfg)
    model.load_state_dict(ck['model_state'])
    model.to(args.device)
    model.train()

    train_ds = MultiTurnDataset(Path(args.data), args.block_size)
    val_ds = MultiTurnDataset(Path(args.data), args.block_size)  # basit: aynı veri (küçük)
    # split
    n = len(train_ds)
    idx = list(range(n))
    random.shuffle(idx)
    split = max(1, n // 10)
    val_idx = idx[:split]
    train_idx = idx[split:]
    # build
    train_ex = [train_ds.examples[i] for i in train_idx]
    val_ex = [train_ds.examples[i] for i in val_idx]
    train_ds.examples = train_ex
    val_ds.examples = val_ex

    from torch.utils.data import DataLoader
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    steps = len(train_dl) * args.epochs
    state = TrainingState(
        config=cfg.__dict__,
        model_state={},
        optimizer_state=None,
        iteration=0,
        best_val_loss=float('inf'),
        last_val_loss=float('inf'),
        rng_state=None,
    )

    def log(msg):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f"{ts} INFO {msg}"
        print(line)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    log(f"Cihaz: {args.device}")
    log(f"Train: {len(train_ds)} val: {len(val_ds)} steps: {steps}")
    it = 0
    for epoch in range(args.epochs):
        for xb, yb in train_dl:
            if xb is None:
                continue
            xb, yb = xb.to(args.device), yb.to(args.device)
            loss = compute_loss(model, xb, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
            it += 1
            state.iteration = it
            if it % args.log_interval == 0:
                log(f"step {it}/{steps} | loss {loss.item():.4f} | lr {opt.param_groups[0]['lr']:.2e}")
            if it % args.eval-interval == 0 or it == steps:
                # eval
                model.eval()
                vloss = 0
                vc = 0
                with torch.no_grad():
                    for xb, yb in val_dl:
                        if xb is None:
                            continue
                        xb, yb = xb.to(args.device), yb.to(args.device)
                        l = compute_loss(model, xb, yb)
                        vloss += l.item() * xb.size(0)
                        vc += xb.size(0)
                vloss = vloss / vc if vc else 0
                model.train()
                state.last_val_loss = vloss
                if vloss < state.best_val_loss:
                    state.best_val_loss = vloss
                    save_state(out_dir / 'cofeu_sft.pt', model, opt, state)
                log(f"  >> VAL {vloss:.4f} (best {state.best_val_loss:.4f})")
            if it % args.save_interval == 0 and it > 0:
                save_state(out_dir / f'cofeu_sft_{it}.pt', model, opt, state)
    save_state(out_dir / 'cofeu_sft.pt', model, opt, state)
    log("DONE")


if __name__ == '__main__':
    main()
