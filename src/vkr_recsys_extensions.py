from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class EASERecommender:
    """
    Embarrassingly Shallow AutoEncoder (EASE).

    Good thesis addition because:
    - very strong sparse-data baseline for implicit feedback;
    - closed-form training, so it is easy to explain and reproduce;
    - often competitive with deep models on medium-size catalogs.

    Expects the same PreparedData object as in the notebook:
    data.train_matrix, data.seen_items.
    """

    def __init__(self, lam: float = 300.0):
        self.lam = lam
        self.B = None
        self.X = None

    def fit(self, data):
        X = data.train_matrix.astype(np.float32).tocsr()
        # Binary implicit setup is usually the safest common choice.
        X_bin = (X > 0).astype(np.float32)
        G = (X_bin.T @ X_bin).toarray().astype(np.float64)
        diag_idx = np.arange(G.shape[0])
        G[diag_idx, diag_idx] += self.lam

        P = np.linalg.inv(G)
        B = -P / np.diag(P)
        B[diag_idx, diag_idx] = 0.0

        self.B = B.astype(np.float32)
        self.X = X_bin
        return self

    def recommend(self, user_idx: int, k: int, seen_items: Optional[set] = None, exclude_seen: bool = True) -> List[int]:
        user_row = self.X[user_idx].toarray().ravel().astype(np.float32)
        scores = user_row @ self.B
        if exclude_seen and seen_items:
            scores[list(seen_items)] = -np.inf
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return list(map(int, top_idx))


class GRU4RecDataset(Dataset):
    """
    Next-item dataset from user sequences.
    Uses the same sequence dict as SASRec in the notebook.
    """

    def __init__(self, sequences: Dict[int, List[int]], max_len: int = 50):
        self.samples = []
        self.max_len = max_len
        for uid, seq in sequences.items():
            if len(seq) < 2:
                continue
            for t in range(1, len(seq)):
                prefix = seq[max(0, t - max_len):t]
                target = seq[t]
                self.samples.append((uid, prefix, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        uid, prefix, target = self.samples[idx]
        seq = prefix[-self.max_len:]
        pad_len = self.max_len - len(seq)
        seq = [0] * pad_len + [x + 1 for x in seq]  # +1: 0 is reserved for padding
        return {
            "uid": uid,
            "seq": torch.tensor(seq, dtype=torch.long),
            "target": torch.tensor(target + 1, dtype=torch.long),
        }


class GRU4RecModel(nn.Module):
    def __init__(self, n_items: int, hidden_size: int = 100, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.item_embedding = nn.Embedding(n_items + 1, hidden_size, padding_idx=0)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, n_items + 1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        emb = self.item_embedding(seq)
        out, _ = self.gru(emb)
        last_hidden = out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        return self.output(last_hidden)


class GRU4RecRecommender:
    """
    Lightweight GRU4Rec-style next-item recommender.

    Why it is a good second new model for your thesis:
    - it gives a clean RNN-vs-Transformer comparison against SASRec;
    - it is natural for Instacart / retail purchase sequences;
    - implementation is compact and easy to comment in the thesis.
    """

    def __init__(
        self,
        max_len: int = 50,
        hidden_size: int = 100,
        num_layers: int = 1,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 5,
        batch_size: int = 512,
        device: Optional[str] = None,
    ):
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.n_items = None
        self.sequences = None

    def fit(self, data):
        self.n_items = data.train_matrix.shape[1]
        self.sequences = data.train_sequences
        train_ds = GRU4RecDataset(self.sequences, max_len=self.max_len)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        self.model = GRU4RecModel(
            n_items=self.n_items,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            self.model.train()
            losses = []
            for batch in train_loader:
                seq = batch["seq"].to(self.device)
                target = batch["target"].to(self.device)

                logits = self.model(seq)
                loss = criterion(logits, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            mean_loss = float(np.mean(losses)) if losses else float("nan")
            print(f"GRU4Rec epoch={epoch + 1}/{self.epochs} loss={mean_loss:.6f}")
        return self

    @torch.no_grad()
    def recommend(self, user_idx: int, k: int, seen_items: Optional[set] = None, exclude_seen: bool = True) -> List[int]:
        self.model.eval()
        seq = self.sequences.get(user_idx, [])[-self.max_len:]
        seq = [0] * (self.max_len - len(seq)) + [x + 1 for x in seq]
        seq_tensor = torch.tensor([seq], dtype=torch.long, device=self.device)
        logits = self.model(seq_tensor).squeeze(0).detach().cpu().numpy()
        scores = logits[1:]  # back to 0..n_items-1
        if exclude_seen and seen_items:
            scores[list(seen_items)] = -np.inf
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return list(map(int, top_idx))


def register_new_models(DEFAULT_MODELS, CONFIG, device: Optional[str] = None):
    """
    Convenience helper for the notebook.

    Usage in the notebook after imports:

    from vkr_recsys_extensions import EASERecommender, GRU4RecRecommender, register_new_models
    DEFAULT_MODELS = register_new_models(DEFAULT_MODELS, CONFIG, device=device)
    """
    DEFAULT_MODELS = dict(DEFAULT_MODELS)
    DEFAULT_MODELS["EASE"] = lambda: EASERecommender(lam=300.0)
    DEFAULT_MODELS["GRU4Rec"] = lambda: GRU4RecRecommender(
        max_len=50,
        hidden_size=100,
        num_layers=1,
        dropout=0.1,
        lr=1e-3,
        epochs=5,
        batch_size=CONFIG.get("sasrec_batch_size", 512),
        device=device,
    )
    return DEFAULT_MODELS
