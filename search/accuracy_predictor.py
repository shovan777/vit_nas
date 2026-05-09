"""Accuracy predictor for vit_nas NAS.

Two predictors are provided:
  - MLPPredictor   : small MLP (PyTorch), trained with MSE loss
  - GBMPredictor   : gradient-boosted trees (scikit-learn), fit in seconds

Both share the same ArchEncoder that converts a variable-length config dict
into a fixed-size float feature vector suitable for any sklearn / torch model.

Typical usage:
    from search.accuracy_predictor import ArchEncoder, MLPPredictor, GBMPredictor

    encoder   = ArchEncoder(search_space)
    predictor = MLPPredictor(encoder)
    predictor.fit(train_configs, train_accs)
    preds = predictor.predict(val_configs)   # list[float], same length

    predictor.save("mlp_predictor.pth")
    predictor = MLPPredictor.load("mlp_predictor.pth", encoder)
"""

from __future__ import annotations

import json
import pickle
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Arch Encoder
# ---------------------------------------------------------------------------

class ArchEncoder:
    """Convert a config dict into a fixed-length float feature vector.

    Encoding scheme (all one-hot per option):
      - embed_dim  : one-hot over embed_dim_options
      - num_layers : one-hot over num_layers_options
      - per-layer  : for each of max_layers positions:
                       num_heads one-hot + mlp_dim one-hot
                       + 1 "active" bit (1 if layer <= num_layers, else 0)

    The vector length is fixed regardless of num_layers, so variable-depth
    configs can all be encoded into the same dimensionality.
    """

    def __init__(self, search_space):
        self.ss       = search_space
        self.max_L    = max(search_space.num_layers_options)

        # precompute sorted option lists for stable indexing
        self.embed_opts = sorted(search_space.embed_dim_options)
        self.head_opts  = sorted(search_space.num_heads_options)
        self.mlp_opts   = sorted(search_space.mlp_dim_options)
        self.layer_opts = sorted(search_space.num_layers_options)

        self.dim = (
            len(self.embed_opts)                            # embed_dim
            + len(self.layer_opts)                          # num_layers
            + self.max_L * (len(self.head_opts) + len(self.mlp_opts) + 1)  # per-layer
        )

    def encode(self, config: dict) -> np.ndarray:
        vec = []
        L   = config["num_layers"]

        # embed_dim one-hot
        vec += self._one_hot(config["embed_dim"], self.embed_opts)

        # num_layers one-hot
        vec += self._one_hot(L, self.layer_opts)

        # per-layer encoding (always max_L slots)
        heads = config["num_heads"]   # length == L
        mlps  = config["mlp_dim"]     # length == L
        for i in range(self.max_L):
            if i < L:
                vec += self._one_hot(heads[i], self.head_opts)
                vec += self._one_hot(mlps[i],  self.mlp_opts)
                vec += [1.0]   # active bit
            else:
                vec += [0.0] * len(self.head_opts)
                vec += [0.0] * len(self.mlp_opts)
                vec += [0.0]   # inactive

        return np.array(vec, dtype=np.float32)

    def encode_batch(self, configs: List[dict]) -> np.ndarray:
        return np.stack([self.encode(c) for c in configs])

    # ------------------------------------------------------------------
    @staticmethod
    def _one_hot(value, options: list) -> list:
        return [1.0 if v == value else 0.0 for v in options]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "ArchEncoder":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# MLP Predictor
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, n_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPPredictor:
    """MLP-based accuracy predictor.

    Args:
        encoder     : ArchEncoder instance
        hidden      : hidden layer width
        n_layers    : number of linear layers (including output)
        dropout     : dropout rate
        lr          : learning rate
        epochs      : training epochs
        batch_size  : mini-batch size
    """

    def __init__(self, encoder: ArchEncoder, hidden: int = 256,
                 n_layers: int = 3, dropout: float = 0.1,
                 lr: float = 1e-3, epochs: int = 300, batch_size: int = 64):
        self.encoder    = encoder
        self.hidden     = hidden
        self.n_layers   = n_layers
        self.dropout    = dropout
        self.lr         = lr
        self.epochs     = epochs
        self.batch_size = batch_size
        self.model_     = None   # set after fit()

    def fit(self, configs: List[dict], accs: List[float],
            val_configs: List[dict] = None, val_accs: List[float] = None,
            verbose: bool = True):
        """Train the MLP on (configs, accs) pairs.

        accs should be in [0, 100] (percent).
        """
        X = torch.tensor(self.encoder.encode_batch(configs))
        y = torch.tensor(accs, dtype=torch.float32)

        dataset = TensorDataset(X, y)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net    = _MLP(self.encoder.dim, self.hidden, self.n_layers, self.dropout).to(device)
        opt    = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        loss_fn = nn.MSELoss()

        for epoch in range(self.epochs):
            net.train()
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(net(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
            sched.step()

            if verbose and (epoch + 1) % 50 == 0:
                rmse = (total / len(X)) ** 0.5
                msg  = f"  epoch {epoch+1:>4}/{self.epochs}  train RMSE={rmse:.3f}%"
                if val_configs is not None:
                    val_preds = self._predict_tensor(net, val_configs, device)
                    val_rmse  = float(np.sqrt(np.mean((val_preds - np.array(val_accs)) ** 2)))
                    tau       = _kendall_tau(val_preds, val_accs)
                    msg      += f"  val RMSE={val_rmse:.3f}%  Kendall-τ={tau:.3f}"
                print(msg)

        self.model_  = net
        self._device = device
        return self

    def predict(self, configs: List[dict]) -> List[float]:
        assert self.model_ is not None, "Call fit() before predict()"
        return self._predict_tensor(self.model_, configs, self._device).tolist()

    def _predict_tensor(self, net, configs, device):
        X = torch.tensor(self.encoder.encode_batch(configs)).to(device)
        net.eval()
        with torch.no_grad():
            return net(X).cpu().numpy()

    def save(self, path: str):
        torch.save({
            "state_dict": self.model_.state_dict(),
            "hidden":     self.hidden,
            "n_layers":   self.n_layers,
            "dropout":    self.dropout,
            "encoder":    self.encoder,
        }, path)
        print(f"Saved MLPPredictor → {path}")

    @staticmethod
    def load(path: str) -> "MLPPredictor":
        data    = torch.load(path, map_location="cpu")
        enc     = data["encoder"]
        pred    = MLPPredictor(enc, data["hidden"], data["n_layers"], data["dropout"])
        net     = _MLP(enc.dim, data["hidden"], data["n_layers"], data["dropout"])
        net.load_state_dict(data["state_dict"])
        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pred.model_  = net.to(device)
        pred._device = device
        return pred


# ---------------------------------------------------------------------------
# GBM Predictor (scikit-learn)
# ---------------------------------------------------------------------------

class GBMPredictor:
    """Gradient-Boosted Machine accuracy predictor (sklearn GradientBoostingRegressor).

    Fits in seconds; useful as a fast baseline against the MLP.
    """

    def __init__(self, encoder: ArchEncoder, n_estimators: int = 500,
                 max_depth: int = 4, learning_rate: float = 0.05,
                 subsample: float = 0.8):
        self.encoder       = encoder
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate
        self.subsample     = subsample
        self.model_        = None

    def fit(self, configs: List[dict], accs: List[float],
            val_configs: List[dict] = None, val_accs: List[float] = None,
            verbose: bool = True):
        from sklearn.ensemble import GradientBoostingRegressor
        X = self.encoder.encode_batch(configs)
        y = np.array(accs)

        self.model_ = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            random_state=0,
        ).fit(X, y)

        if verbose:
            train_preds = self.model_.predict(X)
            train_rmse  = float(np.sqrt(np.mean((train_preds - y) ** 2)))
            msg = f"  GBM train RMSE={train_rmse:.3f}%"
            if val_configs is not None:
                val_preds = self.predict(val_configs)
                val_rmse  = float(np.sqrt(np.mean((np.array(val_preds) - np.array(val_accs)) ** 2)))
                tau       = _kendall_tau(val_preds, val_accs)
                msg      += f"  val RMSE={val_rmse:.3f}%  Kendall-τ={tau:.3f}"
            print(msg)
        return self

    def predict(self, configs: List[dict]) -> List[float]:
        assert self.model_ is not None, "Call fit() before predict()"
        X = self.encoder.encode_batch(configs)
        return self.model_.predict(X).tolist()

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"model": self.model_, "encoder": self.encoder}, f)
        print(f"Saved GBMPredictor → {path}")

    @staticmethod
    def load(path: str) -> "GBMPredictor":
        with open(path, "rb") as f:
            data = pickle.load(f)
        pred         = GBMPredictor(data["encoder"])
        pred.model_  = data["model"]
        return pred


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _kendall_tau(preds, targets) -> float:
    from scipy.stats import kendalltau
    tau, _ = kendalltau(preds, targets)
    return float(tau)


def evaluate_predictor(predictor, configs: List[dict], accs: List[float]) -> dict:
    """Compute RMSE, Kendall-τ, and Spearman-ρ for a fitted predictor."""
    from scipy.stats import kendalltau, spearmanr
    preds = np.array(predictor.predict(configs))
    y     = np.array(accs)
    rmse  = float(np.sqrt(np.mean((preds - y) ** 2)))
    tau,  _ = kendalltau(preds, y)
    rho,  _ = spearmanr(preds, y)
    return {"rmse": rmse, "kendall_tau": float(tau), "spearman_rho": float(rho)}
