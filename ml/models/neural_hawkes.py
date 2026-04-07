"""
EDGE-003: Neural Hawkes Process for Order Flow Prediction
==========================================================

Models the arrival intensity of buy/sell market events as a self-exciting
point process, where each event increases the probability of future events
(microstructure clustering).

A neural network parameterizes the conditional intensity function:
    lambda(t) = softplus( mu + sum_j alpha_j * kernel(t - t_j) )

where the kernel and base intensity are learned end-to-end.

Outputs:
  - P(next event = buy | history)  vs  P(sell)
  - Expected time until next event (1 / integrated intensity)

Requires PyTorch.  Falls back to a Poisson baseline if unavailable.

Conforms to the AlphaModel interface:
    fit(X, y)     -- train on event sequences
    predict(X)    -- predict buy probability for each sample
    score(X, y)   -- negative log-likelihood
"""

import logging
import math
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    logger.warning(
        "EDGE-003: PyTorch not available. NeuralHawkesPredictor will use "
        "a Poisson baseline. Install with: pip install torch"
    )


# ===================================================================
# Neural Hawkes components (PyTorch)
# ===================================================================

if _TORCH_AVAILABLE:

    class ContinuousTimeLSTMCell(nn.Module):
        """CT-LSTM cell that decays hidden state between events.

        Between events, the cell state decays exponentially toward a
        steady-state value, modulated by a learned decay rate.
        """

        def __init__(self, input_dim: int, hidden_dim: int):
            super().__init__()
            self.hidden_dim = hidden_dim
            # Gates: input, forget, output, cell-bar, decay
            self.linear = nn.Linear(input_dim + hidden_dim, hidden_dim * 5, bias=True)

        def forward(self, x: torch.Tensor, h: torch.Tensor,
                    c: torch.Tensor, c_bar: torch.Tensor,
                    dt: torch.Tensor) -> Tuple[torch.Tensor, ...]:
            """Process one event.

            Args:
                x:     (batch, input_dim)  -- event embedding
                h:     (batch, hidden)     -- hidden state before decay
                c:     (batch, hidden)     -- cell state before decay
                c_bar: (batch, hidden)     -- cell steady state
                dt:    (batch, 1)          -- time since last event

            Returns:
                h_new, c_new, c_bar_new, decay
            """
            # Decay cell state toward steady state
            decay_gate = torch.sigmoid(self.linear(torch.cat([x, h], dim=-1))[:, -self.hidden_dim:])
            c_decayed = c_bar + (c - c_bar) * torch.exp(-decay_gate * dt)

            combined = torch.cat([x, h], dim=-1)
            gates = self.linear(combined)

            i_gate = torch.sigmoid(gates[:, :self.hidden_dim])
            f_gate = torch.sigmoid(gates[:, self.hidden_dim:2*self.hidden_dim])
            o_gate = torch.sigmoid(gates[:, 2*self.hidden_dim:3*self.hidden_dim])
            z_gate = torch.tanh(gates[:, 3*self.hidden_dim:4*self.hidden_dim])

            c_new = f_gate * c_decayed + i_gate * z_gate
            c_bar_new = f_gate * c_bar + i_gate * z_gate
            h_new = o_gate * torch.tanh(c_new)

            return h_new, c_new, c_bar_new, decay_gate


    class NeuralHawkesNet(nn.Module):
        """Full neural Hawkes process network.

        Processes a sequence of (event_type, inter_event_time) pairs
        and outputs buy/sell intensity at each step.
        """

        def __init__(self, n_event_types: int = 2, n_features: int = 0,
                     d_model: int = 64, n_layers: int = 2):
            super().__init__()
            self.n_event_types = n_event_types
            self.d_model = d_model

            # Event type embedding
            self.type_emb = nn.Embedding(n_event_types + 1, d_model)  # +1 for padding

            # Optional feature projection
            self.feat_proj = nn.Linear(n_features, d_model) if n_features > 0 else None
            input_dim = d_model * (2 if n_features > 0 else 1)

            # CT-LSTM layers
            self.cells = nn.ModuleList([
                ContinuousTimeLSTMCell(
                    input_dim if i == 0 else d_model,
                    d_model
                )
                for i in range(n_layers)
            ])

            # Intensity heads (one per event type)
            self.intensity_head = nn.Linear(d_model, n_event_types)
            # Time prediction head
            self.time_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1),
                nn.Softplus(),
            )

        def forward(self, event_types: torch.Tensor, inter_times: torch.Tensor,
                    features: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                event_types: (batch, seq_len) LongTensor of event types (0=buy, 1=sell)
                inter_times: (batch, seq_len) float — time since previous event
                features:    (batch, seq_len, n_features) optional extra features

            Returns:
                intensities: (batch, seq_len, n_event_types)
                time_preds:  (batch, seq_len, 1)
            """
            B, L = event_types.shape

            # Embed event types
            x = self.type_emb(event_types)  # (B, L, d_model)
            if self.feat_proj is not None and features is not None:
                feat = self.feat_proj(features)
                x = torch.cat([x, feat], dim=-1)

            dt = inter_times.unsqueeze(-1)  # (B, L, 1)

            # Initialize states
            states = []
            for _cell in self.cells:
                h = torch.zeros(B, self.d_model, device=x.device)
                c = torch.zeros(B, self.d_model, device=x.device)
                c_bar = torch.zeros(B, self.d_model, device=x.device)
                states.append((h, c, c_bar))

            all_h = []
            for t in range(L):
                inp = x[:, t, :]
                for i, cell in enumerate(self.cells):
                    h, c, c_bar = states[i]
                    h, c, c_bar, _ = cell(inp, h, c, c_bar, dt[:, t, :])
                    states[i] = (h, c, c_bar)
                    inp = h
                all_h.append(h)

            hidden = torch.stack(all_h, dim=1)  # (B, L, d_model)

            intensities = F.softplus(self.intensity_head(hidden))  # (B, L, n_types)
            time_preds = self.time_head(hidden)  # (B, L, 1)

            return intensities, time_preds


# ===================================================================
# Poisson Baseline (no-torch fallback)
# ===================================================================

class PoissonBaseline:
    """Simple Poisson process baseline estimating constant buy/sell rates."""

    def __init__(self):
        self.buy_rate = 0.5
        self.sell_rate = 0.5
        self.mean_iat = 1.0  # mean inter-arrival time

    def fit(self, event_types: np.ndarray, inter_times: np.ndarray):
        event_types = np.asarray(event_types).ravel()
        inter_times = np.asarray(inter_times).ravel()
        n_buy = np.sum(event_types == 0)
        n_total = max(len(event_types), 1)
        self.buy_rate = n_buy / n_total
        self.sell_rate = 1.0 - self.buy_rate
        self.mean_iat = float(np.mean(inter_times)) if len(inter_times) > 0 else 1.0

    def predict_buy_prob(self, n: int) -> np.ndarray:
        return np.full(n, self.buy_rate, dtype=np.float64)

    def predict_time(self, n: int) -> np.ndarray:
        return np.full(n, self.mean_iat, dtype=np.float64)


# ===================================================================
# Public API: NeuralHawkesPredictor (AlphaModel interface)
# ===================================================================

class NeuralHawkesPredictor:
    """Neural Hawkes Process predictor for order flow modeling.

    Input X can be:
      - 2-D array (n_samples, n_cols) where col 0 = event_type (0/1),
        col 1 = inter_event_time, remaining cols = features.
      - Dict with keys 'event_types', 'inter_times', and optional 'features'.

    Parameters
    ----------
    d_model : int
        Hidden dimension.
    n_layers : int
        Number of CT-LSTM layers.
    n_event_types : int
        Number of distinct event types (default 2: buy/sell).
    lr : float
        Learning rate.
    epochs : int
        Training epochs.
    batch_size : int
        Mini-batch size.
    seq_len : int
        Sequence length for training windows.
    device : str
        'cpu' or 'cuda'.
    """

    def __init__(self, *, d_model: int = 64, n_layers: int = 2,
                 n_event_types: int = 2, n_features: int = 0,
                 lr: float = 1e-3, epochs: int = 50, batch_size: int = 32,
                 seq_len: int = 50, device: str = "cpu", **kwargs):
        self.params = {
            "d_model": d_model, "n_layers": n_layers,
            "n_event_types": n_event_types, "n_features": n_features,
            "lr": lr, "epochs": epochs, "batch_size": batch_size,
            "seq_len": seq_len, "device": device,
        }
        self._model = None
        self._baseline = PoissonBaseline()
        self._fitted = False

    def _parse_input(self, X):
        """Convert input to (event_types, inter_times, features)."""
        if isinstance(X, dict):
            et = np.asarray(X["event_types"], dtype=np.int64)
            it = np.asarray(X["inter_times"], dtype=np.float64)
            feat = np.asarray(X.get("features"), dtype=np.float64) if "features" in X else None
            return et, it, feat

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 2)
        et = X[:, 0].astype(np.int64)
        it = X[:, 1]
        feat = X[:, 2:] if X.shape[1] > 2 else None
        return et, it, feat

    def _make_windows(self, et, it, feat, seq_len):
        """Slice long sequences into fixed-length windows."""
        n = len(et)
        windows_et, windows_it, windows_feat = [], [], []
        for start in range(0, n - seq_len + 1, seq_len // 2):
            end = start + seq_len
            windows_et.append(et[start:end])
            windows_it.append(it[start:end])
            if feat is not None:
                windows_feat.append(feat[start:end])
        if not windows_et:
            # Pad if sequence too short
            pad_len = seq_len - n
            windows_et.append(np.pad(et, (0, pad_len), constant_values=0))
            windows_it.append(np.pad(it, (0, pad_len), constant_values=0.0))
            if feat is not None:
                windows_feat.append(np.pad(feat, ((0, pad_len), (0, 0))))

        et_arr = np.stack(windows_et)
        it_arr = np.stack(windows_it)
        feat_arr = np.stack(windows_feat) if windows_feat else None
        return et_arr, it_arr, feat_arr

    # ------------------------------------------------------------------
    # AlphaModel interface
    # ------------------------------------------------------------------

    def fit(self, X, y=None) -> "NeuralHawkesPredictor":
        """Train on event sequence data.

        Args:
            X: event data (see _parse_input for formats)
            y: ignored (labels derived from event_types)
        """
        et, it, feat = self._parse_input(X)
        self._baseline.fit(et, it)

        if not _TORCH_AVAILABLE:
            logger.info("EDGE-003: Fitted Poisson baseline (no PyTorch).")
            self._fitted = True
            return self

        p = self.params
        n_feat = feat.shape[-1] if feat is not None else 0

        # Build windows
        et_w, it_w, feat_w = self._make_windows(et, it, feat, p["seq_len"])

        self._model = NeuralHawkesNet(
            n_event_types=p["n_event_types"], n_features=n_feat,
            d_model=p["d_model"], n_layers=p["n_layers"],
        ).to(p["device"])

        optimizer = torch.optim.Adam(self._model.parameters(), lr=p["lr"])
        n_windows = et_w.shape[0]

        self._model.train()
        for epoch in range(p["epochs"]):
            perm = np.random.permutation(n_windows)
            epoch_loss = 0.0

            for start in range(0, n_windows, p["batch_size"]):
                idx = perm[start:start + p["batch_size"]]
                et_b = torch.tensor(et_w[idx], dtype=torch.long, device=p["device"])
                it_b = torch.tensor(it_w[idx], dtype=torch.float32, device=p["device"])
                feat_b = None
                if feat_w is not None:
                    feat_b = torch.tensor(feat_w[idx], dtype=torch.float32, device=p["device"])

                intensities, time_preds = self._model(et_b, it_b, feat_b)

                # Negative log-likelihood loss for point process
                # For each step, the correct event type should have high intensity
                # while total intensity contributes to the compensator
                target_intensity = intensities.gather(
                    2, et_b.unsqueeze(-1)
                ).squeeze(-1)  # (B, L)
                total_intensity = intensities.sum(dim=-1)  # (B, L)

                # NLL = -log(lambda_k(t)) + integral(lambda(s)ds)
                # Approximate integral as total_intensity * inter_time
                log_lambda = torch.log(target_intensity + 1e-8)
                compensator = total_intensity * it_b
                nll = (-log_lambda + compensator).mean()

                # Time prediction MSE
                time_loss = F.mse_loss(time_preds.squeeze(-1), it_b)

                loss = nll + 0.1 * time_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 5.0)
                optimizer.step()
                epoch_loss += loss.item() * len(idx)

            if (epoch + 1) % 10 == 0:
                avg = epoch_loss / n_windows
                logger.debug("EDGE-003 epoch %d/%d  loss=%.4f", epoch + 1, p["epochs"], avg)

        self._fitted = True
        logger.info("EDGE-003: Neural Hawkes trained on %d events (%d windows).",
                     len(et), n_windows)
        return self

    def predict(self, X) -> np.ndarray:
        """Predict P(next event = buy) for each position.

        Returns:
            buy_probabilities: (n_samples,) array in [0, 1]
        """
        et, it, feat = self._parse_input(X)

        if not _TORCH_AVAILABLE or self._model is None:
            return self._baseline.predict_buy_prob(len(et))

        p = self.params
        self._model.eval()

        # Process as a single batch
        et_t = torch.tensor(et, dtype=torch.long, device=p["device"]).unsqueeze(0)
        it_t = torch.tensor(it, dtype=torch.float32, device=p["device"]).unsqueeze(0)
        feat_t = None
        if feat is not None:
            feat_t = torch.tensor(feat, dtype=torch.float32, device=p["device"]).unsqueeze(0)

        with torch.no_grad():
            intensities, _ = self._model(et_t, it_t, feat_t)
            # P(buy) = lambda_buy / (lambda_buy + lambda_sell)
            probs = intensities[0, :, 0] / (intensities[0].sum(dim=-1) + 1e-8)

        return probs.cpu().numpy().astype(np.float64)

    def predict_time(self, X) -> np.ndarray:
        """Predict expected time to next event for each position."""
        et, it, feat = self._parse_input(X)

        if not _TORCH_AVAILABLE or self._model is None:
            return self._baseline.predict_time(len(et))

        p = self.params
        self._model.eval()

        et_t = torch.tensor(et, dtype=torch.long, device=p["device"]).unsqueeze(0)
        it_t = torch.tensor(it, dtype=torch.float32, device=p["device"]).unsqueeze(0)
        feat_t = None
        if feat is not None:
            feat_t = torch.tensor(feat, dtype=torch.float32, device=p["device"]).unsqueeze(0)

        with torch.no_grad():
            _, time_preds = self._model(et_t, it_t, feat_t)

        return time_preds[0, :, 0].cpu().numpy().astype(np.float64)

    def score(self, X, y=None) -> float:
        """Return negative mean log-likelihood (lower = better fit)."""
        et, it, feat = self._parse_input(X)
        buy_probs = self.predict(X)
        # Log-likelihood: sum log P(observed event type)
        is_buy = (et == 0).astype(np.float64)
        log_lik = is_buy * np.log(buy_probs + 1e-8) + (1 - is_buy) * np.log(1 - buy_probs + 1e-8)
        return float(np.mean(log_lik))

    def get_params(self) -> Dict[str, Any]:
        return dict(self.params)

    def __repr__(self) -> str:
        backend = "neural" if (_TORCH_AVAILABLE and self._model is not None) else "poisson"
        status = "fitted" if self._fitted else "unfitted"
        return f"NeuralHawkesPredictor(backend={backend}, {status})"
