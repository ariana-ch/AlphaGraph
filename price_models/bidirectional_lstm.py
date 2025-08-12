from typing import List, Optional
import torch
import torch.nn as nn
from pathlib import Path
import json


def build_mlp(dims: List[int],
              activation: nn.Module,
              dropout_p: float = 0.0) -> nn.Sequential:
    """
    dims: [in, h1, h2, ..., out]
    Applies (Linear -> Activation -> Dropout) for all but the last Linear.
    The last Linear has no activation/dropout (add later if desired).
    """
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2:
            layers.append(activation)
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
    return nn.Sequential(*layers)

class BidirectionalLSTM(nn.Module):
    def __init__(self,
                 input_size: int = 40,
                 stem_dims: List[int] = [64, 128, 16],
                 lstm_hidden_dim: int = 128,
                 lstm_layers: int = 2,
                 lstm_dropout: float = 0.1,  # inter-layer LSTM dropout (only if layers>1)
                 head_dims: List[int] = [64, 1],
                 stem_dropout: float = 0.1,
                 head_dropout: float = 0.1,
                 activation_name: str = "gelu"):
        super().__init__()

        # Choose activation (GELU/SILU generally perform well on finance series)
        act = {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}[activation_name]

        # ----- Stem MLP (time-distributed) -----
        stem_in = input_size
        self.use_stem = len(stem_dims) > 0
        if self.use_stem:
            stem_full_dims = [stem_in] + stem_dims
            self.stem = build_mlp(stem_full_dims, activation=act, dropout_p=stem_dropout)
            lstm_in = stem_dims[-1]
        else:
            self.stem = nn.Identity()
            lstm_in = input_size

        # ----- BiLSTM -----
        self.bilstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=(lstm_dropout if lstm_layers > 1 else 0.0),
            bidirectional=True,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(lstm_hidden_dim * 2)
        # Optional dropout right after LSTM outputs (variational-style on features)
        self.post_lstm_dropout = nn.Dropout(lstm_dropout if lstm_dropout > 0 else 0.0)

        # ----- Head MLP -----
        head_in = 2 * lstm_hidden_dim  # bidirectional concatenation
        head_full_dims = [head_in] + head_dims
        self.head = build_mlp(head_full_dims, activation=act, dropout_p=head_dropout)

        # If you want activation on the very last layer (e.g., classification),
        # add it explicitly outside or here based on your task.

    def forward(self,
                x: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: shape (B, T, input_size)
        lengths: optional 1D tensor of valid lengths for packing (CPU or same device as x)
        Returns: shape (B, T, head_out) using per-step outputs (uses last Linear of head without act)
        """
        B, T, F = x.shape

        # Time-distributed stem: flatten time then restore
        if self.use_stem:
            x = x.reshape(B * T, F)
            x = self.stem(x)
            x = x.reshape(B, T, -1)

        # Pack for variable-length sequences (if provided)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.bilstm(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=T)
        else:
            h, _ = self.bilstm(x)
        h = h[:, -1, :] # Get the last time step output (B, 2*hidden)
        h = self.post_lstm_dropout(h)
        y = self.head(h)
        return y


def load(model_name: str = 'best_model_20250811_122543_bidirectional_lstm_final'):
    model_dir = Path(__file__).parent / 'models' / 'bidirectional_lstm' / f"{model_name}.pt"
    config_dir = (Path(__file__).parent / 'logs' / 'bidirectional_lstm'/ model_name.replace('best_model_', '') / "config.json")
    config = {'input_size': 45,
              'stem_dims': [16],
              'stem_dropout': 0.2,
              'lstm_hidden_dim': 64,
              'lstm_layers': 2,
              'lstm_dropout': 0.1,
              'head_dims': [32, 1],
              'head_dropout': 0.1,
              'activation_name': 'gelu', }
    for k, v in json.loads(config_dir.read_text()).items():
        if k in config:
            config[k] = v

    state = torch.load(model_dir, map_location='cpu')
    model = BidirectionalLSTM(**config)
    model.load_state_dict(state)

    model.eval()
    return model

# ---- Example usage ----
if __name__ == "__main__":
    model = load()
    from dataloader import MarketDataset
    from entmax import entmax15
    from collections import defaultdict
    test_data = MarketDataset(root_dir=Path(__file__).resolve().parent.parent / 'data' / 'market_data', lookback=30,
                              prediction_horizon=5, split='test')
    d = defaultdict(list)
    w = []
    for i in range(len(test_data)):
        X, returns, dates = test_data.__getitem__(i).values()
        prediction_date = dates[0]
        X = X.permute(1, 0, 2)
        weights = entmax15(model(X).squeeze())
        stocks = torch.where(weights != 0)[0]
        w.append(weights)
        weights = weights[weights != 0]
        d['prediction_date'].append(prediction_date)
        d['holding_period'] = dates[1:]
        d['weights'].append(weights.tolist())
        d['stocks'].append(stocks.tolist())

    a = model(test_data.__getitem__(0))
    print(a)

