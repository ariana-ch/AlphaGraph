from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


@dataclass(frozen=True)
class Config:
    """Configuration for market data processing and dataset splits."""
    file_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    data_root: Path = field(init=False)
    N: int = 500

    start_date: date = date(2021, 1, 1)
    end_date: date = date(2025, 6, 30)
    lookback: int = 30
    prediction_horizon: int = 5
    batch_size: int = 8

    def __post_init__(self):
        object.__setattr__(self, 'data_root', self.file_path.parent / 'data/market_data')


def collate_fn(batch):
    """Collate function to stack time series examples for DataLoader."""
    batch_X = torch.stack([b['X'] for b in batch], dim=0)       # (B, L, N, D)
    batch_y = torch.stack([b['y'] for b in batch], dim=0)
    batch_dates = list(zip(*[[b['dates'][0], b['dates'][1:]] for b in batch]))

    return {
        'X': batch_X.squeeze(),
        'y': batch_y.squeeze(),
        'dates': batch_dates[1],
        'prediction_dates': batch_dates[0],
    }

def get_tickers(root_dir: Path = Config().data_root) -> np.ndarray:
    """Load the list of stock tickers."""
    try:
        tickers = np.load(root_dir / "tickers.npy", allow_pickle=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"Missing data file: {e.filename}")
    return tickers


class MarketDataset(Dataset):
    """PyTorch Dataset for market data time series."""
    def __init__(self,
                 root_dir: Path,
                 split: str,
                 lookback: int,
                 prediction_horizon: int):
        assert split in {'train', 'val', 'test'}
        self.root = Path(root_dir)
        self.split = split
        self.lookback = lookback
        self.pred_h = prediction_horizon

        try:
            self.X = np.load(self.root / f"X_{split}.npy")      # (T, N, D)
            self.y = np.load(self.root / f"Y_{split}.npy")      # (T, N)
            self.dates = [pd.to_datetime(dt).date() for dt in np.load(self.root / f"dates_{split}.npy")] # (T,)
        except FileNotFoundError as e:
            raise RuntimeError(f"Missing data file: {e.filename}")

        self.T, self.N, self.D = self.X.shape # T: time steps, N: stocks, D: features

    def __len__(self):
        if self.split == 'train':
            return self.T - self.lookback - self.pred_h + 1
        else:
            return (self.T - self.lookback) // self.pred_h

    def __getitem__(self, idx):
        if self.split == 'train':
            X_slice = self.X[idx: idx + self.lookback, :, :]
            y_slice = self.y[idx + self.lookback: idx + self.lookback + self.pred_h, :]
            dates_slice = self.dates[idx + self.lookback - 1: idx + self.lookback + self.pred_h]
        else:
            X_slice = self.X[idx * self.pred_h: idx * self.pred_h + self.lookback, :, :]
            y_slice = self.y[idx * self.pred_h + self.lookback: (idx + 1) * self.pred_h + self.lookback, :]
            dates_slice = self.dates[idx * self.pred_h + self.lookback - 1: (idx + 1) * self.pred_h + self.lookback]
        return {
            "X": torch.tensor(X_slice, dtype=torch.float32),
            "y": torch.tensor(y_slice, dtype=torch.float32),
            "dates": dates_slice,
        }

def get_dataloaders(
        lookback: int = 30,
        prediction_horizon: int = 5,
        root_dir: Path = Path(__file__).resolve().parent.parent / 'data/market_data',
        batch_size: int = 3,
        collate_fn: callable = collate_fn):
    train_dataset = MarketDataset(root_dir=root_dir, split='train', lookback=lookback,
                                  prediction_horizon=prediction_horizon)
    val_dataset = MarketDataset(root_dir=root_dir, split='val', lookback=lookback,
                                  prediction_horizon=prediction_horizon)
    test_dataset = MarketDataset(root_dir=root_dir, split='test', lookback=lookback,
                                 prediction_horizon=prediction_horizon)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4,
                              pin_memory=True, persistent_workers=True, prefetch_factor=8)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, collate_fn=collate_fn,
                            num_workers=2, pin_memory=True, persistent_workers=False, prefetch_factor=4)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, collate_fn=collate_fn,
                             num_workers=2, pin_memory=True, persistent_workers=False, prefetch_factor=4)
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    import time
    train_loader, val_loader, test_loader  = get_dataloaders(batch_size=16)
    # Test how long it takes to move data to GPU

    for i, batch in enumerate(test_loader):
        for j in batch['dates']:
            if len(j) != 5:
                print(f"Batch dates: {j}")
    print('Done')
