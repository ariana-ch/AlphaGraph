from pathlib import Path
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from logging import getLogger
from collections import Counter
from typing import Union

logger = getLogger(__name__)


def set_device(use_cpu: bool = False):
    """Set the device for training based on available hardware."""
    if use_cpu:
        print("Using CPU backend")
        return torch.device('cpu')
    if torch.backends.mps.is_available():
        print("Using MPS backend")
        torch.backends.mps.benchmark = True
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        return torch.device('mps')
    elif torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        print("Using CUDA backend")
        return torch.device('cuda')
    else:
        print("Using CPU backend")
        return torch.device('cpu')

#
# def set_device(graph_model: bool = False):
#     if torch.cuda.is_available():
#         torch.backends.cudnn.benchmark = True
#
#         if graph_model:
#             # see if there are more than 1 devices available
#             if torch.cuda.device_count() > 1:
#                 print("Using CUDA for graph model")
#                 device = torch.device('cuda:1')
#             else:
#                 device = torch.device('cpu')
#         else:
#             device = torch.device('cuda:0')
#             print("Using CUDA")
#     elif torch.backends.mps.is_available() and not graph_model:
#         torch.backends.mps.benchmark = True
#         device = torch.device("mps")
#         os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
#         os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
#         print("Using MPS")
#     else:
#         device = torch.device("cpu")
#         print("Using CPU")
#     return device

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_scheduler(optimizer, scheduler_type='plateau'):
    """Get learning rate scheduler."""
    if scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    else:
        return None


def compute_metrics(weights, raw_returns, losses=None, batch_idx=None):
    d = dict()
    weights = torch.cat(weights)
    raw_returns = torch.cat(raw_returns)
    periods, holding_window, _ = raw_returns.shape  # Number of holding periods
    annualization_factor =  torch.tensor(252.0 / holding_window, device=raw_returns.device, dtype=raw_returns.dtype)

    portfolio_returns = ((raw_returns * weights.unsqueeze(dim=1)).sum(dim=2) + 1).prod(dim=1) - 1  # per holding period

    mean = portfolio_returns.mean()
    std = portfolio_returns.std(unbiased=False) + 1e-8  # Avoid division by zero

    # Annualized Sharpe
    annualised_sharpe = mean / std * torch.sqrt(annualization_factor)

    # Compound returns
    geometric_return = (1 + portfolio_returns).prod() ** annualization_factor - 1

    components = (weights != 0).type(torch.float16).sum(dim=1).mean()

    min_weight = weights[weights > 0.0].min()

    # max drawdown calculation
    cumulative_returns = torch.cumprod(1 + portfolio_returns, dim=0)
    if cumulative_returns.device.type == 'mps':
        peak = torch.cummax(cumulative_returns.detach().cpu(), dim=0).values
        peak = peak.to(cumulative_returns.device)
    else:
        peak = torch.cummax(cumulative_returns, dim=0).values
    drawdown = (cumulative_returns - peak)/peak
    max_drawdown = drawdown.min()

    # calmar ratio
    calmar_ratio = geometric_return / (max_drawdown + 1e-8)

    d['Annualised Sharpe'] = annualised_sharpe.detach().cpu().item()
    d['Average Annualised Return'] = (mean * annualization_factor).detach().cpu().item()
    d['Average Components'] = components.detach().cpu().item()
    d['Min Weight'] = min_weight.detach().cpu().item()
    d['Max Drawdown'] = max_drawdown.detach().cpu().item()
    d['Calmar Ratio'] = calmar_ratio.detach().cpu().item()

    loss_names = {'utility_target_penalty': 'Loss: Utility Target',
                  'cardinality_penalty': 'Loss: Components',
                  'vol_target_penalty': 'Loss: Vol Target',
                  'min_weight_penalty': 'Loss: Min Weight'}
    for loss_name, loss_value in losses.items():
        d[loss_names.get(loss_name, loss_name)] = (loss_value / (batch_idx + 1))
    return d


class EarlyStopping:
    """Early stopping handler to prevent overfitting."""

    def __init__(self, patience=7, min_delta=0.0, mode='min', warmup_epochs=5):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.best_epoch = 0
        self.warmup_epochs = warmup_epochs  # Number of warm-up epochs to skip early stopping

    def __call__(self, epoch, value, model, path):
        if epoch < self.warmup_epochs:
            return False  # Skip early stopping on the first epoch
        if self.best_value is None:
            self.best_value = value
            self.best_epoch = epoch
            old_value = self.best_value
            self.save_checkpoint(old_value, value, model, path)
        elif self.mode == 'min' and value > self.best_value - self.min_delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        elif self.mode == 'max' and value < self.best_value + self.min_delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            old_value = self.best_value
            self.best_value = value
            self.best_epoch = epoch
            self.save_checkpoint(old_value, value, model, path)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, old_value, value, model, path):
        logger.info(f'Validation value improved ({old_value:.6f} --> {value:.6f}). Saving model ...')
        print(f'[Epoch {self.best_epoch}] Validation value improved ({old_value:.6f} --> {value:.6f}). Saving model ...')
        if isinstance(model, dict):
            torch.save(model, path)
        else:
            torch.save(model.state_dict(), path)


class PortfolioTensorBoardLogger:

    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = Path(log_dir)
        self.epoch_data = {'train': [], 'val': [], 'test': []}

    def log_metrics(self, epoch, batch_idx, weights, raw_returns, losses, epoch_length=None, phase='train', is_epoch=False, **kwargs):
        """Log simple batch-level averages (for training monitoring only)."""
        # Convert to numpy
        d = compute_metrics(weights=weights, raw_returns=raw_returns, losses=losses, batch_idx=batch_idx)

        if is_epoch:
            self.epoch_data[phase].append(d)
            phase = f'Epoch_{phase.title()}'
        else:
            phase = phase.title()
        for k, v in d.items():
            self.writer.add_scalar(f'{phase}:{k}', v, epoch * epoch_length + batch_idx)
        return d['Annualised Sharpe']

    def export_epoch_summary(self):
        """Export epoch summaries to CSV for external analysis."""
        export_dir = self.log_dir / 'summaries'
        export_dir.mkdir(parents=True, exist_ok=True)

        for phase in ['train', 'val', 'test']:
            if self.epoch_data[phase]:
                df = pd.DataFrame(self.epoch_data[phase])
                df.to_csv(export_dir / f'{phase}_summary.csv', index=False)

    def get_xticks(self, dates, freq: str = 'M'):
        '''
        Get x-ticks for plotting based on the dates. freq can be 'M' or 'Q' for monthly or quarterly.
        '''
        df = pd.DataFrame({'date': dates})
        if freq == 'M':
            df['group'] = df['date'].dt.to_period('M')
            label_fmt = lambda d: d.strftime('%b-%Y')
        elif freq == 'Q':
            df['group'] = df['date'].dt.to_period('Q')
            label_fmt = lambda d: f"Q{d.quarter}-{d.year}"
        else:
            raise ValueError("freq must be 'M' or 'Q'")

        first = df.groupby('group').head(1)
        return first.index.tolist(), list(map(label_fmt, first['date']))

    def add_plot(self, epoch: int, metrics: dict, tickers: Union[list, np.ndarray], phase='val'):
        """Log detailed time series plots (optional, for deep analysis)."""
        # This is optional and only called when you want detailed time series analysis
        raw_returns = torch.cat(metrics['raw_returns']).detach().cpu().numpy()
        weights = torch.cat(metrics['weights']).unsqueeze(dim=1).detach().cpu().numpy()

        batches, T, N = raw_returns.shape
        dates = pd.to_datetime(np.array(metrics['dates']).flatten())

        if phase == 'train':
            dates = dates[::5]
            raw_returns = raw_returns[::5, :, :]
            weights = weights[::5, :, :]
            freq = 'Q'
        else:
            freq = 'Q'
        dates = dates[::5]
        periods, holding_window, _ = raw_returns.shape
        annualization_factor = 252.0 / holding_window
        portfolio_returns = ((raw_returns * weights).sum(axis=2) + 1).prod(axis=1) - 1  # per holding period

        weights = weights.squeeze(axis=1)

        # Compute time series metrics
        value_curve = np.cumprod(1 + portfolio_returns)

        # sharpe - expanding
        sharpe = []
        for i in range(15, len(portfolio_returns), 1):
            sharpe.append(portfolio_returns[:i].mean()/portfolio_returns[:i].std() * np.sqrt(annualization_factor))
        x_axis_sharpe = self.get_xticks(dates[15:], freq=freq)
        x_axis = self.get_xticks(dates, freq=freq)

        # Create and save plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'{phase.title()} Epoch {epoch} - Detailed Analysis')

        # Value curve
        axes[0, 0].plot(value_curve)
        axes[0, 0].set_title('Portfolio Value')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].set_xticks(ticks=x_axis[0], labels=x_axis[1], rotation=90)
        axes[0, 0].grid(axis="y", linestyle="--", linewidth=0.5)
        # Daily returns
        axes[0, 1].plot(sharpe)
        axes[0, 1].set_title('Rolling Annualised Sharpe Ratio')
        axes[0, 1].set_ylabel('Annualised Sharpe Ratio')
        axes[0, 1].set_xticks(ticks=x_axis_sharpe[0], labels=x_axis_sharpe[1], rotation=90)
        axes[0, 1].grid(axis="y", linestyle="--", linewidth=0.5)

        # Weight evolution (if available)
        topassets = Counter(np.argsort(weights, axis=1)[:, -10:].flatten())
        top10assets = [[i,j] for i, j in topassets.items() if j >= topassets.most_common(10)[-1][-1]]
        assets, freqs = zip(*top10assets)
        assets = np.array(tickers)[list(assets)]
        cmap = plt.get_cmap('tab20')  # or 'tab20', 'Set3', etc.
        colors = [cmap(i) for i in range(len(assets))]

        axes[1, 0].bar(assets, np.array(freqs)/len(weights), alpha=0.7, color=colors, edgecolor='black', linewidth=0.8,
                       width=0.5)
        axes[1, 0].set_title('Top 10 Assets')
        axes[1, 0].set_ylabel('Frequency')
        for label in axes[1, 0].get_xticklabels():
            label.set_rotation(45)

        # Number of positions over time
        num_positions = (weights > 0).sum(axis=1)
        axes[1, 1].plot(num_positions)
        axes[1, 1].axhline(y=num_positions.mean(), color='r', linestyle='--', label='Average', linewidth=1)
        axes[1, 1].set_title('Number of Active Positions')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_xticks(ticks=x_axis[0], labels=x_axis[1], rotation=90)

        plt.tight_layout()
        if phase != 'train':
            # Save plot
            plot_dir = self.log_dir / 'plots' / phase
            plot_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_dir / f'epoch_{epoch:03d}.png', dpi=100, bbox_inches='tight')

        # Add to TensorBoard
        self.writer.add_figure(f'{phase}-Plots', fig, epoch)
        plt.close(fig)
