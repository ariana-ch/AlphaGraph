import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from pprint import pp
import torch
import torch.optim as optim
from entmax import entmax_bisect, entmax15
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", message=".*pin_memory.*not supported on MPS.*")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

# Add parent directory to Python path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from dataloader import get_dataloaders, get_tickers
from losses import LongOnlyMarkowitzPortfolioLoss
from patch_tst import PatchTSTModel
from bidirectional_lstm import BidirectionalLSTM

from utils import EarlyStopping, PortfolioTensorBoardLogger, get_scheduler, set_seed

def set_device():
    """Set the device for training based on available hardware."""
    if torch.backends.mps.is_available():
        print("Using MPS backend")
        return torch.device('mps')
    elif torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("Using CUDA backend")
        return torch.device('cuda')
    else:
        print("Using CPU backend")
        return torch.device('cpu')


def create_run_name(args):
    """Create a descriptive run name with key hyperparameters."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Format learning rate
    lr_str = f"lr{args.lr:.0e}".replace('-0', '-').replace('+0', '+')

    # Create parameter string
    param_str = f"bs{args.batch_size}_ep{args.epochs}_{lr_str}"

    # Add other distinguishing parameters
    if hasattr(args, 'patience') and args.patience != 15:
        param_str += f"_pat{args.patience}"
    if hasattr(args, 'seed') and args.seed != 42:
        param_str += f"_seed{args.seed}"

    return f"{timestamp}_{param_str}"


def setup_directories(model_name, run_name):
    """Setup log and model directories with descriptive names."""
    # Create directories
    base_dir = Path(__file__).parent
    log_dir = base_dir / 'logs' / model_name / run_name
    model_dir = base_dir / 'models' / model_name

    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f'best_model_{run_name}.pt'

    return log_dir, model_path


def run_cuda(model, train_loader, val_loader, test_loader, optimizer, scheduler, loss_fn, device, logger,
             epochs, model_path, early_stopping, loss_coefficients, tickers, scaler, **kwargs):
    """Main training loop."""

    best_val_sharpe = float('-inf')

    activation = entmax15
    for epoch in range(1, epochs + 1):
        print(f"\n{'=' * 50}\nEpoch {epoch}/{epochs}\n{'=' * 50}")

        # ============== Training ==============
        model.train()
        running_metrics = defaultdict(list)
        running_losses = defaultdict(float)

        all_metrics = defaultdict(list)
        all_losses = defaultdict(float)
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")

        epoch_length = len(train_loader)
        for batch_idx, batch in enumerate(pbar):
            X, y, dates, prediction_dates = batch['X'], batch['y'], batch['dates'], batch['prediction_dates']
            B, seq_len, stocks, features = X.shape
            X, returns = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            batch_loss = torch.tensor(0.0, dtype=X.dtype).to(device)

            # Prepare data for model (reshape as in original)
            x = X.permute(0, 2, 1, 3).contiguous()  # [B, seq_len, stocks, features] -> [B, stocks, seq_len, features]
            x = x.reshape(-1, x.shape[-2], x.shape[-1])  # [B*stocks, seq_len, features]

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type):
                # Forward pass
                weights = model(x)  # [B*stocks, 1] or similar
                weights = weights.reshape(B, stocks)  # [B, stocks]
                weights = activation(weights, dim=-1)
                # clipped_weights = weights.clamp(min=loss_fn.target_min_weight, max=None)
                # Compute loss
                loss_dict = loss_fn(weights, returns)
                for loss_name, coefficient in loss_coefficients.items():
                    batch_loss += (loss_dict[loss_name] * coefficient).mean()
                    running_losses[loss_name] += (loss_dict[loss_name] * coefficient).mean()
                    all_losses[loss_name] += (loss_dict[loss_name] * coefficient).mean()
            if scaler is not None:
                scaler.scale(batch_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_loss.backward()
                optimizer.step()
            running_metrics['weights'].append(weights)
            running_metrics['raw_returns'].append(returns)
            all_metrics['weights'].append(weights)
            all_metrics['raw_returns'].append(returns)
            all_metrics['dates'].extend(dates)
            ann_factor = torch.sqrt(torch.tensor(252.0 / returns.shape[1], device=X.device, dtype=X.dtype))

            if (batch_idx + 1) % 5 == 0:
                # Log batch metrics (simple averages for monitoring)
                logger.log_metrics(epoch=epoch, batch_idx=batch_idx, is_epoch=False, epoch_length=epoch_length,
                                   phase='train', losses=running_losses, **running_metrics, )
                running_metrics = defaultdict(list)
                running_losses = defaultdict(float)
            # Update progress bar

            portfolio_returns = ((returns * weights.unsqueeze(dim=1)).sum(dim=2) + 1).prod(dim=1) - 1  # per holding period

            mean = portfolio_returns.mean()
            std = portfolio_returns.std(unbiased=False) + 1e-8  # Avoid division by zero

            # Annualized Sharpe
            sharpe = mean / std * torch.sqrt(ann_factor)
            pbar.set_postfix({'loss': f'{batch_loss.item():.4f}', 'sharpe': f'{sharpe.item():.2f}'})

        train_sharpe = logger.log_metrics(epoch=epoch, batch_idx=0, phase='train', epoch_length=epoch_length,
                                          is_epoch=True, losses=all_losses, **all_metrics)
        avg_train_loss = sum(list(all_losses.values())) / (batch_idx + 1)
        logger.add_plot(metrics=all_metrics, epoch=epoch, phase='train', tickers=tickers)

        # ============== Evaluation on Validation Data ==============
        model.eval()
        metrics = defaultdict(list)
        losses = defaultdict(float)
        epoch_length = len(val_loader)

        pbar = tqdm(val_loader, desc=f"Evaluating Validation Set")
        with torch.inference_mode():
            for batch_idx, batch in enumerate(pbar):  # I think I am using one batch - tbd
                batch_loss = torch.tensor(0.0, dtype=X.dtype).to(device)

                X, y, dates, prediction_dates = batch['X'], batch['y'], batch['dates'], batch['prediction_dates']
                B, seq_len, stocks, features = X.shape
                X, y = X.to(device), y.to(device)
                x = X.permute(0, 2, 1, 3)  # [B, seq_len, stocks, features] -> [B, stocks, seq_len, features]
                x = x.reshape(-1, x.shape[-2], x.shape[-1])  # [B*stocks, seq_len, features]
                returns = y  # [B, pred_len, stocks]
                with torch.autocast(device.type, enabled=device.type == "cuda"):
                    weights = model(x)
                    weights = weights.reshape(B, stocks)  # [B, stocks]
                    # weights = F.softmax(weights, dim=-1)
                    weights = activation(weights, dim=-1)
                    loss_dict = loss_fn(weights, returns)
                    for loss_name, coefficient in loss_coefficients.items():
                        losses[loss_name] += (loss_dict[loss_name] * coefficient).mean().item()
                        batch_loss += (loss_dict[loss_name] * coefficient).mean()
                metrics['raw_returns'].append(returns)
                metrics['weights'].append(weights)
                metrics['dates'].extend(dates)

                portfolio_returns = ((returns * weights.unsqueeze(dim=1)).sum(dim=2) + 1).prod(
                    dim=1) - 1  # per holding period

                mean = portfolio_returns.mean()
                std = portfolio_returns.std(unbiased=False) + 1e-8  # Avoid division by zero

                # Annualized Sharpe
                sharpe = mean / std * torch.sqrt(ann_factor)
                pbar.set_postfix({'loss': f'{batch_loss.item():.4f}', 'sharpe': f'{sharpe.item():.2f}'})

        # Log epoch summary for training
        val_sharpe = logger.log_metrics(epoch=epoch, batch_idx=0, phase='val', epoch_length=epoch_length,
                                        is_epoch=True, losses=losses, **metrics)
        avg_val_loss = sum(list(losses.values())) / (batch_idx + 1)

        logger.add_plot(metrics=metrics, epoch=epoch, phase='val', tickers=tickers)
        logger.export_epoch_summary()
        # ============== Evaluation on Test Data ==============
        metrics = defaultdict(list)
        losses = defaultdict(float)
        epoch_length = len(test_loader)

        pbar = tqdm(test_loader, desc=f"Evaluating Test Set")
        with torch.inference_mode():
            for batch_idx, batch in enumerate(pbar):  # I think I am using one batch - tbd
                batch_loss = torch.tensor(0.0, dtype=X.dtype).to(device)

                X, y, dates, prediction_dates = batch['X'], batch['y'], batch['dates'], batch['prediction_dates']
                B, seq_len, stocks, features = X.shape
                X, y = X.to(device), y.to(device)
                x = X.permute(0, 2, 1, 3)  # [B, seq_len, stocks, features] -> [B, stocks, seq_len, features]
                x = x.reshape(-1, x.shape[-2], x.shape[-1])  # [B*stocks, seq_len, features]
                returns = y  # [B, pred_len, stocks]
                with torch.autocast(device.type, enabled=device.type == "cuda"):
                    weights = model(x)
                    weights = weights.reshape(B, stocks)  # [B, stocks]
                    # weights = F.softmax(weights, dim=-1)
                    weights = activation(weights, dim=-1)
                    loss_dict = loss_fn(weights, returns)
                    for loss_name, coefficient in loss_coefficients.items():
                        losses[loss_name] += (loss_dict[loss_name] * coefficient).mean().item()
                        batch_loss += (loss_dict[loss_name] * coefficient).mean()
                metrics['raw_returns'].append(returns)
                metrics['weights'].append(weights)
                metrics['dates'].extend(dates)

                portfolio_returns = ((returns * weights.unsqueeze(dim=1)).sum(dim=2) + 1).prod(
                    dim=1) - 1  # per holding period

                mean = portfolio_returns.mean()
                std = portfolio_returns.std(unbiased=False) + 1e-8  # Avoid division by zero

                # Annualized Sharpe
                sharpe = mean / std * torch.sqrt(ann_factor)
                pbar.set_postfix({'loss': f'{batch_loss.item():.4f}', 'sharpe': f'{sharpe.item():.2f}'})

        test_sharpe = logger.log_metrics(epoch=epoch, batch_idx=0, phase='test', epoch_length=epoch_length,
                                         is_epoch=True, losses=losses, **metrics)
        avg_test_loss = sum(list(losses.values())) / (batch_idx + 1)

        logger.add_plot(metrics=metrics, epoch=epoch, phase='test', tickers=tickers)
        logger.export_epoch_summary()
        # Learning rate scheduling
        if scheduler:
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            logger.writer.add_scalar('training/learning_rate', current_lr, epoch)

        # Print epoch summary
        print(f"Train Loss: {avg_train_loss:.6f}, Train Sharpe: {train_sharpe:.4f}")
        print(f"Val Loss: {avg_test_loss:.6f}, Val Sharpe: {val_sharpe:.4f}")
        print(f"Test Loss: {avg_test_loss:.6f}, Test Sharpe: {test_sharpe:.4f}")

        # Early stopping based on validation loss
        early_stopping(epoch, val_sharpe, model, str(model_path))
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

        logger.writer.flush()


def run_mps(model, train_loader, val_loader, test_loader, optimizer, scheduler, loss_fn, device, logger,
             epochs, model_path, early_stopping, loss_coefficients, tickers, **kwargs):
    best_val_sharpe = float('-inf')
    activation = entmax15
    for epoch in range(1, epochs + 1):
        print(f"\n{'=' * 50}\nEpoch {epoch}/{epochs}\n{'=' * 50}")

        # ============== Training ==============
        model.train()
        running_metrics = defaultdict(list)
        running_losses = defaultdict(float)

        all_metrics = defaultdict(list)
        all_losses = defaultdict(float)
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")

        epoch_length = len(train_loader)
        for batch_idx, batch in enumerate(pbar):
            X, y, dates, prediction_dates = batch['X'], batch['y'], batch['dates'], batch['prediction_dates']
            B, seq_len, stocks, features = X.shape
            X, returns = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            batch_loss = torch.tensor(0.0, dtype=X.dtype).to(device)
            ann_factor = torch.sqrt(torch.tensor(252.0 / returns.shape[1], device=X.device, dtype=X.dtype))

            # Prepare data for model (reshape as in original)
            x = X.permute(0, 2, 1, 3).contiguous()  # [B, seq_len, stocks, features] -> [B, stocks, seq_len, features]
            x = x.reshape(-1, x.shape[-2], x.shape[-1])  # [B*stocks, seq_len, features]

            optimizer.zero_grad()
            # Forward pass
            weights = model(x)  # [B*stocks, 1] or similar
            weights = weights.reshape(B, stocks)  # [B, stocks]
            weights = activation(weights, dim=-1)
            # clipped_weights = weights.clamp(min=loss_fn.target_min_weight, max=None)
            # Compute loss
            loss_dict = loss_fn(weights, returns)
            for loss_name, coefficient in loss_coefficients.items():
                batch_loss += (loss_dict[loss_name] * coefficient).mean()
                running_losses[loss_name] += (loss_dict[loss_name] * coefficient).mean()
                all_losses[loss_name] += (loss_dict[loss_name] * coefficient).mean()

            batch_loss.backward()
            optimizer.step()

            running_metrics['weights'].append(weights)
            running_metrics['raw_returns'].append(returns)
            all_metrics['weights'].append(weights)
            all_metrics['raw_returns'].append(returns)
            all_metrics['dates'].extend(dates)

            if (batch_idx + 1) % 5 == 0:
                # Log batch metrics (simple averages for monitoring)
                logger.log_metrics(epoch=epoch, batch_idx=batch_idx, is_epoch=False, epoch_length=epoch_length,
                                   phase='train', losses=running_losses, **running_metrics, )
                running_metrics = defaultdict(list)
                running_losses = defaultdict(float)
            # Update progress bar

            portfolio_returns = ((returns * weights.unsqueeze(dim=1)).sum(dim=2) + 1).prod(dim=1) - 1  # per holding period

            mean = portfolio_returns.mean()
            std = portfolio_returns.std(unbiased=False) + 1e-8  # Avoid division by zero

            # Annualized Sharpe
            sharpe = mean / std * torch.sqrt(ann_factor)
            pbar.set_postfix({'loss': f'{batch_loss.item():.4f}', 'sharpe': f'{sharpe.item():.2f}'})

        train_sharpe = logger.log_metrics(epoch=epoch, batch_idx=0, phase='train', epoch_length=epoch_length,
                                          is_epoch=True, losses=all_losses, **all_metrics)
        avg_train_loss = sum(list(all_losses.values())) / (batch_idx + 1)

        logger.add_plot(metrics=all_metrics, epoch=epoch, phase='train', tickers=tickers)
        # ============== Evaluation on Validation Data ==============
        model.eval()
        metrics = defaultdict(list)
        losses = defaultdict(float)
        epoch_length = len(val_loader)
        pbar = tqdm(val_loader, desc=f"Evaluating Validation Set")
        with torch.inference_mode():
            for batch_idx, batch in enumerate(pbar):  # I think I am using one batch - tbd
                batch_loss = torch.tensor(0.0, dtype=X.dtype).to(device)
                X, y, dates, prediction_dates = batch['X'], batch['y'], batch['dates'], batch['prediction_dates']
                B, seq_len, stocks, features = X.shape
                X, y = X.to(device), y.to(device)
                x = X.permute(0, 2, 1, 3)  # [B, seq_len, stocks, features] -> [B, stocks, seq_len, features]
                x = x.reshape(-1, x.shape[-2], x.shape[-1])  # [B*stocks, seq_len, features]
                returns = y  # [B, pred_len, stocks]
                weights = model(x)
                weights = weights.reshape(B, stocks)  # [B, stocks]
                # weights = F.softmax(weights, dim=-1)
                weights = activation(weights, dim=-1)
                loss_dict = loss_fn(weights, returns)
                for loss_name, coefficient in loss_coefficients.items():
                    losses[loss_name] += (loss_dict[loss_name] * coefficient).mean().item()
                    batch_loss += (loss_dict[loss_name] * coefficient).mean()
                metrics['raw_returns'].append(returns)
                metrics['weights'].append(weights)
                metrics['dates'].extend(dates)

                portfolio_returns = ((returns * weights.unsqueeze(dim=1)).sum(dim=2) + 1).prod(dim=1) - 1  # per holding period

                mean = portfolio_returns.mean()
                std = portfolio_returns.std(unbiased=False) + 1e-8  # Avoid division by zero

                # Annualized Sharpe
                sharpe = mean / std * torch.sqrt(ann_factor)
                pbar.set_postfix({'loss': f'{batch_loss.item():.4f}', 'sharpe': f'{sharpe.item():.2f}'})

        val_sharpe = logger.log_metrics(epoch=epoch, batch_idx=0, phase='val', epoch_length=epoch_length,
                                        is_epoch=True, losses=losses, **metrics)
        avg_val_loss = sum(list(losses.values())) / (batch_idx + 1)
        logger.add_plot(metrics=metrics, epoch=epoch, phase='val', tickers=tickers)

        # ============== Evaluation on Test Data ==============
        metrics = defaultdict(list)
        losses = defaultdict(float)
        epoch_length = len(test_loader)
        model.eval()

        pbar = tqdm(test_loader, desc=f"Evaluating Test Set")
        with torch.inference_mode():
            for batch_idx, batch in enumerate(pbar):  # I think I am using one batch - tbd
                batch_loss = torch.tensor(0.0, dtype=X.dtype).to(device)

                X, y, dates, prediction_dates = batch['X'], batch['y'], batch['dates'], batch['prediction_dates']
                B, seq_len, stocks, features = X.shape
                X, y = X.to(device), y.to(device)
                x = X.permute(0, 2, 1, 3)  # [B, seq_len, stocks, features] -> [B, stocks, seq_len, features]
                x = x.reshape(-1, x.shape[-2], x.shape[-1])  # [B*stocks, seq_len, features]
                returns = y  # [B, pred_len, stocks]
                weights = model(x)
                weights = weights.reshape(B, stocks)  # [B, stocks]
                # weights = F.softmax(weights, dim=-1)
                weights = activation(weights, dim=-1)
                loss_dict = loss_fn(weights, returns)
                for loss_name, coefficient in loss_coefficients.items():
                    losses[loss_name] += (loss_dict[loss_name] * coefficient).mean().item()
                    batch_loss += (loss_dict[loss_name] * coefficient).mean()
                metrics['raw_returns'].append(returns)
                metrics['weights'].append(weights)
                metrics['dates'].extend(dates)

                portfolio_returns = ((returns * weights.unsqueeze(dim=1)).sum(dim=2) + 1).prod(
                    dim=1) - 1  # per holding period

                mean = portfolio_returns.mean()
                std = portfolio_returns.std(unbiased=False) + 1e-8  # Avoid division by zero

                # Annualized Sharpe
                sharpe = mean / std * torch.sqrt(ann_factor)
                pbar.set_postfix({'loss': f'{batch_loss.item():.4f}', 'sharpe': f'{sharpe.item():.2f}'})

        test_sharpe = logger.log_metrics(epoch=epoch, batch_idx=0, phase='test', epoch_length=epoch_length,
                                         is_epoch=True, losses=losses, **metrics)
        avg_test_loss = sum(list(losses.values())) / (batch_idx + 1)
        logger.add_plot(metrics=metrics, epoch=epoch, phase='test', tickers=tickers)
        logger.export_epoch_summary()

        # Learning rate scheduling
        if scheduler:
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            logger.writer.add_scalar('training/learning_rate', current_lr, epoch)
        print(f"Train Loss: {avg_train_loss:.6f}, Train Sharpe: {train_sharpe:.4f}")
        print(f"Val Loss: {avg_test_loss:.6f}, Val Sharpe: {val_sharpe:.4f}")
        print(f"Test Loss: {avg_test_loss:.6f}, Test Sharpe: {test_sharpe:.4f}")


        # Early stopping based on validation loss
        early_stopping(epoch, val_sharpe, model, str(model_path))
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

        logger.writer.flush()


def main(args, model: torch.nn.Module, config: dict):
    device = set_device()
    set_seed(args.seed)

    # Create descriptive run name
    if args.run_name:
        run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.run_name}"
    else:
        run_name = create_run_name(args)

    # Setup directories
    log_dir, model_path = setup_directories(args.model, run_name)

    print(f"Run name: {run_name}")
    print(f"Log directory: {log_dir}")
    print(f"Model will be saved to: {model_path}")

    data_dir = Path(__file__).resolve().parent.parent / args.data_root
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch_size,
                                                            root_dir=data_dir,
                                                            lookback=args.lookback,
                                                            prediction_horizon=args.holding_period)
    tickers = get_tickers(root_dir=data_dir)
    loss_fn = LongOnlyMarkowitzPortfolioLoss(target_volatility=args.volatility,
                                             target_cardinality=args.components,
                                             target_min_weight=args.min_weight,
                                             target_min_utility=args.utility,
                                             risk_aversion=args.risk_aversion,)

    # Model
    model.to(device)

    # Training components
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.scheduler_type is not None:
        print(f"Using scheduler type: {args.scheduler_type}")
        scheduler = get_scheduler(optimizer, args.scheduler_type)
    else:
        scheduler = get_scheduler(optimizer, 'plateau')

    early_stopping = EarlyStopping(patience=args.patience, warmup_epochs=args.warmup_epochs, mode='max')
    logger = PortfolioTensorBoardLogger(log_dir=str(log_dir))

    # Save configuration with run name
    config = {
        'run_name': run_name,
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'patience': args.patience,
        'device': str(device),
        'min_weight': args.min_weight,
        'components': args.components,
        'volatility': args.volatility,
        'utility': args.utility,
        'lc_utility': args.lc_utility,
        'lc_components': args.lc_components,
        'lc_volatility': args.lc_volatility,
        'lc_min_weight': args.lc_min_weight,

        'lookback': args.lookback,
        'holding_period': args.holding_period,
        'scheduler_type': args.scheduler_type,
        'scaler': device.type == 'cuda',
    } | config


    with open(log_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Starting training with config:\n{pp(config)}")

    loss_coefficients = dict(utility_target_penalty=args.lc_utility,
                             cardinality_target_penalty=args.lc_components,
                             vol_target_penalty=args.lc_volatility,
                             min_weight_penalty=args.lc_min_weight)

    if device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        run_cuda(
            model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, device=device, logger=logger,
            epochs=args.epochs, model_path=model_path, early_stopping=early_stopping,
            loss_coefficients=loss_coefficients, scaler=scaler, tickers=tickers
        )
    else:
        run_mps(
            model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, device=device, logger=logger,
            epochs=args.epochs, model_path=model_path, early_stopping=early_stopping,
            loss_coefficients=loss_coefficients, tickers=tickers
        )

    logger.writer.close()
    print(f"Training complete. Results saved to {log_dir}")


def add_common_args(p):
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--scheduler_type', type=str, default='plateau')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--warmup_epochs', type=int, default=10)
    p.add_argument('--data_root', type=str, default='data/market_data')
    p.add_argument('--lookback', type=int, default=30)
    p.add_argument('--holding_period', type=int, default=5)
    p.add_argument('--run_name', type=str, default=None)

    p.add_argument('--min_weight', type=float, default=0.005)
    p.add_argument('--components', type=int, default=40)
    p.add_argument('--volatility', type=float, default=0.14)
    p.add_argument('--risk_aversion', type=float, default=5.0)
    p.add_argument('--utility', type=float, default=5.0)

    p.add_argument('--lc_utility', type=float, default=10.0)
    p.add_argument('--lc_components', type=float, default=5.0)
    p.add_argument('--lc_volatility', type=float, default=0.0)
    p.add_argument('--lc_min_weight', type=float, default=0.0)


def patch_tst(argv):
    parser = argparse.ArgumentParser(description='PatchTST args')
    add_common_args(parser)
    parser.add_argument('--model', type=str, default='patch_tst')  # <-- add
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--enc_in', type=int, default=32)
    parser.add_argument('--patch_len', type=int, default=8)
    parser.add_argument('--stride', type=int, default=4)

    args = parser.parse_args(argv)
    config = dict(d_model=args.d_model,
                  n_heads=args.n_heads,
                  d_ff=args.d_ff,
                  dropout=args.dropout,
                  e_layers=args.e_layers,
                  factor=args.factor,
                  activation=args.activation,
                  enc_in=args.enc_in,
                  patch_len=args.patch_len,
                  stride=args.stride)
    model = PatchTSTModel(**config)
    config['model'] = 'patch_tst'
    main(args, model=model, config=config)


def bidirectional_lstm(argv):
    parser = argparse.ArgumentParser(description='BiLSTM args')
    add_common_args(parser)

    parser.add_argument('--model', type=str, default='bidirectional_lstm')  # <-- add
    parser.add_argument('--input_dim', type=int, default=45)
    parser.add_argument('--stem_dims', type=int, nargs='+', default=[64, 128, 16])
    parser.add_argument('--stem_dropout', type=float, default=0.2)
    parser.add_argument('--lstm_hidden_dim', type=int, default=128)  # NOTE: "hidden"
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--lstm_dropout', type=float, default=0.1)
    parser.add_argument('--head_dims', type=int, nargs='+', default=[64, 1])
    parser.add_argument('--head_dropout', type=float, default=0.1)
    parser.add_argument('--activation', type=str, default='gelu')

    args = parser.parse_args(argv)
    config = dict(input_size=args.input_dim,
                  stem_dims=args.stem_dims,
                  stem_dropout=args.stem_dropout,
                  lstm_hidden_dim=args.lstm_hidden_dim,
                  lstm_layers=args.lstm_layers,
                  lstm_dropout=args.lstm_dropout,
                  head_dims=args.head_dims,
                  head_dropout=args.head_dropout,
                  activation_name=args.activation)
    model = BidirectionalLSTM(**config)
    config['model'] = 'bidirectional_lstm'
    main(args, model=model, config=config)


def main_old():
    parser = argparse.ArgumentParser(description='Train Portfolio Optimization Model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--scheduler_type', type=str, default='plateau', help='Optimiser scheduler')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Warm up epochs for early stopping')
    parser.add_argument('--data_root', type=str, default='data/market_data', help='Location of market data')
    parser.add_argument('--lookback', type=int, default=30, help='Lookback horizon for time series')
    parser.add_argument('--holding_period', type=int, default=5, help='Holding period for portfolio rebalancing')

    parser.add_argument('--run_name', type=str, default='patch_tst', help='Custom run name (optional)')

    parser.add_argument('--min_weight', type=float, default=0.005, help='Target lower bound for weights')
    parser.add_argument('--components', type=int, default=50, help='Target number of portfolio components')
    parser.add_argument('--volatility', type=float, default=0.14, help='Target portfolio volatility')
    parser.add_argument('--risk_aversion', type=float, default=5.0, help='Risk aversion coefficient for the utility function')
    parser.add_argument('--utility', type=float, default=5.0, help='Target portfolio utility = (mean return - risk aversion * portfolio variance) * annualisation factor')

    parser.add_argument('--lc_utility', type=float, default=10.0,
                        help='Loss function coefficient for negative Sharpe ratio')
    parser.add_argument('--lc_components', type=float, default=5.0,
                        help='Loss function coefficient enforcing portfolio components')
    parser.add_argument('--lc_volatility', type=float, default=0.0,
                        help='Loss function coefficient for volatility target')
    parser.add_argument('--lc_min_weight', type=float, default=0.0, help='Loss target coefficient for minimum weight')

    parser.add_argument('--model', type=str, default='patch_tst', help='Model name')
    parser.add_argument('--d_model', type=int, default=64, help='Model dimension for PatchTST')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads for PatchTST')
    parser.add_argument('--d_ff', type=int, default=128, help='Feed-forward dimension for PatchTST')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for PatchTST')
    parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers for PatchTST')
    parser.add_argument('--factor', type=int, default=1, help='Factor for PatchTST')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function for PatchTST')
    parser.add_argument('--enc_in', type=int, default=32, help='Input dimension for PatchTST')
    parser.add_argument('--patch_len', type=int, default=8, help='Patch length for PatchTST')
    parser.add_argument('--stride', type=int, default=4, help='Stride for PatchTST')

    args = parser.parse_args()

    device = set_device()
    set_seed(args.seed)

    # Create descriptive run name
    if args.run_name:
        run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.run_name}"
    else:
        run_name = create_run_name(args)

    # Setup directories
    log_dir, model_path = setup_directories(args.model, run_name)

    print(f"Run name: {run_name}")
    print(f"Log directory: {log_dir}")
    print(f"Model will be saved to: {model_path}")

    data_dir = Path(__file__).resolve().parent.parent / args.data_root
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch_size,
                                                            root_dir=data_dir,
                                                            lookback=args.lookback,
                                                            prediction_horizon=args.holding_period)
    tickers = get_tickers(root_dir=data_dir)
    loss_fn = LongOnlyMarkowitzPortfolioLoss(target_volatility=args.volatility,
                                             target_cardinality=args.components,
                                             target_min_weight=args.min_weight,
                                             target_min_utility=args.utility,
                                             risk_aversion=args.risk_aversion,)

    # Model
    if args.model == 'patch_tst':
        model = PatchTSTModel(
            lookback_len=args.lookback, pred_len=1, d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
            dropout=args.dropout, e_layers=args.e_layers, factor=args.factor, activation=args.activation,
            enc_in=args.enc_in, patch_len=args.patch_len, stride=args.stride
        ).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # Training components
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.scheduler_type is not None:
        print(f"Using scheduler type: {args.scheduler_type}")
        scheduler = get_scheduler(optimizer, args.scheduler_type)
    else:
        scheduler = get_scheduler(optimizer, 'plateau')

    early_stopping = EarlyStopping(patience=args.patience, warmup_epochs=args.warmup_epochs, mode='max')
    logger = PortfolioTensorBoardLogger(log_dir=str(log_dir))

    # Save configuration with run name
    config = {
        'run_name': run_name,
        'model': args.model,
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'patience': args.patience,
        'device': str(device),
        'min_weight': args.min_weight,
        'components': args.components,
        'volatility': args.volatility,
        'utility': args.utility,
        'lc_utility': args.lc_utility,
        'lc_components': args.lc_components,
        'lc_volatility': args.lc_volatility,
        'lc_min_weight': args.lc_min_weight,

        # Model-specific parameters
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'd_ff': args.d_ff,
        'dropout': args.dropout,
        'e_layers': args.e_layers,
        'factor': args.factor,
        'activation': args.activation,
        'enc_in': args.enc_in,
        'patch_len': args.patch_len,
        'stride': args.stride,
        'lookback': args.lookback,
        'holding_period': args.holding_period,
        'scheduler_type': args.scheduler_type,
        'scaler': device.type == 'cuda',
    }

    with open(log_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Starting training with config:\n{pp(config)}")

    loss_coefficients = dict(utility_target_penalty=args.lc_utility,
                             cardinality_target_penalty=args.lc_components,
                             vol_target_penalty=args.lc_volatility,
                             min_weight_penalty=args.lc_min_weight)

    if device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
        run_cuda(
            model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, device=device, logger=logger,
            epochs=args.epochs, model_path=model_path, early_stopping=early_stopping,
            loss_coefficients=loss_coefficients, scaler=scaler, tickers=tickers
        )
    else:
        run_mps(
            model=model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            optimizer=optimizer, scheduler=scheduler, loss_fn=loss_fn, device=device, logger=logger,
            epochs=args.epochs, model_path=model_path, early_stopping=early_stopping,
            loss_coefficients=loss_coefficients, tickers=tickers
        )

    logger.writer.close()
    print(f"Training complete. Results saved to {log_dir}")


def run():
    top = argparse.ArgumentParser(description='Run price model (selector)')
    top.add_argument('--model', default='bidirectional_lstm', type=str,# required=True,
                     choices=['patch_tst', 'bidirectional_lstm'])
    args, remaining = top.parse_known_args()

    if args.model == 'patch_tst':
        patch_tst(remaining)
    elif args.model == 'bidirectional_lstm':
        bidirectional_lstm(remaining)
    else:
        raise ValueError(f"Unknown model: {args.model}")

if __name__ == '__main__':
    run()