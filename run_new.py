import argparse
import json
import os
import sys
import dgl
import numpy as np
import torch
import torch.optim as optim
from entmax import entmax15
from tqdm import tqdm
from typing import Optional, Dict
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from pprint import pp
import pickle
import gzip

# Add parent directory to Python path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from price_models.dataloader import get_dataloaders, get_tickers
from price_models.patch_tst import PatchTSTModel
from price_models.bidirectional_lstm import BidirectionalLSTM
from graph_models.dataloader import (load_temporal_knowledge_graph, collate_fn as graph_collate_fn,
                                     TemporalKnowledgeGraphStatics)
from graph_models.news_graph import GraphModel, initialise_embeddings, MultiAspectEmbedding
from cross_attention import CrossAttentionHead
from losses import LongOnlyMarkowitzPortfolioLoss
from utils import EarlyStopping, PortfolioTensorBoardLogger, get_scheduler, set_device, set_seed


# =============================================================================
# TENSOR SIZE CHECKING AND MODEL PACKING FUNCTIONS
# =============================================================================

def check_tensor_sizes(node_latest_event_time, static_emb, dynamic_emb):
    """Check tensor sizes to predict potential issues before saving"""
    print("Checking tensor sizes...")

    # Check node_latest_event_time
    nlet_size = node_latest_event_time.numel()
    nlet_mb = (nlet_size * 4) / (1024 ** 2)  # Assuming float32
    print(f"node_latest_event_time: {node_latest_event_time.shape} = {nlet_size:,} elements ({nlet_mb:.1f} MB)")

    if nlet_size > 2 ** 31 - 1:
        print("  WARNING: Too large for sparse conversion!")

    # Check embeddings
    static_struct_size = static_emb.structural.numel()
    static_temp_size = static_emb.temporal.numel()
    dynamic_struct_size = dynamic_emb.structural.numel()
    dynamic_temp_size = dynamic_emb.temporal.numel()

    print(f"static_emb.structural: {static_emb.structural.shape} = {static_struct_size:,} elements")
    print(f"static_emb.temporal: {static_emb.temporal.shape} = {static_temp_size:,} elements")
    print(f"dynamic_emb.structural: {dynamic_emb.structural.shape} = {dynamic_struct_size:,} elements")
    print(f"dynamic_emb.temporal: {dynamic_emb.temporal.shape} = {dynamic_temp_size:,} elements")

    total_elements = nlet_size + static_struct_size + static_temp_size + dynamic_struct_size + dynamic_temp_size
    total_mb = (total_elements * 4) / (1024 ** 2)
    print(f"Total tensor elements: {total_elements:,} ({total_mb:.1f} MB)")

    return {
        'node_latest_event_time_size': nlet_size,
        'total_size': total_elements,
        'total_mb': total_mb,
        'sparse_safe': nlet_size <= 2 ** 31 - 1
    }


def pack_model_optimized(price_model, graph_model, static_entity_emb,
                         val_dynamic_entity_emb, node_latest_event_time, val_sharpe):
    """Optimized model packing for very large tensors with compression"""

    # Get devices for proper saving
    price_device = next(price_model.parameters()).device
    graph_device = next(graph_model.parameters()).device
    print(f"Packing model - Price device: {price_device}, Graph device: {graph_device}")

    # Check tensor size
    tensor_size = node_latest_event_time.numel()
    tensor_gb = (tensor_size * 4) / (1024 ** 3)  # float32 = 4 bytes
    print(f"node_latest_event_time: {tensor_size:,} elements ({tensor_gb:.1f} GB)")

    # Move smaller tensors to CPU normally
    static_emb_cpu = MultiAspectEmbedding(
        structural=static_entity_emb.structural.cpu(),
        temporal=static_entity_emb.temporal.cpu()
    )

    dynamic_emb_cpu = MultiAspectEmbedding(
        structural=val_dynamic_entity_emb.structural.cpu(),
        temporal=val_dynamic_entity_emb.temporal.cpu()
    )

    # Handle the massive tensor differently based on size
    if tensor_gb > 5.0:  # If larger than 5GB, use special handling
        print("Using memory-efficient handling for large tensor...")

        # Store only non-zero entries if tensor is sparse
        node_tensor_cpu = node_latest_event_time.cpu()
        nonzero_mask = node_tensor_cpu != 0
        nonzero_count = nonzero_mask.sum().item()
        sparsity = 1 - (nonzero_count / tensor_size)

        print(f"Tensor sparsity: {sparsity:.4f} ({nonzero_count:,} non-zero elements)")

        if sparsity > 0.9:  # If more than 90% zeros, store as coordinates + values
            print("Storing as coordinate format due to high sparsity")
            nonzero_indices = torch.nonzero(node_tensor_cpu, as_tuple=False)
            nonzero_values = node_tensor_cpu[nonzero_mask]

            node_latest_event_time_data = {
                'format': 'coordinate',
                'shape': node_tensor_cpu.shape,
                'indices': nonzero_indices,
                'values': nonzero_values,
                'dtype': node_tensor_cpu.dtype
            }
            del node_tensor_cpu  # Free memory immediately
        else:
            print("Storing as compressed dense tensor")
            # Compress the tensor
            import io
            buffer = io.BytesIO()
            torch.save(node_tensor_cpu, buffer)
            buffer.seek(0)
            compressed_data = gzip.compress(buffer.read())

            node_latest_event_time_data = {
                'format': 'compressed_dense',
                'shape': node_tensor_cpu.shape,
                'compressed_data': compressed_data,
                'dtype': node_tensor_cpu.dtype
            }
            del node_tensor_cpu  # Free memory immediately

        print(f"Compressed size: {len(compressed_data) / (1024 ** 3):.1f} GB")
    else:
        # Small enough to handle normally
        node_latest_event_time_data = {
            'format': 'dense',
            'tensor': node_latest_event_time.cpu()
        }

    # Create the packed model dictionary
    packed_model = {
        'price_model_state_dict': price_model.state_dict(),
        'graph_model_state_dict': graph_model.state_dict(),
        'static_entity_emb': static_emb_cpu,
        'dynamic_entity_emb': dynamic_emb_cpu,
        'node_latest_event_time': node_latest_event_time_data,
        'sharpe': val_sharpe,
        'price_device': str(price_device),
        'graph_device': str(graph_device),
        'model_info': {
            'price_model_type': type(price_model).__name__,
            'graph_model_type': type(graph_model).__name__,
            'static_emb_shapes': {
                'structural': static_entity_emb.structural.shape,
                'temporal': static_entity_emb.temporal.shape
            },
            'dynamic_emb_shapes': {
                'structural': val_dynamic_entity_emb.structural.shape,
                'temporal': val_dynamic_entity_emb.temporal.shape
            },
            'original_tensor_size_gb': tensor_gb
        }
    }

    print(f"Model packed successfully with optimized large tensor handling")
    return packed_model


def pack_model_lightweight(price_model, graph_model, val_sharpe):
    """Ultra-lightweight model packing - only saves model weights"""
    print("Using lightweight model packing (model weights only)")

    packed_model = {
        'price_model_state_dict': price_model.state_dict(),
        'graph_model_state_dict': graph_model.state_dict(),
        'sharpe': val_sharpe,
        'price_device': str(next(price_model.parameters()).device),
        'graph_device': str(next(graph_model.parameters()).device),
        'model_info': {
            'price_model_type': type(price_model).__name__,
            'graph_model_type': type(graph_model).__name__,
        }
    }

    print("Lightweight model packed (embeddings and tensors NOT saved)")
    return packed_model


# =============================================================================
# TRAINING CONTROL FUNCTIONS
# =============================================================================

def setup_differentiated_optimizer(price_model, graph_model, cross_attention_head,
                                   static_emb, dynamic_emb, args):
    """Setup optimizer with different learning rates for price vs graph components"""

    price_params = list(price_model.parameters())
    cross_attention_params = list(cross_attention_head.parameters())
    graph_params = list(graph_model.parameters()) + [
        static_emb.structural, static_emb.temporal,
        dynamic_emb.structural, dynamic_emb.temporal
    ]

    # Use much smaller learning rate for graph components
    optimizer = optim.Adam([
        {'params': price_params, 'lr': args.lr, 'weight_decay': 1e-5},
        {'params': cross_attention_params, 'lr': args.lr, 'weight_decay': 1e-5},
        {'params': graph_params, 'lr': args.lr * args.graph_lr_ratio, 'weight_decay': 1e-4}
    ])

    print(f"Price model LR: {args.lr}")
    print(f"Graph model LR: {args.lr * args.graph_lr_ratio}")

    return optimizer


def setup_staged_optimizer(price_model, graph_model, cross_attention_head,
                           static_emb, dynamic_emb, args, stage='joint'):
    """Setup optimizer for staged training"""

    price_params = list(price_model.parameters())
    cross_attention_params = list(cross_attention_head.parameters())
    graph_params = list(graph_model.parameters()) + [
        static_emb.structural, static_emb.temporal,
        dynamic_emb.structural, dynamic_emb.temporal
    ]

    if stage == 'price_only':
        print("STAGE: Training price model only")
        # Freeze graph components
        for param in graph_params:
            param.requires_grad = False
        optimizer = optim.Adam(price_params + cross_attention_params, lr=args.lr, weight_decay=1e-5)

    elif stage == 'graph_only':
        print("STAGE: Training graph model only (price frozen)")
        # Freeze price model
        for param in price_params:
            param.requires_grad = False
        optimizer = optim.Adam(graph_params + cross_attention_params, lr=args.lr, weight_decay=1e-5)

    else:  # joint
        print("STAGE: Joint training with differentiated learning rates")
        # Unfreeze everything
        for param in price_params + graph_params:
            param.requires_grad = True

        optimizer = optim.Adam([
            {'params': price_params, 'lr': args.lr * 0.1},  # Much slower for pre-trained price model
            {'params': cross_attention_params, 'lr': args.lr},
            {'params': graph_params, 'lr': args.lr * args.graph_lr_ratio}
        ], weight_decay=1e-5)

    return optimizer


def apply_gradient_controls(price_model, graph_model, static_emb, dynamic_emb,
                            max_grad_ratio=0.1, clip_value=1.0):
    """Apply gradient clipping and balancing"""

    # Clip gradients first
    torch.nn.utils.clip_grad_norm_(price_model.parameters(), clip_value)
    torch.nn.utils.clip_grad_norm_(graph_model.parameters(), clip_value)

    # Calculate gradient norms
    price_grad_norm = 0
    for param in price_model.parameters():
        if param.grad is not None:
            price_grad_norm += param.grad.data.norm(2) ** 2
    price_grad_norm = price_grad_norm ** 0.5

    # Scale graph gradients if too large
    graph_components = [graph_model, static_emb.structural, static_emb.temporal,
                        dynamic_emb.structural, dynamic_emb.temporal]

    for component in graph_components:
        if hasattr(component, 'parameters'):
            for param in component.parameters():
                if param.grad is not None and price_grad_norm > 0:
                    graph_grad_norm = param.grad.data.norm(2)
                    if graph_grad_norm > max_grad_ratio * price_grad_norm:
                        scale = (max_grad_ratio * price_grad_norm) / graph_grad_norm
                        param.grad.data *= scale
        elif hasattr(component, 'grad') and component.grad is not None:
            graph_grad_norm = component.grad.data.norm(2)
            if graph_grad_norm > max_grad_ratio * price_grad_norm and price_grad_norm > 0:
                scale = (max_grad_ratio * price_grad_norm) / graph_grad_norm
                component.grad.data *= scale


def load_pretrained_price_model(price_model, pretrained_path):
    """Load pre-trained price model weights"""
    if pretrained_path and Path(pretrained_path).exists():
        print(f"Loading pre-trained price model from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')

        if 'price_model_state_dict' in checkpoint:
            price_model.load_state_dict(checkpoint['price_model_state_dict'])
        else:
            price_model.load_state_dict(checkpoint)  # Direct state dict

        print("Pre-trained price model loaded successfully")
        return True
    else:
        print("No pre-trained price model found, training from scratch")
        return False


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_run_name(args):
    """Create a descriptive run name with key hyperparameters"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    lr_str = f"lr{args.lr:.0e}".replace('-0', '-').replace('+0', '+')
    param_str = f"bs{args.batch_size}_ep{args.epochs}_{lr_str}"

    if hasattr(args, 'patience') and args.patience != 15:
        param_str += f"_pat{args.patience}"
    if hasattr(args, 'seed') and args.seed != 42:
        param_str += f"_seed{args.seed}"

    return f"{timestamp}_{param_str}"


def setup_directories(model_name, run_name):
    """Setup log and model directories with descriptive names"""
    base_dir = Path(__file__).parent
    log_dir = base_dir / 'logs' / model_name / run_name
    model_dir = base_dir / 'models' / model_name

    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f'best_model_{run_name}.pt'
    return log_dir, model_path


# =============================================================================
# MAIN TRAINING FUNCTION - TRAINING LOOP ONLY
# =============================================================================

def run_cuda_multi_gpu(price_model: torch.nn.Module,
                       graph_model: torch.nn.Module,
                       price_train_loader: torch.utils.data.DataLoader,
                       price_val_loader: torch.utils.data.DataLoader,
                       price_test_loader: torch.utils.data.DataLoader,
                       G: dgl.graph,
                       graph_statics: TemporalKnowledgeGraphStatics,
                       init_static_emb: MultiAspectEmbedding,
                       init_dynamic_emb: MultiAspectEmbedding,
                       node_latest_event_time: torch.Tensor,
                       optimizer: Optional[torch.optim.Optimizer],
                       scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
                       loss_fn: torch.nn.Module,
                       logger: PortfolioTensorBoardLogger,
                       epochs: int,
                       model_path: Path,
                       early_stopping: EarlyStopping,
                       loss_coefficients: Dict[str, float],
                       tickers: np.ndarray,
                       cross_attention_head: torch.nn.Module,
                       price_model_device: torch.device,
                       graph_model_device: torch.device,
                       scaler: Optional[torch.cuda.amp.GradScaler] = None,
                       graph_warmup_epochs: int = 20,
                       max_graph_grad_ratio: float = 0.1,
                       **kwargs):
    activation = entmax15

    for epoch in range(1, epochs + 1):
        print(f"\n{'=' * 50}\nEpoch {epoch}/{epochs}\n{'=' * 50}")

        # Reset graph state
        graph_model.node_latest_event_time.zero_()
        node_latest_event_time.zero_()
        dynamic_emb = init_dynamic_emb

        # ============== Training ==============
        graph_model.train()
        price_model.train()
        running_metrics = defaultdict(list)
        running_losses = defaultdict(float)
        all_metrics = defaultdict(list)
        all_losses = defaultdict(float)

        date2idx = graph_statics.date2idx
        idx2date = graph_statics.idx2date
        epoch_length = len(price_train_loader)

        # Process initial dates prior to first prediction date
        prediction_date = next(iter(price_train_loader))['prediction_dates'][0]
        start_idx = date2idx[prediction_date]

        for i in range(start_idx):
            batch_G, date = graph_collate_fn([(i, idx2date[i]), ], G)
            if batch_G is None:
                continue
            dynamic_emb = graph_model(batch_G=batch_G, static_entity_emb=init_static_emb,
                                      dynamic_entity_emb=dynamic_emb)

        pbar = tqdm(price_train_loader, desc=f"Training Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            embeddings = []
            optimizer.zero_grad(set_to_none=True)

            X, y, dates, prediction_dates = batch['X'], batch['y'], batch['dates'], batch['prediction_dates']
            combined = None
            end_idx = date2idx[prediction_dates[-1]] + 1

            # Process graph on graph GPU
            for i in range(start_idx, end_idx):
                batch_G, date = graph_collate_fn([(i, idx2date[i]), ], G)
                if batch_G is None:
                    if date in prediction_dates:
                        if combined is None:
                            combined = graph_model.combine(dynamic_entity_emb=dynamic_emb)
                        embeddings.append(combined)
                    continue
                dynamic_emb = graph_model(batch_G=batch_G, static_entity_emb=init_static_emb,
                                          dynamic_entity_emb=dynamic_emb)
                if date in prediction_dates:
                    combined = graph_model.combine(dynamic_entity_emb=dynamic_emb)
                    embeddings.append(combined)

            start_idx = end_idx
            B, seq_len, stocks, features = X.shape
            assert len(embeddings) == B, f"Expected {B} embeddings, got {len(embeddings)}"

            # Transfer embeddings from graph GPU to price GPU
            embeddings_tensor = torch.cat(embeddings, dim=0).reshape(B, *embeddings[0].shape)
            embeddings_on_price_gpu = embeddings_tensor.to(price_model_device, non_blocking=True)

            X, returns = X.to(price_model_device, non_blocking=True), y.to(price_model_device, non_blocking=True)
            batch_loss = torch.tensor(0.0, dtype=X.dtype, device=price_model_device)

            # Prepare data for model
            x = X.permute(0, 2, 1, 3).contiguous()
            x = x.reshape(-1, x.shape[-2], x.shape[-1])

            # Forward pass with graph contribution scheduling
            with torch.autocast(price_model_device.type):
                # Price model forward
                price_features = price_model(x)
                price_features = price_features.reshape(B, stocks, price_features.shape[-1])

                # Graph contribution scheduling (gradually introduce graph)
                if epoch <= graph_warmup_epochs:
                    graph_weight = epoch / graph_warmup_epochs

                    # Price-only baseline
                    price_only = torch.mean(price_features, dim=-1)
                    price_only = activation(price_only, dim=-1)

                    # Graph-enhanced
                    graph_enhanced = cross_attention_head(price_features, embeddings_on_price_gpu, None)
                    graph_enhanced = activation(graph_enhanced, dim=-1)

                    # Blend gradually
                    weights = (1 - graph_weight) * price_only + graph_weight * graph_enhanced

                    if batch_idx == 0:  # Print once per epoch
                        print(f"Graph warmup: {graph_weight:.3f}")
                else:
                    # Full graph integration
                    weights = cross_attention_head(price_features, embeddings_on_price_gpu, None)
                    weights = activation(weights, dim=-1)

                # Compute loss
                loss_dict = loss_fn(weights, returns)
                for loss_name, coefficient in loss_coefficients.items():
                    batch_loss += (loss_dict[loss_name] * coefficient).mean()
                    running_losses[loss_name] += (loss_dict[loss_name] * coefficient).mean()
                    all_losses[loss_name] += (loss_dict[loss_name] * coefficient).mean()

            # Backward pass with gradient controls
            if scaler is not None:
                scaler.scale(batch_loss).backward()
                scaler.unscale_(optimizer)
                apply_gradient_controls(price_model, graph_model, init_static_emb, init_dynamic_emb,
                                        max_grad_ratio=max_graph_grad_ratio)
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_loss.backward()
                apply_gradient_controls(price_model, graph_model, init_static_emb, init_dynamic_emb,
                                        max_grad_ratio=max_graph_grad_ratio)
                optimizer.step()

            # Clear GPU cache and detach embeddings
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

            dynamic_emb = MultiAspectEmbedding(
                structural=dynamic_emb.structural.detach(),
                temporal=dynamic_emb.temporal.detach()
            )

            # Track metrics
            running_metrics['weights'].append(weights)
            running_metrics['raw_returns'].append(returns)
            all_metrics['weights'].append(weights)
            all_metrics['raw_returns'].append(returns)
            all_metrics['dates'].extend(dates)

            ann_factor = torch.sqrt(torch.tensor(252.0 / returns.shape[1], device=price_model_device, dtype=X.dtype))

            if (batch_idx + 1) % 5 == 0:
                logger.log_metrics(epoch=epoch, batch_idx=batch_idx, is_epoch=False, epoch_length=epoch_length,
                                   phase='train', losses=running_losses, **running_metrics)
                running_metrics = defaultdict(list)
                running_losses = defaultdict(float)

            # Update progress bar
            portfolio_returns = ((returns * weights.unsqueeze(dim=1)).sum(dim=2) + 1).prod(dim=1) - 1
            mean = portfolio_returns.mean()
            std = portfolio_returns.std(unbiased=False) + 1e-8
            sharpe = mean / std * torch.sqrt(ann_factor)
            pbar.set_postfix({'loss': f'{batch_loss.item():.4f}', 'sharpe': f'{sharpe.item():.2f}'})

        # Log epoch training metrics
        train_sharpe = logger.log_metrics(epoch=epoch, batch_idx=0, phase='train', epoch_length=epoch_length,
                                          is_epoch=True, losses=all_losses, **all_metrics)
        avg_train_loss = sum(list(all_losses.values())) / (batch_idx + 1)
        logger.add_plot(metrics=all_metrics, epoch=epoch, phase='train', tickers=tickers)
        torch.cuda.empty_cache()

        # Call evaluation and early stopping functions (defined in Part 2B)
        val_sharpe, avg_val_loss, test_sharpe, avg_test_loss = evaluate_and_save(
            epoch, price_model, graph_model, cross_attention_head,
            price_val_loader, price_test_loader, G, graph_statics,
            init_static_emb, dynamic_emb, node_latest_event_time,
            loss_fn, loss_coefficients, logger, tickers,
            price_model_device, graph_model_device, start_idx,
            date2idx, idx2date, activation, early_stopping, model_path
        )

        # Learning rate scheduling
        if scheduler:
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            logger.writer.add_scalar('training/learning_rate', current_lr, epoch)

        # Print epoch summary
        print(f"Train Loss: {avg_train_loss:.6f}, Train Sharpe: {train_sharpe:.4f}")
        print(f"Val Loss: {avg_val_loss:.6f}, Val Sharpe: {val_sharpe:.4f}")
        print(f"Test Loss: {avg_test_loss:.6f}, Test Sharpe: {test_sharpe:.4f}")
        logger.writer.flush()

        # Check early stopping
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

        torch.cuda.empty_cache()


# =============================================================================
# EVALUATION AND EARLY STOPPING FUNCTIONS
# =============================================================================

def evaluate_and_save(epoch, price_model, graph_model, cross_attention_head,
                      price_val_loader, price_test_loader, G, graph_statics,
                      init_static_emb, dynamic_emb, node_latest_event_time,
                      loss_fn, loss_coefficients, logger, tickers,
                      price_model_device, graph_model_device, start_idx,
                      date2idx, idx2date, activation, early_stopping, model_path):
    """Run validation and test evaluation, handle early stopping"""

    def evaluate_phase(data_loader, phase_name):
        metrics = defaultdict(list)
        losses = defaultdict(float)
        phase_start_idx = start_idx

        pbar = tqdm(data_loader, desc=f"Evaluating {phase_name} Set")
        with torch.inference_mode():
            for batch_idx, batch in enumerate(pbar):
                embeddings = []
                X, y, dates, prediction_dates = batch['X'], batch['y'], batch['dates'], batch['prediction_dates']
                combined = None
                end_idx = date2idx[prediction_dates[-1]] + 1

                # Process graph
                for i in range(phase_start_idx, end_idx):
                    batch_G, date = graph_collate_fn([(i, idx2date[i]), ], G)
                    if batch_G is None:
                        if date in prediction_dates:
                            if combined is None:
                                combined = graph_model.combine(dynamic_entity_emb=dynamic_emb)
                            embeddings.append(combined)
                        continue
                    dynamic_emb = graph_model(batch_G=batch_G, static_entity_emb=init_static_emb,
                                              dynamic_entity_emb=dynamic_emb)
                    if date in prediction_dates:
                        combined = graph_model.combine(dynamic_entity_emb=dynamic_emb)
                        embeddings.append(combined)

                phase_start_idx = end_idx

                if len(X.shape) == 3:
                    X = X.unsqueeze(0)

                B, seq_len, stocks, features = X.shape
                assert len(embeddings) == B, f"Expected {B} embeddings, got {len(embeddings)}"

                # Transfer to price GPU and forward pass
                embeddings_tensor = torch.cat(embeddings, dim=0).reshape(B, *embeddings[0].shape)
                embeddings_on_price_gpu = embeddings_tensor.to(price_model_device, non_blocking=True)
                X, returns = X.to(price_model_device, non_blocking=True), y.to(price_model_device, non_blocking=True)

                x = X.permute(0, 2, 1, 3).contiguous()
                x = x.reshape(-1, x.shape[-2], x.shape[-1])

                with torch.autocast(device_type=price_model_device.type):
                    price_features = price_model(x)
                    price_features = price_features.reshape(B, stocks, price_features.shape[-1])
                    weights = cross_attention_head(price_features, embeddings_on_price_gpu, None)
                    weights = activation(weights, dim=-1)

                    loss_dict = loss_fn(weights, returns)
                    batch_loss = torch.tensor(0.0, dtype=X.dtype, device=price_model_device)
                    for loss_name, coefficient in loss_coefficients.items():
                        losses[loss_name] += (loss_dict[loss_name] * coefficient).mean().item()
                        batch_loss += (loss_dict[loss_name] * coefficient).mean()

                metrics['raw_returns'].append(returns)
                metrics['weights'].append(weights)
                metrics['dates'].extend(dates)

                # Progress update
                portfolio_returns = ((returns * weights.unsqueeze(dim=1)).sum(dim=2) + 1).prod(dim=1) - 1
                mean = portfolio_returns.mean()
                std = portfolio_returns.std(unbiased=False) + 1e-8
                ann_factor = torch.sqrt(
                    torch.tensor(252.0 / returns.shape[1], device=price_model_device, dtype=X.dtype))
                sharpe = mean / std * torch.sqrt(ann_factor)
                pbar.set_postfix({'loss': f'{batch_loss.item():.4f}', 'sharpe': f'{sharpe.item():.2f}'})

        phase_sharpe = logger.log_metrics(epoch=epoch, batch_idx=0, phase=phase_name,
                                          epoch_length=len(data_loader), is_epoch=True,
                                          losses=losses, **metrics)
        avg_loss = sum(list(losses.values())) / len(data_loader)
        logger.add_plot(metrics=metrics, epoch=epoch, phase=phase_name, tickers=tickers)

        return phase_sharpe, avg_loss

    # Run validation and test
    val_sharpe, avg_val_loss = evaluate_phase(price_val_loader, 'val')
    test_sharpe, avg_test_loss = evaluate_phase(price_test_loader, 'test')
    logger.export_epoch_summary()

    # Early stopping with optimized packing
    if epoch >= early_stopping.warmup_epochs:
        try:
            print("Packing model for early stopping check...")
            packed_model = pack_model_optimized(price_model, graph_model, init_static_emb,
                                                dynamic_emb, node_latest_event_time, val_sharpe)
            print("Model packed successfully")

            early_stopping(epoch=epoch, value=val_sharpe, model=packed_model, path=model_path)

        except Exception as e:
            print(f"Error during model packing: {e}")
            print("Attempting lightweight packing as fallback...")
            try:
                packed_model = pack_model_lightweight(price_model, graph_model, val_sharpe)
                early_stopping(epoch=epoch, value=val_sharpe, model=packed_model, path=model_path)
            except Exception as e2:
                print(f"Fallback packing also failed: {e2}")
                print("Continuing training without saving checkpoint...")

    return val_sharpe, avg_val_loss, test_sharpe, avg_test_loss


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    p = argparse.ArgumentParser(description='Combo Model Args')

    # Basic training arguments
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--scheduler_type', type=str, default='plateau')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--warmup_epochs', type=int, default=10)
    p.add_argument('--data_root', type=str, default='data')
    p.add_argument('--lookback', type=int, default=30)
    p.add_argument('--holding_period', type=int, default=5)
    p.add_argument('--N_stocks', type=int, default=512)
    p.add_argument('--ca_hidden_dim', type=int, default=64)

    # Loss function arguments
    p.add_argument('--min_weight', type=float, default=0.005)
    p.add_argument('--components', type=int, default=40)
    p.add_argument('--volatility', type=float, default=0.14)
    p.add_argument('--risk_aversion', type=float, default=5.0)
    p.add_argument('--utility', type=float, default=5.0)
    p.add_argument('--lc_utility', type=float, default=10.0)
    p.add_argument('--lc_components', type=float, default=5.0)
    p.add_argument('--lc_volatility', type=float, default=0.0)
    p.add_argument('--lc_min_weight', type=float, default=0.0)

    # Graph model arguments
    p.add_argument('--gm_in_dim', type=int, default=128)
    p.add_argument('--gm_structural_hid_dim', type=int, default=128)
    p.add_argument('--gm_temporal_hid_dim', type=int, default=128)
    p.add_argument('--gm_structural_RNN', type=str, default='RNN', choices=['RNN', 'GRU'])
    p.add_argument('--gm_temporal_RNN', type=str, default='RNN', choices=['RNN', 'GRU'])
    p.add_argument('--gm_num_gconv_layers', type=int, default=2)
    p.add_argument('--gm_rgcn_bdd_bases', type=int, default=16)
    p.add_argument('--gm_num_rnn_layers', type=int, default=2)
    p.add_argument('--gm_dropout', type=float, default=0.2)
    p.add_argument('--gm_activation', type=str, default='tanh', choices=['tanh', 'relu'])
    p.add_argument('--gm_decay_factor', type=float, default=0.8)
    p.add_argument('--gm_head_dropout', type=float, default=0.2)
    p.add_argument('--gm_out_dim', type=int, default=32)
    p.add_argument('--gm_time_interval_log_transform', action='store_true', default=True)

    # Price model arguments
    p.add_argument('--price_model', type=str, default='patch_tst')
    p.add_argument('--run_name', type=str, default='combo_PatchTST')
    p.add_argument('--d_model', type=int, default=64)
    p.add_argument('--n_heads', type=int, default=2)
    p.add_argument('--d_ff', type=int, default=128)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--e_layers', type=int, default=3)
    p.add_argument('--factor', type=int, default=1)
    p.add_argument('--enc_in', type=int, default=32)
    p.add_argument('--patch_len', type=int, default=8)
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--activation', type=str, default='gelu')

    # GPU and training control arguments
    p.add_argument('--graph_gpu', type=int, default=0)
    p.add_argument('--price_gpu', type=int, default=1)
    p.add_argument('--force_single_gpu', action='store_true')
    p.add_argument('--graph_lr_ratio', type=float, default=0.01)
    p.add_argument('--graph_warmup_epochs', type=int, default=20)
    p.add_argument('--max_graph_grad_ratio', type=float, default=0.1)

    # IMPORTANT: Pre-trained model arguments
    p.add_argument('--pretrained_price_model', type=str, default=None,
                   help='Path to pre-trained price model weights')
    p.add_argument('--training_stage', type=str, default='joint',
                   choices=['price_only', 'graph_only', 'joint'],
                   help='Training stage: price_only, graph_only, or joint')

    args = p.parse_args()

    # Setup devices
    if args.force_single_gpu or torch.cuda.device_count() < 2:
        print("Using single GPU for both models")
        graph_device = price_device = set_device()
    else:
        print(f"Using multi-GPU setup: Graph GPU {args.graph_gpu}, Price GPU {args.price_gpu}")
        graph_device = set_device(args.graph_gpu)
        price_device = set_device(args.price_gpu)

    set_seed(args.seed)

    # Create run name and directories
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.run_name}" if args.run_name else create_run_name(args)

    # Setup price model
    if args.price_model == 'patch_tst':
        price_config = dict(d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff,
                            dropout=args.dropout, e_layers=args.e_layers, factor=args.factor,
                            activation=args.activation, enc_in=args.enc_in, patch_len=args.patch_len,
                            stride=args.stride, output_logits=False)
        price_model = PatchTSTModel(**price_config)
        price_model_out_dim = args.enc_in
        model_name = 'Combo_PatchTST'
    else:
        raise NotImplementedError("Only PatchTST supported in this version")

    log_dir, model_path = setup_directories(model_name, run_name)
    print(f"Run name: {run_name}")
    print(f"Log directory: {log_dir}")

    # Load data
    root_dir = Path(__file__).resolve().parent / args.data_root
    price_train_loader, price_val_loader, price_test_loader = get_dataloaders(
        batch_size=args.batch_size, root_dir=root_dir / 'market_data',
        lookback=args.lookback, prediction_horizon=args.holding_period)
    tickers = get_tickers(root_dir=root_dir / 'market_data')

    loss_fn = LongOnlyMarkowitzPortfolioLoss(target_volatility=args.volatility,
                                             target_cardinality=args.components,
                                             target_min_weight=args.min_weight,
                                             target_min_utility=args.utility,
                                             risk_aversion=args.risk_aversion)

    # Load graph
    G = load_temporal_knowledge_graph(root_dir=root_dir / 'triplets')
    graph_statics = TemporalKnowledgeGraphStatics(root_dir=root_dir / 'triplets')

    # Create graph model components
    graph_model_args = dict(num_nodes=G.number_of_nodes(), num_rels=G.num_relations,
                            in_dim=args.gm_in_dim, structural_hid_dim=args.gm_structural_hid_dim,
                            temporal_hid_dim=args.gm_temporal_hid_dim,
                            structural_RNN=args.gm_structural_RNN, temporal_RNN=args.gm_temporal_RNN,
                            num_gconv_layers=args.gm_num_gconv_layers, rgcn_bdd_bases=args.gm_rgcn_bdd_bases,
                            num_rnn_layers=args.gm_num_rnn_layers, dropout=args.gm_dropout,
                            activation=args.gm_activation, decay_factor=args.gm_decay_factor,
                            head_dropout=args.gm_head_dropout, out_dim=args.gm_out_dim,
                            time_interval_log_transform=args.gm_time_interval_log_transform)

    node_latest_event_time = torch.zeros((G.number_of_nodes(), G.number_of_nodes() + 1, 2),
                                         dtype=torch.float32, device=graph_device)
    static_emb, init_dynamic_emb = initialise_embeddings(num_nodes=G.number_of_nodes(),
                                                         embedding_dim=args.gm_in_dim,
                                                         num_rnn_layers=args.gm_num_rnn_layers,
                                                         device=graph_device)

    # Tensor size analysis
    print("\n" + "=" * 60)
    print("TENSOR SIZE ANALYSIS")
    print("=" * 60)
    tensor_info = check_tensor_sizes(node_latest_event_time, static_emb, init_dynamic_emb)
    if not tensor_info['sparse_safe']:
        print("\n⚠️  WARNING: Large tensors detected!")
    else:
        print("\n✅ All tensors are within safe limits")
    print("=" * 60)

    # Create models
    graph_model = GraphModel(node_latest_event_time=node_latest_event_time, device=graph_device, **graph_model_args)
    cross_attention_head = CrossAttentionHead(query_dim=price_model_out_dim, node_dim=args.gm_out_dim,
                                              hidden_dim=args.ca_hidden_dim, num_nodes=G.number_of_nodes(),
                                              return_attention=False)

    # Move to devices
    cross_attention_head.to(price_device)
    graph_model.to(graph_device)
    price_model.to(price_device)

    # IMPORTANT: Load pre-trained price model if provided
    price_pretrained = False
    if args.pretrained_price_model:
        price_pretrained = load_pretrained_price_model(price_model, args.pretrained_price_model)

    # Setup optimizer based on training stage
    if args.training_stage != 'joint':
        optimizer = setup_staged_optimizer(price_model, graph_model, cross_attention_head,
                                           static_emb, init_dynamic_emb, args, stage=args.training_stage)
    else:
        optimizer = setup_differentiated_optimizer(price_model, graph_model, cross_attention_head,
                                                   static_emb, init_dynamic_emb, args)

    scheduler = get_scheduler(optimizer, args.scheduler_type)
    early_stopping = EarlyStopping(patience=args.patience, mode='max', warmup_epochs=args.warmup_epochs)
    logger = PortfolioTensorBoardLogger(log_dir=str(log_dir))

    # Save config
    config = {
        'run_name': run_name, 'model': model_name, 'training_stage': args.training_stage,
        'pretrained_price_model': args.pretrained_price_model, 'price_pretrained': price_pretrained,
        'graph_lr_ratio': args.graph_lr_ratio, 'graph_warmup_epochs': args.graph_warmup_epochs,
        'max_graph_grad_ratio': args.max_graph_grad_ratio, 'epochs': args.epochs, 'lr': args.lr,
        'batch_size': args.batch_size, 'seed': args.seed
    }

    with open(log_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Training stage: {args.training_stage}")
    print(f"Pre-trained price model: {args.pretrained_price_model}")
    print(f"Price model loaded: {price_pretrained}")

    # Training
    loss_coefficients = dict(utility_target_penalty=args.lc_utility, cardinality_target_penalty=args.lc_components,
                             vol_target_penalty=args.lc_volatility, min_weight_penalty=args.lc_min_weight)

    if price_device.type == 'cuda':
        scaler = torch.amp.GradScaler()
        run_cuda_multi_gpu(
            cross_attention_head=cross_attention_head,
            price_model=price_model,
            graph_model=graph_model,
            price_model_device=price_device,
            graph_model_device=graph_device,
            price_train_loader=price_train_loader,
            price_val_loader=price_val_loader,
            price_test_loader=price_test_loader,
            G=G,
            graph_statics=graph_statics,
            init_static_emb=static_emb,
            init_dynamic_emb=init_dynamic_emb,
            node_latest_event_time=node_latest_event_time,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            logger=logger,
            epochs=args.epochs,
            model_path=model_path,
            early_stopping=early_stopping,
            loss_coefficients=loss_coefficients,
            scaler=scaler,
            tickers=tickers,
            graph_warmup_epochs=args.graph_warmup_epochs,
            max_graph_grad_ratio=args.max_graph_grad_ratio
        )
    else:
        raise NotImplementedError("Only CUDA training is supported")

    logger.writer.close()
    print(f"Training complete. Results saved to {log_dir}")


if __name__ == '__main__':
    main()