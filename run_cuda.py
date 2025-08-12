import argparse
import json
import os
import sys
import dgl
import numpy as np

from typing import Optional, Dict, Tuple
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from pprint import pp

# Add parent directory to Python path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import torch
import torch.optim as optim
from entmax import entmax15
from tqdm import tqdm

from price_models.dataloader import get_dataloaders, get_tickers
from price_models.patch_tst import PatchTSTModel
from price_models.bidirectional_lstm import BidirectionalLSTM

from graph_models.dataloader import (load_temporal_knowledge_graph, collate_fn as graph_collate_fn,
                                     TemporalKnowledgeGraphStatics)
from graph_models.news_graph import GraphModel, initialise_embeddings, MultiAspectEmbedding

from cross_attention import CrossAttentionHead
from losses import LongOnlyMarkowitzPortfolioLoss
from utils import EarlyStopping, PortfolioTensorBoardLogger, get_scheduler, set_device, set_seed

import pickle
import gzip
from pathlib import Path


def pack_model_optimized(price_model, graph_model, static_entity_emb,
                         val_dynamic_entity_emb, node_latest_event_time, val_sharpe):
    """
    Optimized model packing for very large tensors with compression and chunking
    """

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

        # Option 1: Store only non-zero entries (if tensor is sparse)
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
        # Model state dicts
        'price_model_state_dict': price_model.state_dict(),
        'graph_model_state_dict': graph_model.state_dict(),

        # Embeddings (moved to CPU)
        'static_entity_emb': static_emb_cpu,
        'dynamic_entity_emb': dynamic_emb_cpu,

        # Special handling for large tensor
        'node_latest_event_time': node_latest_event_time_data,

        # Metadata
        'sharpe': val_sharpe,
        'price_device': str(price_device),
        'graph_device': str(graph_device),

        # Model architecture info
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


def load_model_optimized(model_path, price_model, graph_model, price_device, graph_device):
    """
    Load model with optimized handling for large tensors
    """
    print(f"Loading model from {model_path}")

    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

    # Load model state dicts
    price_model.load_state_dict(checkpoint['price_model_state_dict'])
    graph_model.load_state_dict(checkpoint['graph_model_state_dict'])

    # Move models to their respective devices
    price_model.to(price_device)
    graph_model.to(graph_device)

    # Reconstruct embeddings on appropriate devices
    static_emb = MultiAspectEmbedding(
        structural=checkpoint['static_entity_emb'].structural.to(graph_device),
        temporal=checkpoint['static_entity_emb'].temporal.to(graph_device)
    )

    dynamic_emb = MultiAspectEmbedding(
        structural=checkpoint['dynamic_entity_emb'].structural.to(graph_device),
        temporal=checkpoint['dynamic_entity_emb'].temporal.to(graph_device)
    )

    # Handle node_latest_event_time based on storage format
    node_data = checkpoint['node_latest_event_time']

    if node_data['format'] == 'coordinate':
        print("Reconstructing tensor from coordinate format...")
        # Reconstruct from coordinates
        shape = node_data['shape']
        indices = node_data['indices']
        values = node_data['values']

        # Create zero tensor and fill in non-zero values
        node_latest_event_time = torch.zeros(shape, dtype=node_data['dtype'])
        node_latest_event_time[indices[:, 0], indices[:, 1], indices[:, 2]] = values

    elif node_data['format'] == 'compressed_dense':
        print("Decompressing dense tensor...")
        # Decompress the tensor
        compressed_data = node_data['compressed_data']
        decompressed_data = gzip.decompress(compressed_data)

        import io
        buffer = io.BytesIO(decompressed_data)
        node_latest_event_time = torch.load(buffer)

    else:  # 'dense'
        node_latest_event_time = node_data['tensor']

    # Move to graph device
    print("Moving large tensor to GPU...")
    node_latest_event_time = node_latest_event_time.to(graph_device)

    print(f"Model loaded successfully. Sharpe: {checkpoint['sharpe']:.4f}")

    return {
        'price_model': price_model,
        'graph_model': graph_model,
        'static_emb': static_emb,
        'dynamic_emb': dynamic_emb,
        'node_latest_event_time': node_latest_event_time,
        'sharpe': checkpoint['sharpe'],
        'model_info': checkpoint['model_info']
    }


# Alternative: Ultra-lightweight saving (only model weights)
def pack_model_lightweight(price_model, graph_model, val_sharpe):
    """
    Ultra-lightweight model packing - only saves model weights, not the large tensors
    Use this if you can recreate the embeddings and tensors from scratch
    """

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
    print("You will need to reinitialize embeddings when loading")

    return packed_model


def check_tensor_sizes(node_latest_event_time, static_emb, dynamic_emb):
    """
    Check tensor sizes to predict potential issues before saving
    """
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


def optimize_node_latest_event_time_size(num_nodes):
    """
    Calculate the size of node_latest_event_time tensor and suggest optimizations
    """
    # Current shape is (num_nodes, num_nodes + 1, 2)
    current_size = num_nodes * (num_nodes + 1) * 2
    current_mb = (current_size * 4) / (1024 ** 2)

    print(f"Current node_latest_event_time size:")
    print(f"  Shape: ({num_nodes}, {num_nodes + 1}, 2)")
    print(f"  Elements: {current_size:,}")
    print(f"  Memory: {current_mb:.1f} MB")
    print(f"  Sparse safe: {current_size <= 2 ** 31 - 1}")

    if current_size > 2 ** 31 - 1:
        print("\nSuggested optimizations:")
        print("1. Use a more memory-efficient data structure")
        print("2. Store only non-zero entries in a dictionary format")
        print("3. Use compression techniques")
        print("4. Consider chunked saving/loading")

    return current_size

def set_device(gpu_id=None):
    """Set device with optional GPU selection"""
    if torch.cuda.is_available():
        if gpu_id is not None:
            device = torch.device(f'cuda:{gpu_id}')
            torch.cuda.set_device(device)
        else:
            device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        print(f"Using CUDA backend on {device}")
        return device
    return torch.device('cpu')


def setup_multi_gpu_devices():
    """Setup devices for multi-GPU training"""
    if torch.cuda.device_count() < 2:
        print("Warning: Less than 2 GPUs available, using same device for both models")
        device = set_device()
        return device, device

    # Use GPU 0 for graph model, GPU 1 for price model
    graph_device = set_device(0)
    price_device = set_device(1)

    print(f"Graph model device: {graph_device}")
    print(f"Price model device: {price_device}")

    return graph_device, price_device


def transfer_embeddings_to_price_device(embeddings, price_device):
    """Efficiently transfer embeddings from graph GPU to price GPU"""
    if isinstance(embeddings, list):
        return [emb.to(price_device, non_blocking=True) for emb in embeddings]
    elif isinstance(embeddings, torch.Tensor):
        return embeddings.to(price_device, non_blocking=True)
    else:
        # Handle MultiAspectEmbedding or similar
        return type(embeddings)(
            structural=embeddings.structural.to(price_device, non_blocking=True),
            temporal=embeddings.temporal.to(price_device, non_blocking=True)
        )


def pack_model(price_model, graph_model, static_entity_emb,
               val_dynamic_entity_emb, node_latest_event_time, val_sharpe):
    """
    Pack model state for saving, handling multi-GPU and large tensors properly
    """

    # Get devices for proper saving
    price_device = next(price_model.parameters()).device
    graph_device = next(graph_model.parameters()).device

    print(f"Packing model - Price device: {price_device}, Graph device: {graph_device}")

    # Move everything to CPU for saving to avoid device issues
    # and handle the large tensor issue
    try:
        # Check tensor size before attempting to_sparse
        tensor_size = node_latest_event_time.numel()
        print(f"node_latest_event_time tensor size: {tensor_size:,} elements")

        if tensor_size > 2 ** 31 - 1:  # INT_MAX
            print("Tensor too large for sparse conversion, saving as dense tensor on CPU")
            node_latest_event_time_cpu = node_latest_event_time.cpu()
        else:
            print("Converting to sparse format")
            node_latest_event_time_cpu = node_latest_event_time.to_sparse().cpu()

    except Exception as e:
        print(f"Error with sparse conversion: {e}")
        print("Falling back to dense tensor on CPU")
        node_latest_event_time_cpu = node_latest_event_time.cpu()

    # Move embeddings to CPU for saving
    static_emb_cpu = MultiAspectEmbedding(
        structural=static_entity_emb.structural.cpu(),
        temporal=static_entity_emb.temporal.cpu()
    )

    dynamic_emb_cpu = MultiAspectEmbedding(
        structural=val_dynamic_entity_emb.structural.cpu(),
        temporal=val_dynamic_entity_emb.temporal.cpu()
    )

    # Create the packed model dictionary
    packed_model = {
        # Model state dicts (these handle device conversion automatically)
        'price_model_state_dict': price_model.state_dict(),
        'graph_model_state_dict': graph_model.state_dict(),

        # Embeddings (moved to CPU)
        'static_entity_emb': static_emb_cpu,
        'dynamic_entity_emb': dynamic_emb_cpu,

        # Event time tensor (handled above)
        'node_latest_event_time': node_latest_event_time_cpu,

        # Metadata
        'sharpe': val_sharpe,
        'price_device': str(price_device),
        'graph_device': str(graph_device),

        # Model architecture info (for reconstruction)
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
            'node_latest_event_time_shape': node_latest_event_time.shape,
            'tensor_is_sparse': tensor_size <= 2 ** 31 - 1
        }
    }

    print(f"Model packed successfully. Total items: {len(packed_model)}")
    return packed_model


def load_model(model_path, price_model, graph_model, price_device, graph_device):
    """
    Load model from saved state, handling multi-GPU setup
    """
    print(f"Loading model from {model_path}")

    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')  # Load to CPU first

    # Load model state dicts
    price_model.load_state_dict(checkpoint['price_model_state_dict'])
    graph_model.load_state_dict(checkpoint['graph_model_state_dict'])

    # Move models to their respective devices
    price_model.to(price_device)
    graph_model.to(graph_device)

    # Reconstruct embeddings on appropriate devices
    static_emb = MultiAspectEmbedding(
        structural=checkpoint['static_entity_emb'].structural.to(graph_device),
        temporal=checkpoint['static_entity_emb'].temporal.to(graph_device)
    )

    dynamic_emb = MultiAspectEmbedding(
        structural=checkpoint['dynamic_entity_emb'].structural.to(graph_device),
        temporal=checkpoint['dynamic_entity_emb'].temporal.to(graph_device)
    )

    # Handle node_latest_event_time
    node_latest_event_time = checkpoint['node_latest_event_time']
    if checkpoint['model_info']['tensor_is_sparse']:
        node_latest_event_time = node_latest_event_time.to_dense()
    node_latest_event_time = node_latest_event_time.to(graph_device)

    print(f"Model loaded successfully. Sharpe: {checkpoint['sharpe']:.4f}")

    return {
        'price_model': price_model,
        'graph_model': graph_model,
        'static_emb': static_emb,
        'dynamic_emb': dynamic_emb,
        'node_latest_event_time': node_latest_event_time,
        'sharpe': checkpoint['sharpe'],
        'model_info': checkpoint['model_info']
    }


def save_checkpoint(epoch, model_dict, optimizer, scheduler, model_path):
    """
    Enhanced checkpoint saving with optimizer and scheduler states
    """
    checkpoint = {
        **model_dict,  # Include the packed model
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }

    torch.save(checkpoint, model_path)
    print(f"Checkpoint saved to {model_path}")


def load_checkpoint(model_path, price_model, graph_model, optimizer, scheduler,
                    price_device, graph_device):
    """
    Load full checkpoint including optimizer and scheduler states
    """
    checkpoint = torch.load(model_path, map_location='cpu')

    # Load models
    model_data = load_model(model_path, price_model, graph_model, price_device, graph_device)

    # Load optimizer and scheduler states
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return {
        **model_data,
        'epoch': checkpoint['epoch'],
        'optimizer': optimizer,
        'scheduler': scheduler
    }


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


def run_cuda_multi_gpu_optimized(price_model: torch.nn.Module,
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
                                 **kwargs):
    activation = entmax15

    # Create CUDA streams for parallel execution
    graph_stream = torch.cuda.Stream(device=graph_model_device)
    price_stream = torch.cuda.Stream(device=price_model_device)
    transfer_stream = torch.cuda.Stream(device=price_model_device)

    for epoch in range(1, epochs + 1):
        print(f"\n{'=' * 50}\nEpoch {epoch}/{epochs}\n{'=' * 50}")
        print(f"Graph GPU: {graph_model_device}, Price GPU: {price_model_device}")

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

        # Process initial dates on graph GPU
        prediction_date = next(iter(price_train_loader))['prediction_dates'][0]
        start_idx = date2idx[prediction_date]

        print(f"Processing initial {start_idx} graph dates on {graph_model_device}")
        with torch.cuda.stream(graph_stream):
            for i in range(start_idx):
                batch_G, date = graph_collate_fn([(i, idx2date[i]), ], G)
                if batch_G is None:
                    continue
                dynamic_emb = graph_model(batch_G=batch_G, static_entity_emb=init_static_emb,
                                          dynamic_entity_emb=dynamic_emb)

        # Synchronize to ensure initial processing is complete
        torch.cuda.synchronize()

        pbar = tqdm(price_train_loader, desc=f"Training Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            embeddings = []
            optimizer.zero_grad(set_to_none=True)

            X, y, dates, prediction_dates = batch['X'], batch['y'], batch['dates'], batch['prediction_dates']
            combined = None
            end_idx = date2idx[prediction_dates[-1]] + 1

            # Step 1: Process graph embeddings on graph GPU
            with torch.cuda.stream(graph_stream):
                torch.cuda.set_device(graph_model_device)
                for i in range(start_idx, end_idx):
                    batch_G, date = graph_collate_fn([(i, idx2date[i]), ], G)
                    if batch_G is None:
                        if date in prediction_dates:
                            if combined is None:
                                combined = graph_model.combine(dynamic_entity_emb=dynamic_emb)
                            embeddings.append(combined.clone())  # Clone to avoid device issues
                        continue
                    dynamic_emb = graph_model(batch_G=batch_G, static_entity_emb=init_static_emb,
                                              dynamic_entity_emb=dynamic_emb)
                    if date in prediction_dates:
                        combined = graph_model.combine(dynamic_entity_emb=dynamic_emb)
                        embeddings.append(combined.clone())  # Clone to avoid device issues

            # Wait for graph processing to complete
            graph_stream.synchronize()

            start_idx = end_idx
            B, seq_len, stocks, features = X.shape
            assert len(embeddings) == B, f"Expected {B} embeddings, got {len(embeddings)}"

            # Step 2: Transfer embeddings to price GPU asynchronously
            with torch.cuda.stream(transfer_stream):
                torch.cuda.set_device(price_model_device)
                embeddings_tensor = torch.cat(embeddings, dim=0).reshape(B, *embeddings[0].shape)
                embeddings_on_price_gpu = embeddings_tensor.to(price_model_device, non_blocking=True)
                # Also transfer price data
                X_gpu = X.to(price_model_device, non_blocking=True)
                returns_gpu = y.to(price_model_device, non_blocking=True)

            # Step 3: Process price model on price GPU
            with torch.cuda.stream(price_stream):
                torch.cuda.set_device(price_model_device)

                # Wait for transfers to complete
                transfer_stream.synchronize()

                batch_loss = torch.tensor(0.0, dtype=X_gpu.dtype, device=price_model_device)

                # Prepare data for model
                x = X_gpu.permute(0, 2, 1, 3).contiguous()
                x = x.reshape(-1, x.shape[-2], x.shape[-1])

                # Forward pass with mixed precision
                with torch.autocast(price_model_device.type):
                    weights = price_model(x)
                    weights = weights.reshape(B, stocks, weights.shape[-1])
                    weights = cross_attention_head(weights, embeddings_on_price_gpu, None)
                    weights = activation(weights, dim=-1)

                    # Compute loss
                    loss_dict = loss_fn(weights, returns_gpu)
                    for loss_name, coefficient in loss_coefficients.items():
                        batch_loss += (loss_dict[loss_name] * coefficient).mean()
                        running_losses[loss_name] += (loss_dict[loss_name] * coefficient).mean()
                        all_losses[loss_name] += (loss_dict[loss_name] * coefficient).mean()

                # Backward pass
                if scaler is not None:
                    scaler.scale(batch_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    batch_loss.backward()
                    optimizer.step()

                # Store metrics
                running_metrics['weights'].append(weights)
                running_metrics['raw_returns'].append(returns_gpu)
                all_metrics['weights'].append(weights)
                all_metrics['raw_returns'].append(returns_gpu)
                all_metrics['dates'].extend(dates)

            # Synchronize price stream before continuing
            price_stream.synchronize()

            # Detach embeddings to prevent memory accumulation
            dynamic_emb = MultiAspectEmbedding(
                structural=dynamic_emb.structural.detach(),
                temporal=dynamic_emb.temporal.detach()
            )

            # Clear cache periodically
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()

            ann_factor = torch.sqrt(
                torch.tensor(252.0 / returns_gpu.shape[1], device=price_model_device, dtype=X_gpu.dtype))

            if (batch_idx + 1) % 5 == 0:
                logger.log_metrics(epoch=epoch, batch_idx=batch_idx, is_epoch=False, epoch_length=epoch_length,
                                   phase='train', losses=running_losses, **running_metrics)
                running_metrics = defaultdict(list)
                running_losses = defaultdict(float)

            # Update progress bar
            portfolio_returns = ((returns_gpu * weights.unsqueeze(dim=1)).sum(dim=2) + 1).prod(dim=1) - 1
            mean = portfolio_returns.mean()
            std = portfolio_returns.std(unbiased=False) + 1e-8
            sharpe = mean / std * torch.sqrt(ann_factor)
            pbar.set_postfix({
                'loss': f'{batch_loss.item():.4f}',
                'sharpe': f'{sharpe.item():.2f}',
                'graph_gpu': f'{torch.cuda.memory_allocated(graph_model_device) / 1e9:.1f}GB',
                'price_gpu': f'{torch.cuda.memory_allocated(price_model_device) / 1e9:.1f}GB'
            })

        # Log epoch metrics
        train_sharpe = logger.log_metrics(epoch=epoch, batch_idx=0, phase='train', epoch_length=epoch_length,
                                          is_epoch=True, losses=all_losses, **all_metrics)
        avg_train_loss = sum(list(all_losses.values())) / (batch_idx + 1)
        logger.add_plot(metrics=all_metrics, epoch=epoch, phase='train', tickers=tickers)
        torch.cuda.empty_cache()

        # ============== Validation ==============
        def evaluate_phase_multi_gpu(data_loader, phase_name):
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

                    # Process graph on graph GPU with stream
                    with torch.cuda.stream(graph_stream):
                        torch.cuda.set_device(graph_model_device)
                        for i in range(phase_start_idx, end_idx):
                            batch_G, date = graph_collate_fn([(i, idx2date[i]), ], G)
                            if batch_G is None:
                                if date in prediction_dates:
                                    if combined is None:
                                        combined = graph_model.combine(dynamic_entity_emb=dynamic_emb)
                                    embeddings.append(combined.clone())
                                continue
                            dynamic_emb = graph_model(batch_G=batch_G, static_entity_emb=init_static_emb,
                                                      dynamic_entity_emb=dynamic_emb)
                            if date in prediction_dates:
                                combined = graph_model.combine(dynamic_entity_emb=dynamic_emb)
                                embeddings.append(combined.clone())

                    # Wait for graph processing
                    graph_stream.synchronize()
                    phase_start_idx = end_idx

                    if len(X.shape) == 3:
                        X = X.unsqueeze(0)

                    B, seq_len, stocks, features = X.shape
                    assert len(embeddings) == B, f"Expected {B} embeddings, got {len(embeddings)}"

                    # Process on price GPU with stream
                    with torch.cuda.stream(price_stream):
                        torch.cuda.set_device(price_model_device)

                        # Transfer data
                        embeddings_tensor = torch.cat(embeddings, dim=0).reshape(B, *embeddings[0].shape)
                        embeddings_gpu = embeddings_tensor.to(price_model_device, non_blocking=True)
                        X_gpu = X.to(price_model_device, non_blocking=True)
                        returns_gpu = y.to(price_model_device, non_blocking=True)

                        batch_loss = torch.tensor(0.0, dtype=X_gpu.dtype, device=price_model_device)

                        # Prepare data
                        x = X_gpu.permute(0, 2, 1, 3).contiguous()
                        x = x.reshape(-1, x.shape[-2], x.shape[-1])

                        # Forward pass
                        with torch.autocast(device_type=price_model_device.type):
                            weights = price_model(x)
                            weights = weights.reshape(B, stocks, weights.shape[-1])
                            weights = cross_attention_head(weights, embeddings_gpu, None)
                            weights = activation(weights, dim=-1)

                            loss_dict = loss_fn(weights, returns_gpu)
                            for loss_name, coefficient in loss_coefficients.items():
                                losses[loss_name] += (loss_dict[loss_name] * coefficient).mean().item()
                                batch_loss += (loss_dict[loss_name] * coefficient).mean()

                        metrics['raw_returns'].append(returns_gpu)
                        metrics['weights'].append(weights)
                        metrics['dates'].extend(dates)

                        # Update progress
                        portfolio_returns = ((returns_gpu * weights.unsqueeze(dim=1)).sum(dim=2) + 1).prod(dim=1) - 1
                        mean = portfolio_returns.mean()
                        std = portfolio_returns.std(unbiased=False) + 1e-8
                        ann_factor = torch.sqrt(torch.tensor(252.0 / returns_gpu.shape[1],
                                                             device=price_model_device, dtype=X_gpu.dtype))
                        sharpe = mean / std * torch.sqrt(ann_factor)
                        pbar.set_postfix({'loss': f'{batch_loss.item():.4f}', 'sharpe': f'{sharpe.item():.2f}'})

                    # Wait for price processing to complete
                    price_stream.synchronize()

            phase_sharpe = logger.log_metrics(epoch=epoch, batch_idx=0, phase=phase_name,
                                              epoch_length=len(data_loader), is_epoch=True,
                                              losses=losses, **metrics)
            avg_loss = sum(list(losses.values())) / len(data_loader)
            logger.add_plot(metrics=metrics, epoch=epoch, phase=phase_name, tickers=tickers)

            return phase_sharpe, avg_loss

        # Run validation and test
        val_sharpe, avg_val_loss = evaluate_phase_multi_gpu(price_val_loader, 'val')
        test_sharpe, avg_test_loss = evaluate_phase_multi_gpu(price_test_loader, 'test')

        logger.export_epoch_summary()

        # Learning rate scheduling
        if scheduler:
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            logger.writer.add_scalar('training/learning_rate', current_lr, epoch)

        # Print epoch summary with GPU memory usage
        print(f"Train Loss: {avg_train_loss:.6f}, Train Sharpe: {train_sharpe:.4f}")
        print(f"Val Loss: {avg_val_loss:.6f}, Val Sharpe: {val_sharpe:.4f}")
        print(f"Test Loss: {avg_test_loss:.6f}, Test Sharpe: {test_sharpe:.4f}")
        print(f"Graph GPU Memory: {torch.cuda.memory_allocated(graph_model_device) / 1e9:.1f}GB")
        print(f"Price GPU Memory: {torch.cuda.memory_allocated(price_model_device) / 1e9:.1f}GB")
        logger.writer.flush()
        # In your training loop, replace the early stopping section with this:

        if epoch >= early_stopping.warmup_epochs:
            try:
                print("Packing model for early stopping check...")

                # Choose packing method based on your needs:

                # Option 1: Full optimized packing (recommended)
                packed_model = pack_model_optimized(price_model, graph_model, init_static_emb,
                                                    dynamic_emb, node_latest_event_time, val_sharpe)

                # Option 2: Lightweight packing (if you can recreate embeddings)
                # packed_model = pack_model_lightweight(price_model, graph_model, val_sharpe)

                print("Model packed successfully")

                early_stopping(epoch=epoch, value=val_sharpe, model=packed_model, path=model_path)

                if early_stopping.early_stop:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

            except Exception as e:
                print(f"Error during model packing: {e}")
                print("Attempting lightweight packing as fallback...")
                try:
                    # Fallback to lightweight packing
                    packed_model = pack_model_lightweight(price_model, graph_model, val_sharpe)
                    early_stopping(epoch=epoch, value=val_sharpe, model=packed_model, path=model_path)
                    if early_stopping.early_stop:
                        print(f"Early stopping triggered at epoch {epoch}")
                        break
                except Exception as e2:
                    print(f"Fallback packing also failed: {e2}")
                    print("Continuing training without saving checkpoint...")

        torch.cuda.empty_cache()


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
                       **kwargs):
    activation = entmax15

    for epoch in range(1, epochs + 1):
        print(f"\n{'=' * 50}\nEpoch {epoch}/{epochs}\n{'=' * 50}")

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

        pbar = tqdm(price_train_loader, desc=f"Training Epoch {epoch}")

        date2idx = graph_statics.date2idx
        idx2date = graph_statics.idx2date

        epoch_length = len(price_train_loader)

        # graph: process previous dates prior to the first prediction date
        prediction_date = next(iter(price_train_loader))['prediction_dates'][0]
        start_idx = date2idx[prediction_date]

        for i in range(start_idx):  # this won't do the prediction date itself which is what we want
            batch_G, date = graph_collate_fn([(i, idx2date[i]), ], G)
            if batch_G is None:
                continue
            dynamic_emb = graph_model(batch_G=batch_G, static_entity_emb=init_static_emb,
                                      dynamic_entity_emb=dynamic_emb)

        for batch_idx, batch in enumerate(pbar):
            embeddings = []
            optimizer.zero_grad(set_to_none=True)

            X, y, dates, prediction_dates = batch['X'], batch['y'], batch['dates'], batch['prediction_dates']
            combined = None
            end_idx = date2idx[prediction_dates[-1]] + 1  # +1 to include the last date

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

            # CRITICAL: Transfer embeddings from graph GPU to price GPU
            embeddings_tensor = torch.cat(embeddings, dim=0).reshape(B, *embeddings[0].shape)
            embeddings_on_price_gpu = embeddings_tensor.to(price_model_device, non_blocking=True)

            X, returns = X.to(price_model_device, non_blocking=True), y.to(price_model_device, non_blocking=True)
            batch_loss = torch.tensor(0.0, dtype=X.dtype).to(price_model_device, non_blocking=True)

            # Prepare data for model (reshape as in original)
            x = X.permute(0, 2, 1, 3).contiguous()  # [B, seq_len, stocks, features] -> [B, stocks, seq_len, features]
            x = x.reshape(-1, x.shape[-2], x.shape[-1])  # [B*stocks, seq_len, features]

            # Forward pass on price GPU
            with torch.autocast(price_model_device.type):
                # Forward pass
                weights = price_model(x)  # [B*stocks, 1] or similar
                weights = weights.reshape(B, stocks, weights.shape[-1])
                weights = cross_attention_head(weights, embeddings_on_price_gpu,
                                               None)  # [B, stocks, features] -> [B, stocks]
                weights = activation(weights, dim=-1)

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

            # Clear GPU cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

            # IMPORTANT: Detach embeddings to prevent memory accumulation
            dynamic_emb = MultiAspectEmbedding(
                structural=dynamic_emb.structural.detach(),
                temporal=dynamic_emb.temporal.detach()
            )

            running_metrics['weights'].append(weights)
            running_metrics['raw_returns'].append(returns)
            all_metrics['weights'].append(weights)
            all_metrics['raw_returns'].append(returns)
            all_metrics['dates'].extend(dates)

            ann_factor = torch.sqrt(torch.tensor(252.0 / returns.shape[1], device=price_model_device, dtype=X.dtype))

            if (batch_idx + 1) % 5 == 0:
                # Log batch metrics (simple averages for monitoring)
                logger.log_metrics(epoch=epoch, batch_idx=batch_idx, is_epoch=False, epoch_length=epoch_length,
                                   phase='train', losses=running_losses, **running_metrics, )
                running_metrics = defaultdict(list)
                running_losses = defaultdict(float)
            # Update progress bar
            portfolio_returns = ((returns * weights.unsqueeze(dim=1)).sum(dim=2) + 1).prod(
                dim=1) - 1  # per holding period
            mean = portfolio_returns.mean()
            std = portfolio_returns.std(unbiased=False) + 1e-8  # Avoid division by zero

            # Annualized Sharpe
            sharpe = mean / std * torch.sqrt(ann_factor)
            pbar.set_postfix({'loss': f'{batch_loss.item():.4f}', 'sharpe': f'{sharpe.item():.2f}'})

        train_sharpe = logger.log_metrics(epoch=epoch, batch_idx=0, phase='train', epoch_length=epoch_length,
                                          is_epoch=True, losses=all_losses, **all_metrics)
        avg_train_loss = sum(list(all_losses.values())) / (batch_idx + 1)
        logger.add_plot(metrics=all_metrics, epoch=epoch, phase='train', tickers=tickers)
        end_date = idx2date[start_idx - 1]
        assert end_date == prediction_dates[-1], f"Expected last date {prediction_dates[-1]}, got {end_date}"
        torch.cuda.empty_cache()

        # ============== Evaluation on Validation Data ==============
        metrics = defaultdict(list)
        losses = defaultdict(float)

        epoch_length = len(price_val_loader)

        pbar = tqdm(price_val_loader, desc=f"Evaluating Validation Set")
        with torch.inference_mode():
            for batch_idx, batch in enumerate(pbar):
                embeddings = []
                X, y, dates, prediction_dates = batch['X'], batch['y'], batch['dates'], batch['prediction_dates']
                combined = None
                end_idx = date2idx[prediction_dates[-1]] + 1  # +1 to include the last date

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

                if len(X.shape) == 3:
                    X = X.unsqueeze(0)  # Add a batch dimension if it's missing

                B, seq_len, stocks, features = X.shape
                assert len(embeddings) == B, f"Expected {B} embeddings, got {len(embeddings)}"

                # Transfer embeddings from graph GPU to price GPU
                embeddings_tensor = torch.cat(embeddings, dim=0).reshape(B, *embeddings[0].shape)
                embeddings_on_price_gpu = embeddings_tensor.to(price_model_device, non_blocking=True)

                X, returns = X.to(price_model_device, non_blocking=True), y.to(price_model_device, non_blocking=True)
                batch_loss = torch.tensor(0.0, dtype=X.dtype).to(price_model_device, non_blocking=True)

                # Prepare data for model (reshape as in original)
                x = X.permute(0, 2, 1,
                              3).contiguous()  # [B, seq_len, stocks, features] -> [B, stocks, seq_len, features]
                x = x.reshape(-1, x.shape[-2], x.shape[-1])  # [B*stocks, seq_len, features]

                # Forward pass
                with torch.autocast(device_type=price_model_device.type):
                    weights = price_model(x)  # [B*stocks, 1] or similar
                    weights = weights.reshape(B, stocks, weights.shape[-1])
                    weights = cross_attention_head(weights, embeddings_on_price_gpu,
                                                   None)  # [B, stocks, features] -> [B, stocks]
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

        val_sharpe = logger.log_metrics(epoch=epoch, batch_idx=0, phase='val', epoch_length=epoch_length,
                                        is_epoch=True, losses=losses, **metrics)
        avg_val_loss = sum(list(losses.values())) / (batch_idx + 1)
        logger.add_plot(metrics=metrics, epoch=epoch, phase='val', tickers=tickers)

        # ============== Evaluation on Test Data ==============
        metrics = defaultdict(list)
        losses = defaultdict(float)
        epoch_length = len(price_test_loader)

        pbar = tqdm(price_test_loader, desc=f"Evaluating Test Set")

        with torch.inference_mode():
            for batch_idx, batch in enumerate(pbar):
                embeddings = []
                X, y, dates, prediction_dates = batch['X'], batch['y'], batch['dates'], batch['prediction_dates']
                combined = None
                end_idx = date2idx[prediction_dates[-1]] + 1  # +1 to include the last date

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
                if len(X.shape) == 3:
                    X = X.unsqueeze(0)  # Add a batch dimension if it's missing

                B, seq_len, stocks, features = X.shape
                assert len(embeddings) == B, f"Expected {B} embeddings, got {len(embeddings)}"

                # Transfer embeddings from graph GPU to price GPU
                embeddings_tensor = torch.cat(embeddings, dim=0).reshape(B, *embeddings[0].shape)
                embeddings_on_price_gpu = embeddings_tensor.to(price_model_device, non_blocking=True)

                X, returns = X.to(price_model_device, non_blocking=True), y.to(price_model_device, non_blocking=True)
                batch_loss = torch.tensor(0.0, dtype=X.dtype).to(price_model_device, non_blocking=True)

                # Prepare data for model (reshape as in original)
                x = X.permute(0, 2, 1,
                              3).contiguous()  # [B, seq_len, stocks, features] -> [B, stocks, seq_len, features]
                x = x.reshape(-1, x.shape[-2], x.shape[-1])  # [B*stocks, seq_len, features]

                # Forward pass
                with torch.autocast(device_type=price_model_device.type):
                    weights = price_model(x)  # [B*stocks, 1] or similar
                    weights = weights.reshape(B, stocks, weights.shape[-1])
                    weights = cross_attention_head(weights, embeddings_on_price_gpu,
                                                   None)  # [B, stocks, features] -> [B, stocks]
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
        print(f"Val Loss: {avg_val_loss:.6f}, Val Sharpe: {val_sharpe:.4f}")
        print(f"Test Loss: {avg_test_loss:.6f}, Test Sharpe: {test_sharpe:.4f}")
        logger.writer.flush()

        # In your training loop, replace the early stopping section with this:

        if epoch >= early_stopping.warmup_epochs:
            try:
                print("Packing model for early stopping check...")

                # Choose packing method based on your needs:

                # Option 1: Full optimized packing (recommended)
                packed_model = pack_model_optimized(price_model, graph_model, init_static_emb,
                                                    dynamic_emb, node_latest_event_time, val_sharpe)

                # Option 2: Lightweight packing (if you can recreate embeddings)
                # packed_model = pack_model_lightweight(price_model, graph_model, val_sharpe)

                print("Model packed successfully")

                early_stopping(epoch=epoch, value=val_sharpe, model=packed_model, path=model_path)

                if early_stopping.early_stop:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

            except Exception as e:
                print(f"Error during model packing: {e}")
                print("Attempting lightweight packing as fallback...")
                try:
                    # Fallback to lightweight packing
                    packed_model = pack_model_lightweight(price_model, graph_model, val_sharpe)
                    early_stopping(epoch=epoch, value=val_sharpe, model=packed_model, path=model_path)
                    if early_stopping.early_stop:
                        print(f"Early stopping triggered at epoch {epoch}")
                        break
                except Exception as e2:
                    print(f"Fallback packing also failed: {e2}")
                    print("Continuing training without saving checkpoint...")

        torch.cuda.empty_cache()

def main():
    p = argparse.ArgumentParser(description='Combo Model Args')

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
    p.add_argument('--N_stocks', type=int, default=512, help='Universe of stocks to select from')
    p.add_argument('--ca_hidden_dim', type=int, default=64, help='Hidden dimension for Cross-Attention')

    p.add_argument('--min_weight', type=float, default=0.005)
    p.add_argument('--components', type=int, default=40)
    p.add_argument('--volatility', type=float, default=0.14)
    p.add_argument('--risk_aversion', type=float, default=5.0)
    p.add_argument('--utility', type=float, default=5.0)

    p.add_argument('--lc_utility', type=float, default=10.0)
    p.add_argument('--lc_components', type=float, default=5.0)
    p.add_argument('--lc_volatility', type=float, default=0.0)
    p.add_argument('--lc_min_weight', type=float, default=0.0)

    # graph model parameters
    p.add_argument('--gm_in_dim', type=int, default=128, help='Input embedding dimension for graph model')
    p.add_argument('--gm_structural_hid_dim', type=int, default=128, help='Hidden dimension for structural embeddings in graph convolution')
    p.add_argument('--gm_temporal_hid_dim', type=int, default=128, help='Hidden dimension for temporal embeddings in graph convolution')
    p.add_argument('--gm_structural_RNN', type=str, default='RNN', choices=['RNN', 'GRU'], help='RNN type for structural graph convolution (RNN or GRU)')
    p.add_argument('--gm_temporal_RNN', type=str, default='RNN', choices=['RNN', 'GRU'], help='RNN type for temporal graph convolution (RNN or GRU)')
    p.add_argument('--gm_num_gconv_layers', type=int, default=2, help='Number of relational graph convolution layers in RGCN')
    p.add_argument('--gm_rgcn_bdd_bases', type=int, default=16, help='Number of basis decomposition bases for RGCN regularization. in_dim and hid_dim (both structural and temporal) need to be divisible by this number')
    p.add_argument('--gm_num_rnn_layers', type=int, default=2, help='Number of RNN layers for both structural and temporal processing')
    p.add_argument('--gm_dropout', type=float, default=0.2, help='Dropout rate for graph model regularization')
    p.add_argument('--gm_activation', type=str, default='tanh', choices=['tanh', 'relu'], help='Activation function for graph convolutions')
    p.add_argument('--gm_decay_factor', type=float, default=0.8, help='State decay factor for dynamic embeddings (0 < factor <= 1)')
    p.add_argument('--gm_head_dropout', type=float, default=0.2, help='Dropout rate for the combiner head that fuses structural and temporal embeddings')
    p.add_argument('--gm_out_dim', type=int, default=32, help='Output dimension of the graph model combiner')
    p.add_argument('--gm_time_interval_log_transform', action='store_true', default=True, help='Apply log transformation to time intervals in temporal processing')
    p.add_argument('--gm_gpu', type=int, default=-1, help='GPU device for graph model')

    p.add_argument('--price_model', type=str, default='patch_tst')  # <-- add
    p.add_argument('--run_name', type=str, default='combo_PatchTST')  # <-- add

    p.add_argument('--input_dim', type=int, default=45)
    p.add_argument('--stem_dims', type=int, nargs='+', default=[16])
    p.add_argument('--stem_dropout', type=float, default=0.2)
    p.add_argument('--lstm_hidden_dim', type=int, default=64)  # NOTE: "hidden"
    p.add_argument('--lstm_layers', type=int, default=2)
    p.add_argument('--lstm_dropout', type=float, default=0.1)
    p.add_argument('--head_dims', type=int, nargs='+', default=[32, 16])
    p.add_argument('--head_dropout', type=float, default=0.1)

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

    p.add_argument('--graph_gpu', type=int, default=0, help='GPU ID for graph model')
    p.add_argument('--price_gpu', type=int, default=1, help='GPU ID for price model')
    p.add_argument('--force_single_gpu', action='store_true', help='Force single GPU usage')

    args = p.parse_args()

    # Multi-GPU device setup
    if args.force_single_gpu or torch.cuda.device_count() < 2:
        print("Using single GPU for both models")
        graph_device = price_device = set_device()
    else:
        print(f"Using multi-GPU setup: Graph GPU {args.graph_gpu}, Price GPU {args.price_gpu}")
        graph_device = set_device(args.graph_gpu)
        price_device = set_device(args.price_gpu)

        # Verify devices are different
        if graph_device == price_device:
            print("Warning: Graph and price models on same device")

    set_seed(args.seed)

    # Create descriptive run name
    if args.run_name:
        run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.run_name}"
    else:
        run_name = create_run_name(args)

    if args.price_model == 'patch_tst':
        price_config = dict(d_model=args.d_model,
                            n_heads=args.n_heads,
                            d_ff=args.d_ff,
                            dropout=args.dropout,
                            e_layers=args.e_layers,
                            factor=args.factor,
                            activation=args.activation,
                            enc_in=args.enc_in,
                            patch_len=args.patch_len,
                            stride=args.stride,
                            output_logits=False)

        price_model = PatchTSTModel(**price_config)
        price_model_out_dim = args.enc_in
        model_name = 'Combo_PatchTST'
    else:
        price_config = dict(input_size=args.input_dim,
                      stem_dims=args.stem_dims,
                      stem_dropout=args.stem_dropout,
                      lstm_hidden_dim=args.lstm_hidden_dim,
                      lstm_layers=args.lstm_layers,
                      lstm_dropout=args.lstm_dropout,
                      head_dims=args.head_dims,
                      head_dropout=args.head_dropout,
                      activation_name=args.activation)
        head_dims = args.head_dims
        price_model_out_dim = head_dims[-1]
        model_name = 'Combo_BiLSTM'
        price_model = BidirectionalLSTM(**price_config)


    # Setup directories
    log_dir, model_path = setup_directories(model_name, run_name)

    print(f"Run name: {run_name}")
    print(f"Log directory: {log_dir}")
    print(f"Model will be saved to: {model_path}")

    root_dir = Path(__file__).resolve().parent / args.data_root
    price_train_loader, price_val_loader, price_test_loader = get_dataloaders(batch_size=args.batch_size,
                                                                              root_dir=root_dir / 'market_data',
                                                                              lookback=args.lookback,
                                                                              prediction_horizon=args.holding_period)
    tickers = get_tickers(root_dir=root_dir / 'market_data')
    loss_fn = LongOnlyMarkowitzPortfolioLoss(target_volatility=args.volatility,
                                             target_cardinality=args.components,
                                             target_min_weight=args.min_weight,
                                             target_min_utility=args.utility,
                                             risk_aversion=args.risk_aversion,)


    G = load_temporal_knowledge_graph(root_dir=root_dir / 'triplets')
    graph_statics = TemporalKnowledgeGraphStatics(root_dir=root_dir / 'triplets')

    graph_model_args = dict(num_nodes=G.number_of_nodes(),
                            num_rels=G.num_relations,
                            in_dim=args.gm_in_dim,
                            structural_hid_dim=args.gm_structural_hid_dim,
                            temporal_hid_dim=args.gm_temporal_hid_dim,
                            structural_RNN=args.gm_structural_RNN,
                            temporal_RNN=args.gm_temporal_RNN,
                            num_gconv_layers=args.gm_num_gconv_layers,
                            rgcn_bdd_bases=args.gm_rgcn_bdd_bases,
                            num_rnn_layers=args.gm_num_rnn_layers,
                            dropout=args.gm_dropout,
                            activation=args.gm_activation,
                            decay_factor=args.gm_decay_factor,
                            head_dropout=args.gm_head_dropout,
                            out_dim=args.gm_out_dim,
                            time_interval_log_transform=args.gm_time_interval_log_transform)

    node_latest_event_time = torch.zeros(
        (G.number_of_nodes(), G.number_of_nodes() + 1, 2),
        dtype=torch.float32,
        device=graph_device  # Create directly on GPU
    )
    static_emb, init_dynamic_emb = initialise_embeddings(num_nodes=G.number_of_nodes(),
                                                         embedding_dim=args.gm_in_dim,
                                                         num_rnn_layers=args.gm_num_rnn_layers,
                                                         device=graph_device,)

    # ADD THE TENSOR SIZE CHECKS HERE - RIGHT AFTER TENSOR CREATION
    print("\n" + "=" * 60)
    print("TENSOR SIZE ANALYSIS")
    print("=" * 60)

    # Check individual tensor sizes
    tensor_info = check_tensor_sizes(node_latest_event_time, static_emb, init_dynamic_emb)

    # Check node_latest_event_time optimization
    optimize_node_latest_event_time_size(G.number_of_nodes())

    # Print summary
    if not tensor_info['sparse_safe']:
        print("\nWARNING: Large tensors detected!")
        print("   - Sparse tensor conversion will be disabled for safety")
        print("   - Consider reducing the graph size or using chunked processing")
    else:
        print("\nAll tensors are within safe limits for sparse conversion")

    print("=" * 60)
    graph_model = GraphModel(node_latest_event_time=node_latest_event_time, device=graph_device, **graph_model_args)

    cross_attention_head = CrossAttentionHead(query_dim=price_model_out_dim,
                                              node_dim=args.gm_out_dim,
                                              hidden_dim=args.ca_hidden_dim,
                                              num_nodes=G.number_of_nodes(),
                                              return_attention=False)
    cross_attention_head.to(price_device)
    graph_model.to(graph_device)
    price_model.to(price_device)

    graph_params = list(graph_model.parameters()) + [
        static_emb.structural, static_emb.temporal, init_dynamic_emb.structural, init_dynamic_emb.temporal
    ]
    price_params = list(price_model.parameters())
    cross_attention_params = list(cross_attention_head.parameters())

    optimizer = optim.Adam(graph_params + price_params + cross_attention_params, lr=args.lr, weight_decay=1e-5)

    if args.scheduler_type is not None:
        print(f"Using scheduler type: {args.scheduler_type}")
        scheduler = get_scheduler(optimizer, args.scheduler_type)
    else:
        scheduler = get_scheduler(optimizer, 'plateau')

    early_stopping = EarlyStopping(patience=args.patience, mode='max', warmup_epochs=args.warmup_epochs)
    logger = PortfolioTensorBoardLogger(log_dir=str(log_dir))

    # Save configuration with run name
    config = {
        'run_name': run_name,
        'model': model_name,
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'patience': args.patience,
        'min_weight': args.min_weight,

        'components': args.components,
        'volatility': args.volatility,
        'risk_aversion': args.risk_aversion,
        'utility': args.utility,
        'lc_utility': args.lc_utility,
        'lc_components': args.lc_components,
        'lc_volatility': args.lc_volatility,
        'lc_min_weight': args.lc_min_weight,
        'scaler': torch.cuda.is_available(),
        'lookback': args.lookback,
        'holding_period': args.holding_period,
        'scheduler_type': args.scheduler_type,

        'price_device': str(price_device),
        'graph_device': str(graph_device),
    }

    full_config = {'run_config': config,
                   'cross_attention_hid_dim': args.ca_hidden_dim,
                   'price_model_config': price_config,
                   'graph_model_config': graph_model_args}

    with open(log_dir / 'config.json', 'w') as f:
        json.dump(full_config, f, indent=2)

    print(f"Starting training with config:\n{pp(full_config)}")
    print('GPU for price model:', price_device)

    loss_coefficients = dict(utility_target_penalty=args.lc_utility,
                             cardinality_target_penalty=args.lc_components,
                             vol_target_penalty=args.lc_volatility,
                             min_weight_penalty=args.lc_min_weight)
    if price_device.type == 'cuda':
        scaler = torch.amp.GradScaler()
        run_cuda_multi_gpu(
            # models
            cross_attention_head=cross_attention_head,
            price_model=price_model,
            graph_model=graph_model,
            price_model_device=price_device,
            graph_model_device=graph_device,
            # data loaders - Prices
            price_train_loader=price_train_loader,
            price_val_loader=price_val_loader,
            price_test_loader=price_test_loader,
            # data loaders - Graph
            G=G,
            graph_statics=graph_statics,
            # Graph model inputs - embeddings
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
            tickers=tickers
        )
    else:
        raise NotImplementedError

    logger.writer.close()
    print(f"Training complete. Results saved to {log_dir}")


if __name__ == '__main__':
    main()
