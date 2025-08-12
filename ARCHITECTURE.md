# End-to-End Architecture

## Data
- Prices: `price_models/dataloader.py` → batches with `X` (lookback), `y` (future returns over holding period), `dates`, `prediction_dates`.
- Graph: `graph_models/dataloader.py` → DGL graph with triplets and `collate_fn` for per-time subgraphs.

## Training
1. For each batch window, stream subgraphs from `start_idx` to `end_idx` through the graph model to update dynamic embeddings; on each prediction date, compute combined per-node embeddings and collect them.
2. Prepare price inputs `[B, seq, stocks, feat] → [B*stocks, seq, feat]`; run the price model to get per-stock queries.
3. Cross-attention: attend from queries to node context → per-stock logits → entmax to portfolio weights.
4. Compute loss terms with `LongOnlyMarkowitzPortfolioLoss` and combine via coefficients `lc_*`; backprop and step.

## Devices
- `run.py`:
  - Price model on primary device from `utils.set_device()` (CUDA/MPS/CPU).
  - Graph model updates on CPU; dynamic embeddings remain on CPU.
- `run_dual_gpu.py`:
  - If 2+ CUDA GPUs: graph model on `cuda:1`, price model on `cuda:0`; otherwise falls back to CPU+GPU or CPU+MPS.
  - Transfers graph outputs to the price device before attention; enables mixed precision for price model on CUDA.

## Early stopping & logging
- Early stopping monitors validation Sharpe; best snapshot packs both models and embedding state.
- `PortfolioTensorBoardLogger` logs batch/epoch metrics and plots weights/returns.

## Notes
- Dynamic embeddings are large; the CPU-resident design enables scaling to larger graphs with limited VRAM.
- Date alignment uses `TemporalKnowledgeGraphStatics` mappings.
- The current cross-attention mask is optional; enable it to restrict to active nodes if desired.
