# News Graph Model

Temporal relational graph encoder for news-derived triplets.

Files:
- `graph_models/dataloader.py`: load temporal triplets; `collate_fn` returns per-time subgraphs; mappings for entities/relations/dates.
- `graph_models/news_graph.py`: core model and embedding pipeline.

## Data and batching
- Triplets (h, r, t, dt) → DGL graph with `rel_type` and `time` on edges.
- `collate_fn([(i, date)], G)` extracts subgraph for time index `i` and computes node norms.

## Embeddings
- `initialise_embeddings`: static structural/temporal (learned); dynamic structural `[N, L, D]` and temporal `[N, L, D, 2]` (fwd/rev).
- Dynamic embeddings live on CPU globally; moved to `device` for updates, then written back to CPU (scales to large graphs).

## Temporal state & decay
- `node_latest_event_time` `[N, N+1, 2]` stores latest event times (per-neighbour and global) for forward/reverse directions.
- `EventTimeHelper` computes inter-event times and updates this state.
- `NodeStateDecay(decay_factor)`: multiplies dynamic state by `alpha` each step to regularise stale nodes.

## Structural & temporal paths
- Structural: RGCN over relation-typed edges → features → `GraphStructuralRNNConv` updates structural hidden state for active nodes.
- Temporal: edge scaling from inter-event times on graph and reversed graph → RGCN → `GraphTemporalRNNConv` updates temporal state (both directions).
- `UpdateEmbedding` orchestrates: move → decay → update active nodes → write back.

## Combiner
- Concatenates last structural layer with both temporal directions and projects to `out_dim`. Cross-attention consumes this per-node embedding as context.

## Devices
- Pass `device` to `GraphModel` to select processing device. Dynamic tensors remain on CPU outside the forward pass.
