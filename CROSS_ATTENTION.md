# Cross Attention (`cross_attention.py`)

Bridges price-model queries to graph node embeddings to produce portfolio weights.

## Components
- `CrossAttentionBase`: linear projections for query/key/value; adds learned node-id embeddings to context.
- `CrossAttentionMasked`: scaled dot-product attention from per-stock queries to all nodes; optional masking code present but commented.
- `CrossAttentionHead`: attention + linear head → one logit per stock; applies `entmax15` to get sparse, non-negative weights summing to 1.

## Shapes
- Query (price): `(B, S, query_dim)`; Context (graph): `(B, N, node_dim)`; Output weights: `(B, S)`.

## Subtleties
- Masking inactive nodes: `active_nid_lists` is provided; to restrict attention to active nodes, re-enable the mask or implement sparse attention.
- Double entmax: The head applies `entmax15`; training also applies `entmax15` in some paths. Avoid double application to prevent over-sparsification.
- Node identity: learned node-id embeddings stabilise attention among similar nodes.
- Device: ensure query and context are on the same device before attention (the runners transfer context to the price device).
