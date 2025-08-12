import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from entmax import entmax_bisect, entmax15


class CrossAttentionBase(nn.Module):
    def __init__(self, query_dim, node_dim, hidden_dim, num_nodes, return_weights=False):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(node_dim, hidden_dim)
        self.value_proj = nn.Linear(node_dim, hidden_dim)
        self.node_id_embedding = nn.Embedding(num_nodes, node_dim)
        self.return_weights = return_weights
        self.register_buffer("node_ids", torch.arange(num_nodes), persistent=False)

    def project(self, query, context):
        B, N, _ = context.shape
        node_id_embeds = self.node_id_embedding(self.node_ids)             # (N, node_dim)
        node_id_embeds = node_id_embeds.unsqueeze(0).expand(B, -1, -1)     # (B, N, node_dim)
        context = context + node_id_embeds                                 # Inject identity
        Q = self.query_proj(query)                                         # (B, S, hidden_dim)
        K = self.key_proj(context)                                         # (B, N, hidden_dim)
        V = self.value_proj(context)                                       # (B, N, hidden_dim)
        return Q, K, V


class CrossAttentionMasked(CrossAttentionBase):
    def forward(self, query, context, active_nid_lists: List[torch.Tensor]):
        """
        query:   (B, S, query_dim)           # one vector per stock
        context: (B, N, node_dim)            # full graph embedding per batch
        active_nid_lists: list of Tensors of active node IDs per batch item
        Returns:
            attended: (B, S, hidden_dim)
            attn_weights: (B, S, N)
        """
        B, N, _ = context.shape
        Q, K, V = self.project(query, context)

        # Build binary mask: True = masked out
        device = query.device
        # mask = torch.ones(B, N, dtype=torch.bool, device=device)
        # for b in range(B):
        #     mask[b, active_nid_lists[len(active_nid_lists) - b - 1]] = False  # unmask active nodes

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / (K.size(-1) ** 0.5)  # (B, S, N)
        # attn_scores = attn_scores.masked_fill(mask.unsqueeze(1), float('-inf'))  # (B, S, N)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, S, N)

        attended = torch.matmul(attn_weights, V)  # (B, S, hidden_dim)
        if self.return_weights:
            return attended, attn_weights
        return attended


class CrossAttentionHead(nn.Module):
    def __init__(self, query_dim, node_dim, hidden_dim, num_nodes, return_attention=False):
        super().__init__()
        self.cross_attention = CrossAttentionMasked(query_dim, node_dim, hidden_dim, num_nodes,
                                                    return_weights=return_attention)
        self.return_attention = return_attention
        self.head = nn.Linear(hidden_dim, 1)
        self.register_buffer("node_ids", torch.arange(num_nodes), persistent=False)

    def forward(self, query, context, active_nid_lists: List[torch.Tensor]):
        if self.return_attention:
            attended, attn_weights = self.cross_attention(query, context, active_nid_lists)
        else:
            attended = self.cross_attention(query, context, active_nid_lists)
        attended = self.head(attended).squeeze()  # (B, S)
        attended = entmax15(attended, dim=-1)
        return attended


# class CrossAttentionBase(nn.Module):
#     def __init__(self, query_dim, node_dim, hidden_dim, num_nodes):
#         super().__init__()
#         self.query_proj = nn.Linear(query_dim, hidden_dim)
#         self.key_proj = nn.Linear(node_dim, hidden_dim)
#         self.value_proj = nn.Linear(node_dim, hidden_dim)
#         self.node_id_embedding = nn.Embedding(num_nodes, node_dim)
#         self.register_buffer("node_ids", torch.arange(num_nodes), persistent=False)
#
#     def project(self, query, context):
#         B = query.shape[0]
#         node_id_embeds = self.node_id_embedding(self.node_ids).unsqueeze(0).expand(B, -1, -1)
#         context = context + node_id_embeds
#         Q = self.query_proj(query).unsqueeze(1)  # (B, 1, hidden)
#         K = self.key_proj(context)
#         V = self.value_proj(context)
#         return Q, K, V
#
#
# class CrossAttentionFullMask(CrossAttentionBase):
#     """Option 1: Full context with key padding mask"""
#     def forward(self, query, context, active_nid_lists: List[torch.Tensor]):
#         B, N, _ = context.shape
#         Q, K, V = self.project(query, context)
#
#         # Build key mask: (B, N) -> True for masked (inactive) nodes
#         device = query.device
#         mask = torch.ones(B, N, dtype=torch.bool, device=device)
#         for b in range(B):
#             mask[b, active_nid_lists[b]] = False  # unmask active
#
#         attn_scores = torch.matmul(Q, K.transpose(1, 2)) / (K.size(-1) ** 0.5)
#         attn_scores = attn_scores.masked_fill(mask.unsqueeze(1), float('-inf'))
#         attn_weights = F.softmax(attn_scores, dim=-1)
#         attended = torch.matmul(attn_weights, V)
#         return attended.squeeze(1)
#
#
# class CrossAttentionSparse(CrossAttentionBase):
#     """Option 2: Sparse context by indexing active nodes only"""
#     def forward(self, query, context, active_nid_lists: List[torch.Tensor]):
#         outputs = []
#         device = query.device
#         for b, active_ids in enumerate(active_nid_lists):
#             ctx_b = context[b, active_ids]  # (k, node_dim)
#             node_ids_b = self.node_ids[active_ids]
#             ctx_b += self.node_id_embedding(node_ids_b)
#
#             Q = self.query_proj(query[b:b+1]).unsqueeze(1)  # (1, 1, hidden)
#             K = self.key_proj(ctx_b).unsqueeze(0)           # (1, k, hidden)
#             V = self.value_proj(ctx_b).unsqueeze(0)
#
#             scores = torch.matmul(Q, K.transpose(1, 2)) / (K.size(-1) ** 0.5)
#             weights = F.softmax(scores, dim=-1)
#             out = torch.matmul(weights, V).squeeze(1)       # (1, hidden)
#             outputs.append(out)
#         return torch.cat(outputs, dim=0)
#
#
# class CrossAttentionTopK(CrossAttentionBase):
#     """Option 3: Full context but attention restricted to top-k nodes"""
#     def __init__(self, query_dim, node_dim, hidden_dim, num_nodes, k_top):
#         super().__init__(query_dim, node_dim, hidden_dim, num_nodes)
#         self.k_top = k_top
#
#     def forward(self, query, context):
#         B, N, _ = context.shape
#         Q, K, V = self.project(query, context)
#
#         scores = torch.matmul(Q, K.transpose(1, 2)) / (K.size(-1) ** 0.5)  # (B, 1, N)
#         topk_scores, topk_idx = scores.topk(self.k_top, dim=-1)           # (B, 1, k)
#         topk_weights = F.softmax(topk_scores, dim=-1)
#
#         # Gather top-k values
#         V_gathered = torch.gather(
#             V, dim=1, index=topk_idx.expand(-1, -1, V.size(-1))  # (B, 1, k, d)
#         )
#         attended = torch.matmul(topk_weights, V_gathered)        # (B, 1, d)
#         return attended.squeeze(1)
#
