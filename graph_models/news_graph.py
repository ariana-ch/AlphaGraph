from collections import namedtuple
import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch.conv.relgraphconv import RelGraphConv
import dgl.function as fn
from typing import Optional


MultiAspectEmbedding = namedtuple('MultiAspectEmbedding', ['structural', 'temporal'], defaults=[None, None])


def node_norm_to_edge_norm(G):
    G = G.local_var()
    for et in G.canonical_etypes:
        if G.num_edges(et) == 0:
            print(f"Warning: Edge type {et} has no edges, skipping norm computation.")

    G.apply_edges(lambda edges: {'norm': edges.dst['norm']})
    return G.edata['norm']


class TimeIntervalTransform(nn.Module):
    EPSILON = 1e-10

    def __init__(self, log_transform=True, normalize=False, time_intervals=None):
        super().__init__()
        self.enc_dim = 1
        self.log_transform = log_transform
        self.normalize = normalize
        if self.log_transform and time_intervals is not None:
            self.time_intervals = torch.log(time_intervals + self.EPSILON)
        else:
            self.time_intervals = time_intervals
        if self.normalize:
            self.mean_time_interval = self.time_intervals.mean()
            self.std_time_interval = self.time_intervals.std()

    def forward(self, time_intervals):
        return self.transform(time_intervals)

    def transform(self, time_intervals):
        if self.log_transform:
            time_intervals = torch.log(time_intervals + self.EPSILON)
        if self.normalize:
            return (time_intervals - self.mean_time_interval) / self.std_time_interval
        else:
            return time_intervals

    def reverse_transform(self, time_intervals):
        if self.log_transform:
            time_intervals = torch.exp(time_intervals)
        if self.normalize:
            return time_intervals * self.std_time_interval + self.mean_time_interval
        else:
            return time_intervals

    def __repr__(self):
        field_desc = [f"log_transform={self.log_transform}", f"normalize={self.normalize}"]
        return f"{self.__class__.__name__}({', '.join(field_desc)})"


class EventTimeHelper:
    @classmethod
    def get_sparse_inter_event_times(cls, batch_G, node_latest_event_time, _global=False):
        batch_sparse_latest_event_times = cls.get_sparse_latest_event_times(batch_G, node_latest_event_time, _global)
        return batch_G.edata['time'] - batch_sparse_latest_event_times

    @classmethod
    def get_sparse_latest_event_times(cls, batch_G, node_latest_event_time, _global=False):
        batch_G_nid = batch_G.ndata[dgl.NID].long()
        batch_latest_event_time = node_latest_event_time[batch_G_nid]
        batch_G_src, batch_G_dst = batch_G.edges()
        device = batch_G.ndata[dgl.NID].device
        if _global:
            return batch_latest_event_time[batch_G_dst.long(), -1].to(device)
        else:
            return batch_latest_event_time[batch_G_dst.long(), batch_G_nid[batch_G_src.long()]].to(device)

    @classmethod
    def get_inter_event_times(cls, batch_G, node_latest_event_time, update_latest_event_time=True):
        batch_G_nid = batch_G.ndata[dgl.NID].long()
        batch_latest_event_time = node_latest_event_time[batch_G_nid]
        batch_G_src, batch_G_dst = batch_G.edges()
        batch_G_src, batch_G_dst = batch_G_src.long(), batch_G_dst.long()
        batch_G_time = batch_G.edata['time'].to(node_latest_event_time.dtype)
        device = batch_G.ndata[dgl.NID].device

        # FIXED: Create batch_inter_event_times on the same device as node_latest_event_time
        batch_inter_event_times = torch.zeros(
            batch_G.num_nodes(), batch_G.num_all_nodes + 1,
            dtype=node_latest_event_time.dtype,
            device=node_latest_event_time.device  # Use same device as node_latest_event_time
        )

        batch_inter_event_times[batch_G_dst, batch_G_nid[batch_G_src]] = \
            batch_G_time - batch_latest_event_time[batch_G_dst, batch_G_nid[batch_G_src]].to(device)

        batch_G.update_all(message_func=lambda edges: {'t': edges.data['time'].float()},
                           reduce_func=fn.max('t', 'max_event_time'))
        batch_G_max_event_time = batch_G.ndata['max_event_time'].to(torch.int32)
        batch_max_latest_event_time = batch_latest_event_time[:, -1].to(device)
        batch_G_max_event_time = torch.max(batch_G_max_event_time, batch_max_latest_event_time)
        batch_inter_event_times[:, -1] = batch_G_max_event_time - batch_max_latest_event_time

        if update_latest_event_time:
            # FIXED: Keep everything on the same device - don't move to CPU
            node_latest_event_time[batch_G_nid[batch_G_dst], batch_G_nid[batch_G_src]] = batch_G_time.to(
                node_latest_event_time.device)
            node_latest_event_time[batch_G_nid, -1] = batch_G_max_event_time.to(node_latest_event_time.device)

        return batch_inter_event_times


class EventTimeHelperOLD:
    @classmethod
    def get_sparse_inter_event_times(cls, batch_G, node_latest_event_time, _global=False):
        batch_sparse_latest_event_times = cls.get_sparse_latest_event_times(batch_G, node_latest_event_time, _global)
        return batch_G.edata['time'] - batch_sparse_latest_event_times

    @classmethod
    def get_sparse_latest_event_times(cls, batch_G, node_latest_event_time, _global=False):
        batch_G_nid = batch_G.ndata[dgl.NID].long()
        batch_latest_event_time = node_latest_event_time[batch_G_nid]
        batch_G_src, batch_G_dst = batch_G.edges()
        device = batch_G.ndata[dgl.NID].device
        if _global:
            return batch_latest_event_time[batch_G_dst.long(), -1].to(device)
        else:
            return batch_latest_event_time[batch_G_dst.long(), batch_G_nid[batch_G_src.long()]].to(device)

    @classmethod
    def get_inter_event_times(cls, batch_G, node_latest_event_time, update_latest_event_time=True):
        batch_G_nid = batch_G.ndata[dgl.NID].long()
        batch_latest_event_time = node_latest_event_time[batch_G_nid]
        batch_G_src, batch_G_dst = batch_G.edges()
        batch_G_src, batch_G_dst = batch_G_src.long(), batch_G_dst.long()
        batch_G_time = batch_G.edata['time'].to(node_latest_event_time.dtype)
        device = batch_G.ndata[dgl.NID].device
        batch_inter_event_times = torch.zeros(batch_G.num_nodes(), batch_G.num_all_nodes + 1,
                                              dtype=node_latest_event_time.dtype).to(device)
        batch_inter_event_times[batch_G_dst, batch_G_nid[batch_G_src]] = \
            batch_G_time - batch_latest_event_time[batch_G_dst, batch_G_nid[batch_G_src]].to(device)
        # batch_G.update_all(fn.copy_e('time', 't'), fn.max('t', 'max_event_time'))
        batch_G.update_all(message_func=lambda edges: {'t': edges.data['time'].float()},
                           reduce_func=fn.max('t', 'max_event_time'))
        batch_G_max_event_time = batch_G.ndata['max_event_time'].to(torch.int32)
        batch_max_latest_event_time = batch_latest_event_time[:, -1].to(device)
        batch_G_max_event_time = torch.max(batch_G_max_event_time, batch_max_latest_event_time)
        batch_inter_event_times[:, -1] = batch_G_max_event_time - batch_max_latest_event_time
        if update_latest_event_time:
            node_latest_event_time[batch_G_nid[batch_G_dst], batch_G_nid[batch_G_src]] = batch_G_time.cpu()
            node_latest_event_time[batch_G_nid, -1] = batch_G_max_event_time.cpu()
        return batch_inter_event_times


# =============================================================================
# NEW: Node State Decay Module
# =============================================================================
class NodeStateDecay(nn.Module):
    """Applies state decay to dynamic embeddings."""

    def __init__(self, decay_factor=0.99):
        super().__init__()
        assert 0.0 < decay_factor <= 1.0, "Decay factor must be in (0, 1]"
        self.alpha = decay_factor

    def forward(self, dynamic_embeddings, base_embeddings):
        """
        Applies decay to both structural and temporal dynamic embeddings.

        Args:
            dynamic_embeddings (MultiAspectEmbedding): The current dynamic embeddings, must be on the target device.
            base_embeddings (MultiAspectEmbedding): The baseline to decay towards (e.g., static embeddings).

        Returns:
            MultiAspectEmbedding: The decayed dynamic embeddings.
        """
        # Ensure base embeddings are on the same device as dynamic embeddings
        device = dynamic_embeddings.structural.device
        alpha = torch.tensor(self.alpha, device=device).type(dynamic_embeddings.structural.dtype)
        # Decay structural part
        decayed_structural = self.alpha * dynamic_embeddings.structural

        # Decay temporal part
        decayed_temporal = self.alpha * dynamic_embeddings.temporal

        return MultiAspectEmbedding(structural=decayed_structural, temporal=decayed_temporal)


class RGCN(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 n_layers,
                 num_rels,
                 regularizer="basis",
                 num_bases=None,
                 use_bias=True,
                 activation='relu',
                 use_self_loop=True,
                 dropout=0.0,
                 layer_norm=False):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.regularizer = regularizer
        self.num_bases = num_bases
        self.use_bias = use_bias
        self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()
        self.use_self_loop = use_self_loop
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        if self.n_layers == 1:
            self.layers.append(RelGraphConv(
                self.in_dim, self.out_dim, self.num_rels, self.regularizer, self.num_bases, self.use_bias,
                activation=None, self_loop=self.use_self_loop, dropout=self.dropout,
                layer_norm=self.layer_norm,
            ))
        else:
            self.layers.append(RelGraphConv(
                self.in_dim, self.hid_dim, self.num_rels, self.regularizer, self.num_bases, self.use_bias,
                activation=self.activation, self_loop=self.use_self_loop, dropout=self.dropout,
                layer_norm=self.layer_norm,
            ))
            for i in range(1, self.n_layers - 1):
                self.layers.append(RelGraphConv(
                    self.hid_dim, self.hid_dim, self.num_rels, self.regularizer, self.num_bases, self.use_bias,
                    activation=self.activation, self_loop=self.use_self_loop, dropout=self.dropout,
                    layer_norm=self.layer_norm,
                ))
            self.layers.append(RelGraphConv(
                self.hid_dim, self.out_dim, self.num_rels, self.regularizer, self.num_bases, self.use_bias,
                activation=None, self_loop=self.use_self_loop, dropout=self.dropout,
                layer_norm=self.layer_norm,
            ))
        assert self.n_layers == len(self.layers), (self.n_layers, len(self.layers))

    def forward(self, G, emb, etypes, edge_norm=None):
        if edge_norm is not None:
            edge_norm = edge_norm.view(-1, 1)
        for layer in self.layers:
            emb = layer(G, emb, etypes, edge_norm)
        return emb


class GraphStructuralRNNConv(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 num_rels: int,
                 rnn_layer: str = 'RNN',
                 in_dim: int = 200,
                 hid_dim: int = 200,
                 num_conv_layers: int = 2,
                 rgcn_bdd_bases: int = 100,
                 num_rnn_layers: int = 1,
                 add_entity_emb=False,
                 dropout: float = 0.2,
                 activation: str = 'tanh'):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.convolution = RGCN(in_dim=in_dim, out_dim=hid_dim, hid_dim=hid_dim, n_layers=num_conv_layers,
                                num_rels=num_rels, num_bases=rgcn_bdd_bases, regularizer="bdd", dropout=dropout,
                                activation=activation, layer_norm=False)
        structural_rnn_in_dim = hid_dim
        self.add_entity_emb = add_entity_emb
        if self.add_entity_emb:
            structural_rnn_in_dim += hid_dim
        if rnn_layer == "GRU":
            self.rnn = nn.GRU(input_size=structural_rnn_in_dim, hidden_size=hid_dim,
                              num_layers=num_rnn_layers, batch_first=True, dropout=0.0)
        elif rnn_layer == "RNN":
            self.rnn = nn.RNN(input_size=structural_rnn_in_dim, hidden_size=hid_dim,
                              num_layers=num_rnn_layers, batch_first=True, dropout=0.0)
        else:
            raise ValueError(f"Invalid RNN layer: {rnn_layer}")
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_G, dynamic_entity_emb, static_entity_emb, device, batch_node_indices=None):
        if batch_node_indices is None:
            batch_node_indices = batch_G.nodes().long()
        batch_structural_static_entity_emb = static_entity_emb.structural[batch_G.ndata[dgl.NID].long()].to(device)
        edge_norm = node_norm_to_edge_norm(batch_G)
        conv_structural_static_emb = self.convolution(batch_G, batch_structural_static_entity_emb,
                                                      batch_G.edata['rel_type'].long(), edge_norm)
        structural_rnn_input = [conv_structural_static_emb[batch_node_indices]]
        if self.add_entity_emb:
            structural_rnn_input.append(
                static_entity_emb.structural[batch_G.ndata[dgl.NID][batch_node_indices].long()].to(device))
        structural_rnn_input = torch.cat(structural_rnn_input, dim=1).unsqueeze(1)
        # NOTE: dynamic_entity_emb is now expected to be on the correct device already
        structural_dynamic = dynamic_entity_emb.structural[batch_G.ndata[dgl.NID][batch_node_indices].long()]
        output, hn = self.rnn(structural_rnn_input, structural_dynamic.transpose(0, 1).contiguous())
        updated_structural_dynamic_entity_emb = hn.transpose(0, 1)
        return updated_structural_dynamic_entity_emb


class GraphTemporalRNNConv(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 num_rels: int,
                 node_latest_event_time: torch.Tensor = None,
                 time_interval_transform: Optional[torch.nn.Module] = None,
                 rnn_layer: str = 'RNN',
                 in_dim: int = 200,
                 hid_dim: int = 200,
                 rgcn_bdd_bases: int = 100,
                 num_conv_layers: int = 2,
                 num_rnn_layers: int = 1,
                 dropout: float = 0.0,
                 activation: str = 'tanh'):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.node_latest_event_time = node_latest_event_time
        self.time_interval_transform = time_interval_transform
        self.convolution = RGCN(in_dim=in_dim, out_dim=hid_dim, hid_dim=hid_dim, n_layers=num_conv_layers,
                                num_rels=num_rels, regularizer="bdd", num_bases=rgcn_bdd_bases, dropout=dropout,
                                activation=activation, layer_norm=False)
        temporal_rnn_in_dim = hid_dim
        if rnn_layer == "GRU":
            self.rnn = nn.GRU(input_size=temporal_rnn_in_dim, hidden_size=hid_dim, num_layers=num_rnn_layers,
                              batch_first=True, dropout=0.0)
        elif rnn_layer == "RNN":
            self.rnn = nn.RNN(input_size=temporal_rnn_in_dim, hidden_size=hid_dim, num_layers=num_rnn_layers,
                              batch_first=True, dropout=0.0)
        else:
            raise ValueError(f"Invalid RNN layer: {rnn_layer}")
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_G, dynamic_entity_emb, static_entity_emb, device, batch_node_indices=None):
        if batch_node_indices is None:
            batch_node_indices = batch_G.nodes().long()
        batch_G_sparse_inter_event_times = EventTimeHelper.get_sparse_inter_event_times(
            batch_G, self.node_latest_event_time[..., 0])
        EventTimeHelper.get_inter_event_times(batch_G, self.node_latest_event_time[..., 0], update_latest_event_time=True)

        rev_batch_G = dgl.reverse(batch_G, copy_ndata=True, copy_edata=True)
        rev_batch_G.num_relations = batch_G.num_relations
        rev_batch_G.num_all_nodes = batch_G.num_all_nodes
        rev_batch_G_sparse_inter_event_times = EventTimeHelper.get_sparse_inter_event_times(
            rev_batch_G, self.node_latest_event_time[..., 1])
        EventTimeHelper.get_inter_event_times(rev_batch_G, self.node_latest_event_time[..., 1], update_latest_event_time=True)
        batch_temporal_static_entity_emb = static_entity_emb.temporal[batch_G.ndata[dgl.NID].long()].to(device)
        edge_norm = (1 / self.time_interval_transform(batch_G_sparse_inter_event_times).clamp(min=1e-10)).clamp(
            max=10.0)
        batch_G_conv_temporal_static_emb = self.convolution(batch_G, batch_temporal_static_entity_emb,
                                                            batch_G.edata['rel_type'].long(), edge_norm)
        temporal_rnn_input_batch_G = torch.cat([batch_G_conv_temporal_static_emb], dim=1)[batch_node_indices].unsqueeze(
            1)
        rev_batch_temporal_static_entity_emb = static_entity_emb.temporal[rev_batch_G.ndata[dgl.NID].long()].to(device)
        rev_edge_norm = (1 / self.time_interval_transform(rev_batch_G_sparse_inter_event_times).clamp(min=1e-10)).clamp(
            max=10.0)
        rev_batch_G_conv_temporal_static_emb = self.convolution(rev_batch_G, rev_batch_temporal_static_entity_emb,
                                                                batch_G.edata['rel_type'].long(), rev_edge_norm)
        temporal_rnn_input_rev_batch_G = torch.cat([rev_batch_G_conv_temporal_static_emb], dim=1)[
            batch_node_indices].unsqueeze(1)
        # NOTE: dynamic_entity_emb is now expected to be on the correct device already
        temporal_dynamic = dynamic_entity_emb.temporal[batch_G.ndata[dgl.NID][batch_node_indices].long()]
        temporal_dynamic_batch_G = temporal_dynamic[..., 0]
        temporal_dynamic_rev_batch_G = temporal_dynamic[..., 1]
        output, hn = self.rnn(temporal_rnn_input_batch_G, temporal_dynamic_batch_G.transpose(0, 1).contiguous())
        updated_temporal_dynamic_batch_G = hn.transpose(0, 1)
        output, hn = self.rnn(temporal_rnn_input_rev_batch_G, temporal_dynamic_rev_batch_G.transpose(0, 1).contiguous())
        updated_temporal_dynamic_rev_batch_G = hn.transpose(0, 1)
        updated_temporal_dynamic_entity_emb = torch.cat([updated_temporal_dynamic_batch_G.unsqueeze(-1),
                                                         updated_temporal_dynamic_rev_batch_G.unsqueeze(-1)], dim=-1)
        return updated_temporal_dynamic_entity_emb


class UpdateEmbedding(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 num_rels: int,
                 node_latest_event_time: torch.Tensor = None,
                 in_dim: int = 200,
                 structural_hid_dim: int = 200,
                 temporal_hid_dim: int = 200,
                 structural_RNN: str = 'RNN',
                 temporal_RNN: str = 'RNN',
                 num_gconv_layers: int = 2,
                 rgcn_bdd_bases: int = 100,
                 num_rnn_layers: int = 1,
                 time_interval_transform=None,
                 dropout: float = 0.2,
                 activation: str = 'tanh',
                 decay_factor: float = 0.99,
                 device: torch.device = torch.device('cpu')):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.in_dim = in_dim
        self.structural_hid_dim = structural_hid_dim
        self.temporal_hid_dim = temporal_hid_dim
        self.node_latest_event_time = node_latest_event_time
        self.state_decay = NodeStateDecay(decay_factor=decay_factor)
        self.device = device
        self.graph_structural_convolution = GraphStructuralRNNConv(
            rnn_layer=structural_RNN, num_conv_layers=num_gconv_layers,
            num_rnn_layers=num_rnn_layers, num_nodes=num_nodes, num_rels=num_rels,
            in_dim=in_dim, hid_dim=structural_hid_dim, rgcn_bdd_bases=rgcn_bdd_bases,
            dropout=dropout, activation=activation
        )
        self.graph_temporal_convolution = GraphTemporalRNNConv(
            rnn_layer=temporal_RNN, num_conv_layers=num_gconv_layers,
            num_rnn_layers=num_rnn_layers, num_nodes=num_nodes, num_rels=num_rels,
            in_dim=in_dim, hid_dim=temporal_hid_dim, rgcn_bdd_bases=rgcn_bdd_bases,
            node_latest_event_time=node_latest_event_time,
            time_interval_transform=time_interval_transform,
            dropout=dropout, activation=activation
        )

    def forward(self,
                batch_G: dgl.graph,
                static_entity_emb: MultiAspectEmbedding,
                dynamic_entity_emb: MultiAspectEmbedding,
                batch_node_indices=None):

        # FIXED: Now dynamic embeddings can be on GPU from the start
        device = self.device

        # 1. Move dynamic embeddings to the target device for processing (if not already there)
        dynamic_structural_gpu = dynamic_entity_emb.structural.to(device)
        dynamic_temporal_gpu = dynamic_entity_emb.temporal.to(device)
        dynamic_entity_emb_gpu = MultiAspectEmbedding(structural=dynamic_structural_gpu, temporal=dynamic_temporal_gpu)

        # 2. Apply state decay to the entire graph's state on the GPU
        dynamic_entity_emb_decayed_gpu = self.state_decay(dynamic_entity_emb_gpu, static_entity_emb)

        # 3. Compute updates for active nodes using the decayed state
        batch_G = batch_G.to(device)
        if batch_node_indices is None:
            batch_node_indices = batch_G.nodes().long()

        # The convolution modules now receive the decayed GPU tensor
        batch_structural_dynamic_entity_emb = self.graph_structural_convolution(
            batch_G=batch_G, dynamic_entity_emb=dynamic_entity_emb_decayed_gpu, static_entity_emb=static_entity_emb,
            device=device, batch_node_indices=batch_node_indices)

        batch_temporal_dynamic_entity_emb = self.graph_temporal_convolution(
            batch_G=batch_G, dynamic_entity_emb=dynamic_entity_emb_decayed_gpu, static_entity_emb=static_entity_emb,
            device=device, batch_node_indices=batch_node_indices)

        # 4. FIXED: Keep everything on GPU - don't move back to CPU
        # Start with the decayed state for ALL nodes
        updated_structural = dynamic_entity_emb_decayed_gpu.structural.clone()
        updated_temporal = dynamic_entity_emb_decayed_gpu.temporal.clone()

        # Overwrite the embeddings for the ACTIVE nodes with their newly computed state
        if batch_structural_dynamic_entity_emb is not None:
            updated_structural[batch_G.ndata[dgl.NID][batch_node_indices].long()] = batch_structural_dynamic_entity_emb
        if batch_temporal_dynamic_entity_emb is not None:
            updated_temporal[batch_G.ndata[dgl.NID][batch_node_indices].long()] = batch_temporal_dynamic_entity_emb

        updated_dynamic_entity_emb = MultiAspectEmbedding(structural=updated_structural, temporal=updated_temporal)

        return updated_dynamic_entity_emb


    def forwardOLD(self,
                batch_G: dgl.graph,
                static_entity_emb: MultiAspectEmbedding,
                dynamic_entity_emb: MultiAspectEmbedding,
                batch_node_indices=None):

        # The main dynamic embedding tensor lives on the CPU.
        # assert all([emb.device == torch.device('cpu') for emb in dynamic_entity_emb]), \
        #     "Initial dynamic embeddings must be on CPU"

        # =====================================================================
        # MODIFIED: State decay logic at the beginning of the forward pass
        # =====================================================================
        device = self.device

        # 1. Move dynamic embeddings to the target device for processing
        dynamic_structural_gpu = dynamic_entity_emb.structural.to(device)
        dynamic_temporal_gpu = dynamic_entity_emb.temporal.to(device)
        dynamic_entity_emb_gpu = MultiAspectEmbedding(structural=dynamic_structural_gpu, temporal=dynamic_temporal_gpu)

        # 2. Apply state decay to the entire graph's state on the GPU
        dynamic_entity_emb_decayed_gpu = self.state_decay(dynamic_entity_emb_gpu, static_entity_emb)

        # 3. Compute updates for active nodes using the decayed state
        batch_G = batch_G.to(device)
        if batch_node_indices is None:
            batch_node_indices = batch_G.nodes().long()

        # The convolution modules now receive the decayed GPU tensor
        batch_structural_dynamic_entity_emb = self.graph_structural_convolution(
            batch_G=batch_G, dynamic_entity_emb=dynamic_entity_emb_decayed_gpu, static_entity_emb=static_entity_emb,
            device=device, batch_node_indices=batch_node_indices)

        batch_temporal_dynamic_entity_emb = self.graph_temporal_convolution(
            batch_G=batch_G, dynamic_entity_emb=dynamic_entity_emb_decayed_gpu, static_entity_emb=static_entity_emb,
            device=device, batch_node_indices=batch_node_indices)

        # 4. Construct the final updated tensor on the CPU
        # Start with the decayed state for ALL nodes
        updated_structural = dynamic_entity_emb_decayed_gpu.structural.cpu()
        updated_temporal = dynamic_entity_emb_decayed_gpu.temporal.cpu()

        # Overwrite the embeddings for the ACTIVE nodes with their newly computed state
        if batch_structural_dynamic_entity_emb is not None:
            updated_structural[
                batch_G.ndata[dgl.NID][batch_node_indices].long()] = batch_structural_dynamic_entity_emb.cpu()
        if batch_temporal_dynamic_entity_emb is not None:
            updated_temporal[
                batch_G.ndata[dgl.NID][batch_node_indices].long()] = batch_temporal_dynamic_entity_emb.cpu()

        updated_dynamic_entity_emb = MultiAspectEmbedding(structural=updated_structural, temporal=updated_temporal)

        return updated_dynamic_entity_emb


class Combiner(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 dropout: float = 0.2,):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(in_dim, out_dim),
                                  nn.ReLU(),
                                  nn.Dropout(dropout))

    def combine(self, dynamic_entity_emb):
        """
        Combine static and dynamic embeddings.
        This method should be overridden by subclasses to implement specific combination logic.
        """
        temporal = dynamic_entity_emb.temporal
        structural = dynamic_entity_emb.structural
        structural = structural.contiguous()[:, -1, :]
        temporal = temporal.contiguous()[:, -1, :, :].view(temporal.size(0), -1)
        combined = torch.cat([structural, temporal], dim=-1)
        return combined

    def forward(self, dynamic_entity_emb):
        """
        Combine static and dynamic embeddings.
        This method should be overridden by subclasses to implement specific combination logic.
        """
        # assert all([emb.device == torch.device('cpu') for emb in dynamic_entity_emb])
        combined = self.combine(dynamic_entity_emb)
        combined = self.head(combined)
        return combined

def initialise_embeddings(num_nodes, embedding_dim, num_rnn_layers, device=None):
    # Create tensors directly on the target device to avoid .to() breaking nn.Parameter
    static_structural = nn.Parameter(torch.zeros(num_nodes, embedding_dim, device=device))
    static_temporal = nn.Parameter(torch.zeros(num_nodes, embedding_dim, device=device))
    nn.init.xavier_uniform_(static_structural, gain=nn.init.calculate_gain('relu'))
    nn.init.xavier_uniform_(static_temporal, gain=nn.init.calculate_gain('relu'))

    dynamic_structural = nn.Parameter(torch.zeros(num_nodes, num_rnn_layers, embedding_dim, device=device))
    dynamic_temporal = nn.Parameter(torch.zeros(num_nodes, num_rnn_layers, embedding_dim, 2, device=device))

    static = MultiAspectEmbedding(structural=static_structural, temporal=static_temporal)
    dynamic = MultiAspectEmbedding(structural=dynamic_structural, temporal=dynamic_temporal)
    return static, dynamic


class GraphModel(nn.Module):
    def __init__(self,
                 num_nodes: int,
                 num_rels: int,
                 node_latest_event_time: torch.Tensor = None,
                 in_dim: int = 100,
                 structural_hid_dim: int = 100,
                 temporal_hid_dim: int = 100,
                 structural_RNN: str = 'RNN',
                 temporal_RNN: str = 'RNN',
                 num_gconv_layers: int = 2,
                 rgcn_bdd_bases: int = 20,
                 num_rnn_layers: int = 2,
                 time_interval_log_transform: bool = False,
                 dropout: float = 0.2,
                 activation: str = 'tanh',
                 decay_factor: float = 0.8,
                 head_dropout: float = 0.2,
                 out_dim: int = 20,
                 device: torch.device = torch.device('cpu'),
                 **kwargs):
        super().__init__()
        time_interval_transform = TimeIntervalTransform(log_transform=time_interval_log_transform)
        self.embedding_updater = UpdateEmbedding(
            num_nodes=num_nodes, num_rels=num_rels, node_latest_event_time=node_latest_event_time, in_dim=in_dim,
            structural_hid_dim=structural_hid_dim, temporal_hid_dim=temporal_hid_dim, structural_RNN=structural_RNN,
            temporal_RNN=temporal_RNN, num_gconv_layers=num_gconv_layers, rgcn_bdd_bases=rgcn_bdd_bases,
            num_rnn_layers=num_rnn_layers, time_interval_transform=time_interval_transform, dropout=dropout,
            activation=activation, decay_factor=decay_factor, device=device)
        self.combiner = Combiner(
            in_dim=structural_hid_dim + temporal_hid_dim * 2, out_dim=out_dim, dropout=head_dropout)
        self.node_latest_event_time = node_latest_event_time

    def forward(self,
                batch_G: dgl.graph,
                static_entity_emb: MultiAspectEmbedding,
                dynamic_entity_emb: MultiAspectEmbedding):
        return self.embedding_updater(batch_G, static_entity_emb, dynamic_entity_emb)

    def combine(self,dynamic_entity_emb):
        """
        Combine static and dynamic embeddings.
        This method should be overridden by subclasses to implement specific combination logic.
        """
        return self.combiner(dynamic_entity_emb)

