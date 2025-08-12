from pathlib import Path

import dgl
import numpy as np
import torch
from tqdm import tqdm


from dataloader import collate_fn, load_temporal_knowledge_graph, TemporalKnowledgeGraphStatics
from news_graph import GraphModel, initialise_embeddings
import sys
# Ensure the current directory is in the path for module imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

# from utils import set_device
def set_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        print("Using CUDA backend")
        return torch.device('cuda')
    return torch.device('cpu')


def train(in_dim: int = 100,
          structural_hid_dim: int = 100,
          temporal_hid_dim: int = 100,
          structural_RNN: str = 'RNN',
          temporal_RNN: str = 'RNN',
          num_gconv_layers: int = 2,
          rgcn_bdd_bases: int = 20,
          num_rnn_layers: int = 2,
          dropout: float = 0.2,
          activation: str = 'tanh',
          decay_factor: float = 0.8,
          head_dropout: float = 0.2,
          out_dim: int = 20,
          time_interval_log_transform: bool = True,
          graph_model: str = 'GraphModel',
          seed: int = 42,
          use_cpu: bool = True,
          graph_gpu: int = -1,
          root_dir: Path = Path(__file__).resolve().parent.parent / 'data' / 'triplets',
          # train parameters
          epochs: int = 1):
    from torch.nn import MSELoss
    loss = MSELoss()
    np.random.seed(seed)
    torch.manual_seed(seed)
    graph_device = set_device()

    # Setup graph model
    G = load_temporal_knowledge_graph(root_dir=root_dir)
    G.to(graph_device)
    graph_statics = TemporalKnowledgeGraphStatics(root_dir=root_dir)
    date2idx = graph_statics.date2idx
    idx2date = graph_statics.idx2date

    # FIXED: Create node_latest_event_time directly on the GPU device
    node_latest_event_time = torch.zeros(
        (G.number_of_nodes(), G.number_of_nodes() + 1, 2),
        dtype=torch.float32,
        device=graph_device  # Create directly on GPU
    )

    graph_parameters = dict(num_nodes=G.number_of_nodes,
                            num_rels=G.num_relations,
                            in_dim=in_dim,
                            structural_hid_dim=structural_hid_dim,
                            temporal_hid_dim=temporal_hid_dim,
                            structural_RNN=structural_RNN,
                            temporal_RNN=temporal_RNN,
                            num_gconv_layers=num_gconv_layers,
                            rgcn_bdd_bases=rgcn_bdd_bases,
                            num_rnn_layers=num_rnn_layers,
                            dropout=dropout,
                            activation=activation,
                            decay_factor=decay_factor,
                            head_dropout=head_dropout,
                            out_dim=out_dim,
                            time_interval_log_transform=time_interval_log_transform,
                            graph_model=graph_model)

    if graph_model == 'GraphModel':
        graph_model = GraphModel(node_latest_event_time=node_latest_event_time, device=graph_device, **graph_parameters)
    else:
        raise NotImplementedError(f'Unknown graph model: {graph_model}')

    graph_model.to(graph_device)

    # FIXED: Create embeddings directly on GPU device
    static_emb, init_dynamic_emb = initialise_embeddings(
        num_nodes=G.number_of_nodes(), embedding_dim=100, num_rnn_layers=2, device=graph_device)

    params = list(graph_model.parameters()) + [
        static_emb.structural, static_emb.temporal, init_dynamic_emb.structural, init_dynamic_emb.temporal
    ]
    optimizer = torch.optim.Adam(params, lr=0.001)

    for epoch in range(epochs):
        graph_model.train()
        optimizer.zero_grad()

        # FIXED: Reset tensors on GPU
        graph_model.node_latest_event_time.zero_()
        node_latest_event_time.zero_()
        dynamic_emb = init_dynamic_emb
        pbar = tqdm(date2idx, desc=f"Training Epoch {epoch}")

        for i, date in enumerate(pbar):
            idx = date2idx[date]
            batch_G, date = collate_fn([(idx, date), ], G)
            if batch_G is None:
                continue
            dynamic_emb = graph_model(batch_G=batch_G, static_entity_emb=static_emb, dynamic_entity_emb=dynamic_emb)
            combined = graph_model.combine(dynamic_entity_emb=dynamic_emb)
            if (i + 1) % 10 == 0:
                optimizer.zero_grad()

                l = loss(combined, torch.randn_like(combined).to(device=graph_device))
                pbar.set_postfix({'loss': l.item()})
                l.backward()
                optimizer.step()
                # Detach embeddings to prevent gradient accumulation across time steps
                dynamic_emb = type(dynamic_emb)(
                    structural=dynamic_emb.structural.detach(),
                    temporal=dynamic_emb.temporal.detach()
                )

if __name__ == "__main__":
    train()