
from dataclasses import dataclass, field
from datetime import date
from functools import partial
from pathlib import Path
from typing import Dict

import dgl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class Config:
    """Configuration for market data processing and dataset splits."""
    file_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    data_root: Path = field(init=False)

    start_date: date = date(2021, 1, 1)
    end_date: date = date(2025, 6, 30)
    lookback: int = 30
    batch_size: int = 8
    id_type: torch.dtype = torch.int32


    def __post_init__(self):
        object.__setattr__(self, 'data_root', self.file_path.parent / 'data/triplets')


def get_edge_mask(num_edges, edge_index_from, edge_index_until):
    """
    return binary edge masks for edges from edge_index_from (inclusive) till edge_index_until (exclusive)
    """
    assert 0 <= edge_index_from < edge_index_until <= num_edges
    mask = torch.zeros(num_edges, dtype=torch.bool)
    mask[edge_index_from: edge_index_until] = True
    return mask


def comp_deg_norm(G):
    in_deg = G.in_degrees(range(G.number_of_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0, as_tuple=False).view(-1)] = 1
    norm = 1.0 / in_deg

    return norm

# noinspection PyTypeChecker
def collate_fn(batch_time, G):
    batch = batch_time[0] if isinstance(batch_time, (tuple, list)) else batch_time
    idx, date = batch
    batch_edges = torch.nonzero(G.edata['time'] == idx, as_tuple=False).squeeze().type(G.idtype)
    batch_G = dgl.edge_subgraph(G, batch_edges, relabel_nodes=True)
    if batch_G.num_edges() == 0:
        return None, date
    batch_G.num_relations = G.num_relations
    batch_G.num_all_nodes = G.num_nodes()  # preserve the structure of the original graph - all nodes are always present
    batch_G.ndata['norm'] = comp_deg_norm(batch_G)  # used by R-GCN
    return batch_G, date

def load_temporal_knowledge_graph(root_dir: Path, idtype: torch.dtype = torch.int32):
    """ Load a temporal knowledge graph from triplet files."""
    train = pd.read_table(root_dir / 'train.txt', sep='\t', names=['h', 'r', 't', 'dt'])
    val = pd.read_table(root_dir / 'val.txt', sep='\t', names=['h', 'r', 't', 'dt'])
    test = pd.read_table(root_dir / 'test.txt', sep='\t', names=['h', 'r', 't', 'dt'])
    all_data = pd.concat([train, val, test], ignore_index=True)

    num_entities = len(set(all_data['h'].tolist() + all_data['t'].tolist()))
    num_relations = len(set(all_data['r'].tolist()))

    heads = torch.from_numpy(all_data['h'].to_numpy()).type(idtype)
    tails = torch.from_numpy(all_data['t'].to_numpy()).type(idtype)
    rels = torch.from_numpy(all_data['r'].to_numpy()).type(idtype)
    times = torch.from_numpy(all_data['dt'].to_numpy()).float() # need float because it will be treated as an edge feature

    # create a dgl graph and add edge masks
    G = dgl.graph((heads, tails), num_nodes=num_entities, idtype=idtype)
    G.num_relations = num_relations
    G.edata['rel_type'] = rels
    G.edata['time'] = times
    G.train_times = np.sort(np.unique(train['dt'].to_numpy()))
    G.val_times = np.sort(np.unique(val['dt'].to_numpy()))
    G.test_times = np.sort(np.unique(test['dt'].to_numpy()))

    num_edges = len(all_data)
    train_edge_mask = get_edge_mask(num_edges, 0, len(train))
    val_edge_mask = get_edge_mask(num_edges, len(train), len(train) + len(val))
    test_edge_mask = get_edge_mask(num_edges, len(train) + len(val), num_edges)

    G.edata['train_mask'] = train_edge_mask
    G.edata['val_mask'] = val_edge_mask
    G.edata['test_mask'] = test_edge_mask
    G.train_times = np.sort(np.unique(train['dt'].to_numpy()))
    G.val_times = np.sort(np.unique(val['dt'].to_numpy()))
    G.test_times = np.sort(np.unique(test['dt'].to_numpy()))
    return G


class TemporalKnowledgeGraphStatics:
    """PyTorch Dataset for market data time series."""
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.nodes2entities: Dict[int, str] = dict()
        self.edge_types2relation: Dict[int, str] = None
        self.idx2date: Dict[int, date] = dict()
        self.date2idx: Dict[date, int] = dict()
        self.num_nodes: int = 0
        self.num_relations: int = 0
        self.__post_init()

    def __post_init(self):
        def df2dict(df: pd.DataFrame) -> dict[int, str]:
            """Convert DataFrame to dictionary mapping from index to value."""
            return {row['idx']: row['name'] for _, row in df.iterrows()}
        self._train_dates = sorted(list(set(pd.read_table(self.root_dir / 'train.txt', sep='\t', names=['h', 'r', 't', 'dt'])['dt'].to_list())))
        self._val_dates = sorted(list(set(pd.read_table(self.root_dir / 'val.txt', sep='\t', names=['h', 'r', 't', 'dt'])['dt'].to_list())))
        self._test_dates = sorted(list(set(pd.read_table(self.root_dir / 'test.txt', sep='\t', names=['h', 'r', 't', 'dt'])['dt'].tolist())))
        id2r = pd.read_table(self.root_dir / 'relations2idx.txt', sep='\t', names=['name', 'idx'])
        id2e = pd.read_table(self.root_dir / 'entities2idx.txt', sep='\t', names=['name', 'idx'])
        id2dt = pd.read_table(self.root_dir / 'dates2idx.txt', sep='\t', names=['name', 'idx'])
        id2dt['name'] = pd.to_datetime(id2dt['name'], format='%Y-%m-%d').dt.date
        self.nodes2entities = df2dict(id2e)
        self.edge_types2relation = df2dict(id2r)
        self.idx2date = df2dict(id2dt)
        self.date2idx = dict((v, k) for k, v in self.idx2date.items())

    @property
    def train_dates(self):
        return dict((i, self.idx2date[i]) for i in self._train_dates)

    @property
    def val_dates(self):
        return dict((i, self.idx2date[i]) for i in self._val_dates)

    @property
    def test_dates(self):
        return dict((i, self.idx2date[i]) for i in self._test_dates)



# noinspection PyTypeChecker
if __name__ == '__main__':
    pass