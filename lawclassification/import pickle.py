import pickle
import torch
from itertools import repeat
from torch_sparse import coalesce as coalesce_fn
from torch_geometric.utils import remove_self_loops

path = '/home/jaco/Projetos/landMarkClassification/data/Planetoid/Cora/raw/ind.cora.graph'

with open(path, 'rb') as f:
        out = pickle.load(f)
list = []

row, col = [], []
for key, value in out.items():
    row += repeat(key, len(value))
    col += value
    print('cu')
edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)
#edge_indx = list.append([row,col])
if True:
    # NOTE: There are some duplicated edges and self loops in the datasets.
    #       Other implementations do not remove them!
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = coalesce_fn(edge_index, None, 2708, 2708)