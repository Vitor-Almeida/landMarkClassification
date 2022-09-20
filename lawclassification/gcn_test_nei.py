import copy
import os.path as osp

import torch
import torch.nn.functional as F
#from tqdm import tqdm

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.datasets import Reddit

import torch_sparse as S

from dataset.dataset_graph import deep_graph
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
#dataset = Reddit(path)

#dataset = deep_graph('ohsumed',4096,device)

f = open('/home/jaco/Projetos/landMarkClassification/data/r8_chines/pygraph.pickle','rb')
dataset = pickle.load(f)
f.close()

# Already send node features/labels to GPU for faster access during sampling:
data = dataset.to(device)#, 'x', 'y')

kwargs = {'batch_size': 256}#, 'num_workers': 6, 'persistent_workers': True}
train_loader = NeighborLoader(data, input_nodes=data.train_mask,
                              num_neighbors=[-1, 10], shuffle=True, **kwargs)

subgraph_loader = NeighborLoader(copy.copy(data), input_nodes=None,
                                 num_neighbors=[-1], shuffle=False, **kwargs)

next(iter(subgraph_loader))

# No need to maintain these features during evaluation:
del subgraph_loader.data.x, subgraph_loader.data.y
# Add global node index information.
subgraph_loader.data.num_nodes = data.num_nodes
subgraph_loader.data.n_id = torch.arange(data.num_nodes)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, add_self_loops=False, normalize=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, add_self_loops=False, normalize=False))

    def forward(self, x, edge_index, edge_weight):
        #x = S.SparseTensor.from_torch_sparse_coo_tensor(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        #pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs))
        #pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        x_all = S.SparseTensor.from_torch_sparse_coo_tensor(x_all)
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:

                x = x_all[batch.n_id.to(device)].to(device)

                if i == 0:
                    x = x.to_torch_sparse_coo_tensor()

                x = conv(x, batch.edge_index.to(device), batch.edge_weight.to(device))
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
                #pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        #pbar.close()
        return x_all


model = SAGE(dataset.num_features, 256, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)


def train(epoch):
    model.train()

    #pbar = tqdm(total=int(len(train_loader.dataset)))
    #pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = total_examples = 0
    for batch in train_loader:
        optimizer.zero_grad()
        y = batch.y[:batch.batch_size]
        y_hat = model(batch.x, batch.edge_index, batch.edge_weight)[:batch.batch_size]
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch.batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch.batch_size
        #pbar.update(batch.batch_size)
    #pbar.close()

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def test():
    model.eval()
    y_hat = model.inference(data.x, subgraph_loader).argmax(dim=-1)
    y = data.y.to(y_hat.device)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((y_hat[mask] == y[mask]).sum()) / int(mask.sum()))
    return accs

best_test_acc = best_val_acc = 0

for epoch in range(1, 200):
    loss, acc = train(epoch)
    #print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
    train_acc, val_acc, test_acc = test()

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_val_acc = val_acc

    print(f'Epoch: {epoch:02d}, Train Loss : {loss:.4f} Train: {train_acc:.4f}, Val: {best_val_acc:.4f}, Test: {best_test_acc:.4f}')