import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch_sparse as S

class Text_GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, device):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, add_self_loops=False, normalize=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, add_self_loops=False, normalize=False))
        self.device = device

    def forward(self, x, edge_index, edge_weight):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def inference(self, x_all, subgraph_loader):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        with torch.no_grad():
            x_all = S.SparseTensor.from_torch_sparse_coo_tensor(x_all)
            for i, conv in enumerate(self.convs):
                xs = []
                for batch in subgraph_loader:

                    x = x_all[batch.n_id.to(self.device)].to(self.device)

                    if i == 0:
                        x = x.to_torch_sparse_coo_tensor()

                    x = conv(x, batch.edge_index.to(self.device), batch.edge_weight.to(self.device))
                    if i < len(self.convs) - 1:
                        x = x.relu_()
                    xs.append(x[:batch.batch_size].cpu())
                x_all = torch.cat(xs, dim=0)
            return x_all