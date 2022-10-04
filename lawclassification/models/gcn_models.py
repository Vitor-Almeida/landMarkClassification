import torch
from torch_geometric.nn import GCNConv, BatchNorm
import torch.nn.functional as F
import torch_sparse as S

class Text_GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, device):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        #self.act = torch.nn.ModuleList()
        #self.batch_norms = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels, add_self_loops=True, normalize=True, improved=False))
        #self.act.append(torch.nn.PReLU(num_parameters=hidden_channels))
        #self.batch_norms.append(BatchNorm(hidden_channels))
        #self.convs.append(GCNConv(hidden_channels, hidden_channels, add_self_loops=False, normalize=False))
        self.convs.append(GCNConv(hidden_channels, out_channels, add_self_loops=True, normalize=True, improved=False))
        
        self.device = device

    def forward(self, x, edge_index, edge_weight):
        #for conv, batch_norm, act  in zip(self.convs[:-1],self.batch_norms, self.act):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            #x = batch_norm(x)
            #x = F.leaky_relu(x, negative_slope=0.2) #t
            x = F.relu(x)
            #x = act(x) #num_parameters
            #x = self.act[0](x)
            x = F.dropout(x, p=0.5, training=self.training)
        return self.convs[-1](x, edge_index, edge_weight)

    def inference_bertGCN(self, x_all, subgraph_loader):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        with torch.no_grad():
            for i, conv in enumerate(self.convs):
                xs = []
                for batch in subgraph_loader:

                    x = x_all[batch.n_id.to(self.device)].to(self.device)

                    x = conv(x, batch.edge_index.to(self.device), batch.edge_weight.to(self.device))
                    if i < len(self.convs) - 1:
                        #x = self.batch_norms[0](x) #only works for 1 layer
                        x = F.leaky_relu(x, negative_slope=0.2)
                        #x = x.relu_()

                    xs.append(x[:batch.batch_size].cpu())
                x_all = torch.cat(xs, dim=0)
            return x_all 


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
                        #x = self.batch_norms[0](x) #only works for 1 layer
                        x = F.leaky_relu(x, negative_slope=0.2)
                        #x = x.relu_()

                    xs.append(x[:batch.batch_size].cpu())
                x_all = torch.cat(xs, dim=0)
            return x_all 