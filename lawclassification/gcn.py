import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from dataset.dataset_graph import deep_graph
import numpy as np
from tqdm import tqdm
from torch.nn import ModuleList

from datetime import datetime



class Text_GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Text_GCN,self).__init__()

        #self.convs = ModuleList(
        #    [GCNConv(in_channels, hidden_channels,add_self_loops=False,normalize=False),
        #    GCNConv(hidden_channels, out_channels,add_self_loops=False,normalize=False)]
        #)

        self.conv1 = GCNConv(in_channels, hidden_channels,add_self_loops=False,normalize=False)
        self.conv2 = GCNConv(hidden_channels, out_channels,add_self_loops=False,normalize=False)

    def forward(self, x, edge_index, edge_weight):
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x,p=0.5,training=self.training)
        x = self.conv2(x, edge_index, edge_weight)

        return x

def train(model,optimizer,loader,device):

    model.train()
    total_correct = total_loss = total_examples = 0

    for batch in loader:

        batch = batch.to(device)
        #debug:
        print(batch)

        optimizer.zero_grad()

        out = model(batch.x.to_torch_sparse_coo_tensor(), batch.edge_index, batch.edge_weight)
        loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()

        total_correct += int((out.argmax(dim=-1) == batch.y).sum())
        total_loss += float(loss) * len(batch.y[batch.train_mask])
        total_examples += len(batch.y[batch.train_mask])

    return total_correct / total_examples, total_loss / total_examples


def test(model,loader,device):

    model.eval()
    total_correct = total_examples = 0
    with torch.no_grad():

        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x.to_torch_sparse_coo_tensor(), batch.edge_index, batch.edge_weight)
            total_correct += int((out[batch.test_mask].argmax(dim=-1) == batch.y[batch.test_mask]).sum())
            total_examples += len(batch.y[batch.test_mask])
        
    return total_correct / total_examples

def val(model,loader,device):

    model.eval()
    total_correct = total_examples = 0
    with torch.no_grad():

        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x.to_torch_sparse_coo_tensor(), batch.edge_index, batch.edge_weight)
            total_correct += int((out[batch.val_mask].argmax(dim=-1) == batch.y[batch.val_mask]).sum())
            total_examples += len(batch.y[batch.val_mask])
        
    return total_correct / total_examples

def main(dataname:str,hidden_channels:int,lr:float,epochs:int) -> None:

    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    graph = deep_graph(dataname,batch_size=1)

    model = Text_GCN(graph.dataset.num_features, hidden_channels, graph.dataset.num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):

        train_acc,loss = train(model,optimizer,graph.ClusterLoaderTrain,device)
        test_acc = test(model,graph.ClusterLoaderTrain,device)
        val_acc = val(model,graph.ClusterLoaderTrain,device)

        print(f'Epoch: {epoch} Loss: {round(loss,4)} Train: {round(train_acc,4)} Val: {round(val_acc,4)} Test: {round(test_acc,4)}')

if __name__ == '__main__':
    main(dataname='twitter_chines',
         hidden_channels=200,
         lr=0.02,
         epochs=200)