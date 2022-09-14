import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from dataset.dataset_graph import deeep_graph
import numpy as np

class Text_GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Text_GCN,self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels,add_self_loops=False,normalize=False) #add_self_loops=False ja colocou no dataset
        self.conv2 = GCNConv(hidden_channels, out_channels,add_self_loops=False,normalize=False) #add_self_loops=False ja colocou no dataset

    def forward(self, x, edge_index, edge_weight):
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x,p=0.5,training=self.training)
        x = self.conv2(x, edge_index, edge_weight)

        return x

def train(model,optimizer,data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_weight)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)

def test(model,data):

    with torch.no_grad():
        model.eval()

        pred = model(data.x, data.edge_index, data.edge_weight).argmax(dim=-1)

        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))

    return accs

def main(dataname:str,hidden_channels:int,lr:float,epochs:int) -> None:

    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    graph = deeep_graph(dataname,device)

    model = Text_GCN(graph.dataset.num_features, hidden_channels, graph.dataset.num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        loss = train(model,optimizer,graph.dataset)
        train_acc, val_acc, tmp_test_acc = test(model,graph.dataset)
        print(f'Epoch: {epoch} Loss: {round(loss,4)} Train: {round(train_acc,4)} Val: {round(val_acc,4)} Test: {round(tmp_test_acc,4)}')

if __name__ == '__main__':
    main(dataname='r8_chines',
         hidden_channels=200,
         lr=0.02,
         epochs=200)