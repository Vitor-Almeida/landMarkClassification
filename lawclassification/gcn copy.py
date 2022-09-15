import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from dataset.dataset_graph import deep_graph
import numpy as np
from tqdm import tqdm
from torch.nn import ModuleList

class Text_GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Text_GCN,self).__init__()

        self.convs = ModuleList(
            [GCNConv(in_channels, hidden_channels,add_self_loops=False,normalize=False),
            GCNConv(hidden_channels, out_channels,add_self_loops=False,normalize=False)]
        )

        #self.conv1 = GCNConv(in_channels, hidden_channels,add_self_loops=False,normalize=False) #add_self_loops=False ja colocou no dataset
        #self.conv2 = GCNConv(hidden_channels, out_channels,add_self_loops=False,normalize=False) #add_self_loops=False ja colocou no dataset

    def forward1(self, x, edge_index, edge_weight):
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x,p=0.5,training=self.training)
        x = self.conv2(x, edge_index, edge_weight)

        return x

    def forward(self, x, edge_index, edge_weight):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def inference(self, x_all, subgraph_loader, device):
        #pbar = tqdm(total=x_all.size(0) * len(self.convs))
        #pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = conv((x, x_target), edge_index)
                if i != len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                #pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        #pbar.close()

        return x_all

def train(model,optimizer,loader,device):

    model.train()
    total_correct = total_loss = total_examples = 0

    #debug:
    mediaPerc = allNodes = 0

    for batch in loader:
        batch = batch.to(device)
        #print(batch)
        #debug:
        mediaPerc += batch.x.size(0)
        allNodes += batch.x.size(0)

        optimizer.zero_grad()
        out = model(batch.x.to_torch_sparse_coo_tensor(), batch.edge_index, batch.edge_weight)
        loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()

        total_correct += int((out.argmax(dim=-1) == batch.y).sum())
        total_loss += float(loss) * batch.x.size(0)
        total_examples += batch.x.size(0)

    #print(round(mediaPerc / allNodes * 100,4))
    return total_correct / total_examples, total_loss / total_examples


def test(model, graph, device):

    model.eval()
    total_correct = total_examples = 0
    with torch.no_grad():

        out = model.inference(graph.dataset.x, graph.subgraph_loader, device)
        y_pred = out.argmax(dim=-1)

        accs = []
        for mask in [graph.dataset.train_mask, graph.dataset.val_mask, graph.dataset.test_mask]:
            correct = y_pred[mask].eq(graph.dataset.y[mask]).sum().item()
            accs.append(correct / mask.sum().item())
    return accs

def val(model,loader,device):

    model.eval()
    total_correct = total_examples = 0
    with torch.no_grad():

        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x.to_torch_sparse_coo_tensor(), batch.edge_index, batch.edge_weight)
            total_correct += int((out[batch.val_mask].argmax(dim=-1) == batch.y[batch.val_mask]).sum())
            total_examples += batch.num_nodes
        
    return total_correct / total_examples

def main(dataname:str,hidden_channels:int,lr:float,epochs:int) -> None:

    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    graph = deep_graph(dataname,batch_size=32)

    model = Text_GCN(graph.dataset.num_features, hidden_channels, graph.dataset.num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        #train_acc,loss = train(model,optimizer,graph.graphLoaderTrain,device)
        train_acc,loss = train(model,optimizer,graph.ClusterLoaderTrain,device)
        test_acc = test(model,graph,device)
        val_acc = val(model,graph.ClusterLoaderTrain,device)
        #print(f'Epoch: {epoch} Loss: {round(loss,4)} Train: {round(train_acc,4)}')
        print(f'Epoch: {epoch} Loss: {round(loss,4)} Train: {round(train_acc,4)} Val: {round(val_acc,4)} Test: {round(test_acc,4)}')

if __name__ == '__main__':
    main(dataname='twitter_chines',
         hidden_channels=200,
         lr=0.02,
         epochs=200)