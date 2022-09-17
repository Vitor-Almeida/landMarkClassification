import torch
from torch_geometric.nn import SAGEConv,global_mean_pool,GCNConv,SGConv
import torch.nn.functional as F
from dataset.dataset_graph import deep_graph

class Text_GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, add_self_loops=False, normalize=False)
        self.conv2 = GCNConv(hidden_channels, out_channels, add_self_loops=False, normalize=False)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5)
        x = self.conv2(x, edge_index, edge_weight)

        return x

def train(model,optimizer,loader,device):

    model.train()
    total_correct = total_loss = total_examples = 0

    for batch in loader:

        batch = batch.to(device)
        #debug:
        #print(batch)

        optimizer.zero_grad()

        y = batch.y[:batch.batch_size]
        y_hat = model(batch.x, batch.edge_index, batch.edge_weight)[:batch.batch_size]
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch.batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch.batch_size

    return total_correct / total_examples, total_loss / total_examples


def test(model,loader,device):

    model.eval()
    total_correct = total_examples = 0
    with torch.no_grad():

        for batch in loader:
            batch = batch.to(device)
            #out = model(batch.x.to_torch_sparse_coo_tensor(), batch.edge_index, batch.edge_weight)
            out = model(batch.x, batch.edge_index, batch.edge_weight)
            total_correct += int((out.argmax(dim=-1) == batch.y).sum())
            total_examples += len(batch.y)
        
    return total_correct / total_examples

def val(model,loader,device):

    model.eval()
    total_correct = total_examples = 0
    with torch.no_grad():

        for batch in loader:
            batch = batch.to(device)
            #out = model(batch.x.to_torch_sparse_coo_tensor(), batch.edge_index, batch.edge_weight)
            out = model(batch.x, batch.edge_index, batch.edge_weight)
            total_correct += int((out.argmax(dim=-1) == batch.y).sum())
            total_examples += len(batch.y)
        
    return total_correct / total_examples

def main(dataname:str,hidden_channels:int,lr:float,epochs:int) -> None:

    torch.cuda.empty_cache()
    #print(torch.cuda.memory_summary())
    #print(torch.cuda.memory_allocated())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    graph = deep_graph(dataname,batch_size=64, device=device)

    model = Text_GCN(graph.dataset.num_features, hidden_channels, graph.dataset.num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)

    best_test = best_val = 0

    for epoch in range(1, epochs + 1):

        train_acc,loss = train(model,optimizer,graph.graphLoaderTrain,device)
        test_acc = test(model,graph.graphLoaderTest,device)
        val_acc = test(model,graph.graphLoaderVal,device)

        if test_acc > best_test:
            best_test = test_acc
            best_val = val_acc

        print(f'Epoch: {epoch} Loss: {round(loss,4)} Train: {round(train_acc,4)} Val: {round(best_val,4)} Test: {round(best_test,4)}')

if __name__ == '__main__':
    main(dataname='twitter_chines',
         hidden_channels=256,
         lr=0.02,
         epochs=1000)