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

        #out = model(batch.x.to_torch_sparse_coo_tensor(), batch.edge_index, batch.edge_weight, batch.batch, batch.root_n_id)
        out = model(batch.x, batch.edge_index, batch.edge_weight)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()

        total_correct += int((out.argmax(dim=-1) == batch.y).sum())
        total_loss += float(loss) * len(batch.y)
        total_examples += len(batch.y)

    return total_correct / total_examples, total_loss / total_examples


def test(model,loader):

    model.eval()
    total_correct = total_examples = 0
    with torch.no_grad():

        #out = model(loader.x.to_torch_sparse_coo_tensor(), loader.edge_index, loader.edge_weight)
        out = model(loader.x, loader.edge_index, loader.edge_weight)
        total_correct += int((out[loader.test_mask].argmax(dim=-1) == loader.y[loader.test_mask]).sum())
        total_examples += len(loader.y[loader.test_mask])
        
    return total_correct / total_examples

def val(model,loader):

    model.eval()
    total_correct = total_examples = 0
    with torch.no_grad():

        #out = model(loader.x.to_torch_sparse_coo_tensor(), loader.edge_index, loader.edge_weight)
        out = model(loader.x, loader.edge_index, loader.edge_weight)
        total_correct += int((out[loader.val_mask].argmax(dim=-1) == loader.y[loader.val_mask]).sum())
        total_examples += len(loader.y[loader.val_mask])
        
    return total_correct / total_examples

def main(dataname:str,hidden_channels:int,lr:float,epochs:int) -> None:

    torch.cuda.empty_cache()
    #print(torch.cuda.memory_summary())
    #print(torch.cuda.memory_allocated())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    graph = deep_graph(dataname,batch_size=64, device=device)

    model = Text_GCN(graph.dataset.num_features, hidden_channels, graph.dataset.num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_test = best_val = 0

    for epoch in range(1, epochs + 1):

        train_acc,loss = train(model,optimizer,graph.graphLoaderTrain,device)
        test_acc = test(model,graph.graphLoaderValTest)
        val_acc = val(model,graph.graphLoaderValTest)

        if test_acc > best_test:
            best_test = test_acc
            best_val = val_acc

        print(f'Epoch: {epoch} Loss: {round(loss,4)} Train: {round(train_acc,4)} Val: {round(best_val,4)} Test: {round(best_test,4)}')

if __name__ == '__main__':
    main(dataname='r8_chines',
         hidden_channels=256,
         lr=0.02,
         epochs=500)