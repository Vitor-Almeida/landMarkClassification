import os
import torch
from torch_geometric.nn import GCNConv
import pandas as pd
from utils.definitions import ROOT_DIR
import torch.nn.functional as F
from torch_geometric.logging import log
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import HeteroData
from torch_geometric.data import Data
import pandas as pd
import numpy as np

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True,
                             normalize=True,bias=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True,
                             normalize=True,bias=True)

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
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

def dataload(dataname:str):

    #data.x: Node feature matrix with shape [num_nodes, num_node_features]
    #data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
    #data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
    #data.y: Target to train against (may have arbitrary shape), e.g., node-level targets of shape [num_nodes, *] or graph-level targets of shape [1, *]
    #data.pos: Node position matrix with shape [num_nodes, num_dimensions]

    torch.cuda.empty_cache()

    pathFile = os.path.join(ROOT_DIR,'data',dataname,'interm','graph.csv')
    rawData = pd.read_csv(pathFile,dtype={'src':str,'tgt':str,'weight':float,'labels':int,'split':str})
    rawData = rawData.sample(frac=0.9)

    #for faster testing:
    #rawData = rawData.head(n=10000)

    #rawData = rawData.sample(frac=1)
    #rawData = rawData.sample(frac=0.1)

    #srcNodes = 
    #tgtNodes = np.array(rawData['tgt'],dtype=str)
    allNodes = np.array(rawData['src'],dtype=str)
    wgtEdges = np.array(rawData['weight'])
    labels = np.array(rawData['labels'])
    splits = np.array(rawData['split'])

    allUniqueNodes = np.unique(allNodes)
    #allUniqueLabels = np.unique(labels[np.where(labels!=999)])
    allUniqueLabels = np.unique(labels)

    numClasses = len(allUniqueLabels)

    allUniquePairsNodeSplit = np.unique(rawData[['src','split']].to_numpy(dtype=str),axis=0)
    allUniquePairsNodeLabels = np.unique(rawData[['src','labels']].to_numpy(dtype=str),axis=0)
    labelToNodes = allUniquePairsNodeLabels[:,1].astype('int')

    #trainMask = [True for node in allNodes if node in trainMask]
    trainMask = [True if label=='train' else False for label in allUniquePairsNodeSplit[:,1]]
    valMask = [True if label=='val' else False for label in allUniquePairsNodeSplit[:,1]]
    testMask = [True if label=='test' else False for label in allUniquePairsNodeSplit[:,1]]

    #encoding dos nodes, nome=>numero    id2labelNodes = {id:label for id,label in enumerate(allUniqueNodes.tolist())}
    label2idNodes = {label:id for id,label in enumerate(allUniqueNodes.tolist())}

    allUniqueNodesId = np.array([label2idNodes[idx] for idx in allUniqueNodes.tolist()])

    #encoding das labels, 'int'=>0..n
    id2label = {id:label for id,label in enumerate(allUniqueLabels.tolist())}
    label2id = {label:id for id,label in enumerate(allUniqueLabels.tolist())}

    srcNodesId = np.array([label2idNodes[idx] for idx in allNodes.tolist()])
    labelId = np.array([label2id[idx] for idx in labels.tolist()])

    oneHotMtx = np.zeros((allUniqueNodesId.size,allUniqueNodesId.size))
    oneHotMtx[np.arange(allUniqueNodesId.size),allUniqueNodesId] = 1
    edgeIndex = np.concatenate((np.expand_dims(srcNodesId,0),np.expand_dims(labelId,0)),axis=0)

    oneHotMtx = torch.tensor(oneHotMtx,dtype=torch.float32)
    edgeIndex = torch.tensor(edgeIndex,dtype=torch.long)
    wgtEdges = torch.tensor(wgtEdges,dtype=torch.float32)
    labels = torch.tensor(labelToNodes,dtype=torch.long)
    #lablesUnique = torch.tensor(allUniqueLabels,dtype=torch.long)
    trainMask = torch.tensor(trainMask,dtype=torch.bool)
    valMask = torch.tensor(valMask,dtype=torch.bool)
    testMask = torch.tensor(testMask,dtype=torch.bool)

    dataset = Data(x=oneHotMtx,
                   edge_index=edgeIndex,
                   edge_weight=wgtEdges,
                   y=labels,
                   test_mask=testMask,
                   train_mask=trainMask,
                   val_mask=valMask)

    return dataset,numClasses


def main(dataname:str) -> None:

    
    dataset,num_classes = dataload(dataname)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #params:
    hidden_channels = 16
    lr = 0.001
    epochs = 1000

    model = GCN(dataset.num_features, hidden_channels, num_classes)
    model, data = model.to(device), dataset.to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=0)
    ], lr=lr)  # Only perform weight-decay on first convolution.

    best_val_acc = final_test_acc = 0
    for epoch in range(1, epochs + 1):
        loss = train(model,optimizer,dataset)
        train_acc, val_acc, tmp_test_acc = test(model,data)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)


if __name__ == '__main__':
    main(dataname='ohsumed')