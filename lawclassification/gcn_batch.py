import os
import torch
from torch_geometric.nn import GCNConv
import pandas as pd
from utils.definitions import ROOT_DIR
import torch.nn.functional as F
from torch_geometric.logging import log
from torch_geometric.loader import DataLoader
import torch_geometric.data as data
import pandas as pd
import numpy as np

from datetime import datetime

from torch_geometric.loader import ClusterData
from torch_geometric.loader import ClusterLoader


from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import TUDataset

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels,
                             normalize=True,bias=True)
        self.conv2 = GCNConv(hidden_channels, out_channels,
                             normalize=True,bias=True)

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

def train(model,optimizer,loader,device):

    model.train()
    total_loss = total_nodes = 0

    print('começo train:', datetime.now().strftime("%H:%M:%S"))

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_weight) #<=
        loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()

        nodes = batch.train_mask.sum().item()
        total_loss += loss.item() * nodes
        total_nodes += nodes

    print('fim train:', datetime.now().strftime("%H:%M:%S"))
    return float(total_loss / total_nodes)

def test(model,data):

    #for batch in data:
    print('começo test:', datetime.now().strftime("%H:%M:%S"))

    with torch.no_grad():
        model.eval()
        pred = model(data.x, data.edge_index, data.edge_weight).argmax(dim=-1)

        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))

    print('fim test:', datetime.now().strftime("%H:%M:%S"))
    return accs

def dataload(dataname:str):

    #data.x: Node feature matrix with shape [num_nodes, num_node_features]
    #data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
    #data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
    #data.y: Target to train against (may have arbitrary shape), e.g., node-level targets of shape [num_nodes, *] or graph-level targets of shape [1, *]
    #data.pos: Node position matrix with shape [num_nodes, num_dimensions]

    #dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
    #loader = DataLoader(dataset, batch_size=700, shuffle=False)
    #for idx,batch in enumerate(loader): 
    #    print(batch,idx)


    torch.cuda.empty_cache()

    pathFile = os.path.join(ROOT_DIR,'data',dataname,'interm','graph.csv')
    rawData = pd.read_csv(pathFile,dtype={'src':str,'tgt':str,'weight':float,'labels':int,'split':str})

    #rawData = rawData[(rawData['labels']<8) | (rawData['split']=='wordword')]

    rawData = rawData.sample(frac=1)

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

    dataset = data.Data(x=oneHotMtx,
                   edge_index=edgeIndex,
                   edge_weight=wgtEdges,
                   y=labels,
                   test_mask=testMask,
                   train_mask=trainMask,
                   val_mask=valMask)

    return ClusterData(dataset,20,log=False),dataset,numClasses


def main(dataname:str) -> None:

    torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data,fullGraph,num_classes = dataload(dataname)

    #data = data.to(device)
    
    dataLoader = ClusterLoader(data,batch_size=4,shuffle=True,num_workers=8,drop_last=True)

    #dataLoader = DataLoader([data], batch_size=32, shuffle=True)

    #params:
    hidden_channels = 200
    lr = 0.02
    epochs = 200

    model = GCN(data.data.num_features, hidden_channels, num_classes)

    model = model.to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=0)
    ], lr=lr)  # Only perform weight-decay on first convolution.

    best_val_acc = final_test_acc = 0
    for epoch in range(1, epochs + 1):
        loss = train(model, optimizer, dataLoader, device)
        train_acc, val_acc, tmp_test_acc = test(model, fullGraph.to(device))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        print(f'Epoch: {epoch} Loss: {loss} Train: {train_acc} Val: {val_acc} Test: {test_acc}')
        #log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)


if __name__ == '__main__':
    main(dataname='twitter_chines')