import os
import torch
import pandas as pd
import gc
import numpy as np
import torch_geometric.data as data
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch_geometric.utils as U
from torch_geometric.loader import ClusterLoader,NeighborLoader,ShaDowKHopSampler,ClusterData

import torch_sparse as S

#from torch_sparse import coalesce 

#from torch_geometric.data import Dataset
from torch_geometric.data import InMemoryDataset
from utils.definitions import ROOT_DIR

class deep_graph_full():
    def __init__(self,dataname:str,batch_size:int,device):

        pathFile = os.path.join(ROOT_DIR,'data',dataname,'interm','graph.csv')
        rawData = pd.read_csv(pathFile,dtype={'src':str,'tgt':str,'weight':float,'labels':int,'split':str})

        rawData.dropna(inplace=True)

        rawData = rawData.sample(frac=1)

        empty=[]
        np.testing.assert_array_equal(np.array(rawData[(rawData['tgt'].str.contains("doc_"))]['src']),np.array(empty))

        allNodesSrc = np.array(rawData['src'],dtype=str)
        allNodesTgt = np.array(rawData['tgt'],dtype=str)

        wgtEdges = np.array(rawData['weight'])

        labels = np.array(rawData['labels'])

        allUniqueNodes = np.unique(allNodesSrc) #in our "undirected" graph the [src] have all the nodes.
        allUniqueLabels = np.unique(labels)

        label2idNodes = {label:id for id,label in enumerate(allUniqueNodes.tolist())}
        allUniqueNodesId = np.array([label2idNodes[idx] for idx in allUniqueNodes.tolist()])

        label2id = {label:id for id,label in enumerate(allUniqueLabels.tolist())}

        rawData['nodeId'] = rawData['src'].apply(lambda row: label2idNodes[row])
        rawData['labels'] = rawData['labels'].apply(lambda row: label2id[row])
        rawData = rawData[['nodeId','labels','split']].drop_duplicates()
        rawData['train_mask'] = rawData['split'].apply(lambda row: True if row=='train' else False)
        rawData['test_mask'] = rawData['split'].apply(lambda row: True if row=='test' else False)
        rawData['val_mask'] = rawData['split'].apply(lambda row: True if row=='val' else False)

        numClasses = len(allUniqueLabels) - 1

        trainMask = rawData['train_mask'].to_numpy(dtype=bool)
        valMask = rawData['val_mask'].to_numpy(dtype=bool)
        testMask = rawData['test_mask'].to_numpy(dtype=bool)
        labelToNodes = rawData['labels'].to_numpy(dtype=int)
        allUniqueNodesId = rawData['nodeId'].to_numpy(dtype=int)

        arr1inds = allUniqueNodesId.argsort()[::-1][:len(allUniqueNodesId)]
        trainMask = trainMask[arr1inds[::-1]]
        valMask = valMask[arr1inds[::-1]]
        testMask = testMask[arr1inds[::-1]]
        labelToNodes = labelToNodes[arr1inds[::-1]]
        allUniqueNodesId = allUniqueNodesId[arr1inds[::-1]]

        del rawData
        gc.collect()

        srcNodesId = np.array([label2idNodes[idx] for idx in allNodesSrc.tolist()])
        tgtNodesId = np.array([label2idNodes[idx] for idx in allNodesTgt.tolist()])
        edgeIndex = np.concatenate((np.expand_dims(srcNodesId,0),np.expand_dims(tgtNodesId,0)),axis=0)

        edgeAttr = np.expand_dims(wgtEdges,0).T

        oneHotMtx = S.SparseTensor.eye(M=len(allUniqueNodesId),dtype=torch.float32)
        #oneHotMtx = oneHotMtx.to_dense()
        #oneHotMtx = oneHotMtx.to_torch_sparse_coo_tensor()
        #oneHotMtx = oneHotMtx.to_torch_sparse_coo_tensor()

        edgeIndex = torch.tensor(edgeIndex,dtype=torch.long)
        wgtEdges = torch.tensor(wgtEdges,dtype=torch.float32)
        labels = torch.tensor(labelToNodes,dtype=torch.long)
        trainMask = torch.tensor(trainMask,dtype=torch.bool)
        valMask = torch.tensor(valMask,dtype=torch.bool)
        testMask = torch.tensor(testMask,dtype=torch.bool)

        uniIndex = U.to_undirected(edgeIndex,wgtEdges)

        self.dataset = data.Data(x = oneHotMtx,
                                 edge_index = uniIndex[0],
                                 edge_weight = uniIndex[1],
                                 #edge_attr = edgeAttr,
                                 y = labels,
                                 #num_nodes = len(allUniqueNodesId), #novo
                                 num_classes = numClasses,
                                 test_mask = testMask,
                                 train_mask = trainMask,
                                 val_mask = valMask).to(device)

        print('c')

        #normalize(self.dataset)

        #self.dataset.subgraph(train_mask)

        #self.ClusterLoaderTrain = ClusterLoader(ClusterData(self.dataset,num_parts=2,log=False), 
        #                                        batch_size=batch_size, 
        #                                        shuffle=True,
        #                                        drop_last=False)

        #self.graphLoaderTrain = ShaDowKHopSampler(self.dataset, 
        #                                       depth=3, 
        #                                       num_neighbors=10,
        #                                       node_idx=self.dataset.train_mask,
        #                                       batch_size = batch_size,
        #                                       num_workers = 4,
        #                                       persistent_workers = True,
        #                                       shuffle=True, 
        #                                       drop_last=False)#,num_workers=16

        #self.graphLoaderTest = ShaDowKHopSampler(self.dataset, 
        #                                       depth=3, 
        #                                       num_neighbors=10,
        #                                       node_idx=self.dataset.test_mask,
        #                                       batch_size = batch_size,
        #                                       num_workers = 4,
        #                                       persistent_workers = True,
        #                                       shuffle=True, 
        #                                       drop_last=False)#,num_workers=16

        #self.graphLoaderVal = ShaDowKHopSampler(self.dataset, 
        #                                       depth=3, 
        #                                       num_neighbors=10,
        #                                       node_idx=self.dataset.val_mask,
        #                                       batch_size = batch_size,
        #                                       num_workers = 4,
        #                                       persistent_workers = True,
        #                                       shuffle=True, 
        #                                       drop_last=False)#,num_workers=16
        