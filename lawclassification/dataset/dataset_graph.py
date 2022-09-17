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

class deeep_graph(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        super(deep_graph,self).__init__(root, name, transform, pre_transform)

        self.name = name

    @property
    def raw_file_names(self):
        return [os.path.join(ROOT_DIR,'data',self.name,'interm','graph.csv')]

    @property
    def processed_file_names(self):
        #return [os.path.join(ROOT_DIR,'data',self.name,'interm','graphProcessed.csv')]
        return [os.path.join(ROOT_DIR,'data',self.name,'interm','graph.pt')]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = [...]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    #def len(self):
    #    return len(self.processed_file_names)

    #def get(self, idx):
    #    data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
    #    return data

class deep_graph():
    def __init__(self,dataname:str,batch_size:int,device:str):

        pathFile = os.path.join(ROOT_DIR,'data',dataname,'interm','graph.csv')
        rawData = pd.read_csv(pathFile,dtype={'src':str,'tgt':str,'weight':float,'labels':int,'split':str})
        rawData2 = pd.read_csv(pathFile,dtype={'src':str,'tgt':str,'weight':float,'labels':int,'split':str})
        rawData2.rename(columns={'tgt':'tgt_','src':'tgt'},inplace=True)
        rawData2.rename(columns={'tgt_':'src'},inplace=True)

        rawData.dropna(inplace=True)
        rawData2.dropna(inplace=True)

        rawData2 = rawData2.sample(frac=1)
        rawData = rawData.sample(frac=1)

        empty=[]
        np.testing.assert_array_equal(np.array(rawData2[(rawData2['src'].str.contains("doc_"))]['tgt']),np.array(empty))

        #add selfloops:
        dfSelfLoop = rawData2[(rawData2['tgt'].str.contains("doc_")) & (rawData2['split']!='wordword')][['tgt','labels','split']].drop_duplicates()
        dfSelfLoop['weight'] = 1.0 #try 0 here
        dfSelfLoop['src'] = dfSelfLoop['tgt']
        dfSelfLoop = dfSelfLoop[['src','tgt','weight','labels','split']]

        rawData = pd.concat([rawData,rawData2],ignore_index=True)
        #rawData = rawData.drop_duplicates()
        rawData = pd.concat([rawData.drop_duplicates(),dfSelfLoop],ignore_index=True)

        #normalize = T.NormalizeFeatures(attrs=['edge_weight'])
        #addSelfLoop = T.GCNNorm()

        allNodesSrc = np.array(rawData['src'],dtype=str)
        allNodesTgt = np.array(rawData['tgt'],dtype=str)

        np.testing.assert_array_equal(np.array(rawData['src'].sort_values()),np.array(rawData['tgt'].sort_values()))

        wgtEdges = np.array(rawData['weight'])

        labels = np.array(rawData['labels'])

        allUniqueNodes = np.unique(allNodesSrc) #AllNodesSrc e Tgt uniques are equal. undirected graph
        allUniqueLabels = np.unique(labels)

        label2idNodes = {label:id for id,label in enumerate(allUniqueNodes.tolist())}
        allUniqueNodesId = np.array([label2idNodes[idx] for idx in allUniqueNodes.tolist()])

        label2id = {label:id for id,label in enumerate(allUniqueLabels.tolist())}

        rawData2['nodeId'] = rawData2['tgt'].apply(lambda row: label2idNodes[row])
        rawData2['labels'] = rawData2['labels'].apply(lambda row: label2id[row])
        rawData2 = rawData2[['nodeId','labels','split']].drop_duplicates()
        rawData2['train_mask'] = rawData2['split'].apply(lambda row: True if row=='train' else False)
        rawData2['test_mask'] = rawData2['split'].apply(lambda row: True if row=='test' else False)
        rawData2['val_mask'] = rawData2['split'].apply(lambda row: True if row=='val' else False)

        numClasses = len(allUniqueLabels) - 1

        trainMask = rawData2['train_mask'].to_numpy(dtype=bool)
        valMask = rawData2['val_mask'].to_numpy(dtype=bool)
        testMask = rawData2['test_mask'].to_numpy(dtype=bool)
        labelToNodes = rawData2['labels'].to_numpy(dtype=int)
        allUniqueNodesId = rawData2['nodeId'].to_numpy(dtype=int)

        arr1inds = allUniqueNodesId.argsort()[::-1][:len(allUniqueNodesId)]
        trainMask = trainMask[arr1inds[::-1]]
        valMask = valMask[arr1inds[::-1]]
        testMask = testMask[arr1inds[::-1]]
        labelToNodes = labelToNodes[arr1inds[::-1]]
        allUniqueNodesId = allUniqueNodesId[arr1inds[::-1]]

        del rawData
        del rawData2
        gc.collect()

        srcNodesId = np.array([label2idNodes[idx] for idx in allNodesSrc.tolist()])
        tgtNodesId = np.array([label2idNodes[idx] for idx in allNodesTgt.tolist()])
        edgeIndex = np.concatenate((np.expand_dims(srcNodesId,0),np.expand_dims(tgtNodesId,0)),axis=0)

        edgeAttr = np.expand_dims(wgtEdges,0).T

        oneHotMtx = S.SparseTensor.eye(M=len(allUniqueNodesId),dtype=torch.float32)
        #oneHotMtx = oneHotMtx.to_dense()
        oneHotMtx = oneHotMtx.to_torch_sparse_coo_tensor()
        #oneHotMtx = oneHotMtx.to_torch_sparse_coo_tensor()

        edgeIndex = torch.tensor(edgeIndex,dtype=torch.long)
        wgtEdges = torch.tensor(wgtEdges,dtype=torch.float32)
        labels = torch.tensor(labelToNodes,dtype=torch.long)
        trainMask = torch.tensor(trainMask,dtype=torch.bool)
        valMask = torch.tensor(valMask,dtype=torch.bool)
        testMask = torch.tensor(testMask,dtype=torch.bool)

        self.dataset = data.Data(x = oneHotMtx,
                                 edge_index = edgeIndex,
                                 edge_weight = wgtEdges,
                                 #edge_attr = edgeAttr,
                                 y = labels,
                                 num_classes = numClasses,
                                 test_mask = testMask,
                                 train_mask = trainMask,
                                 val_mask = valMask)

        #bestBatchSize = len(allUniqueNodesId) / torch.mean(U.degree(self.dataset.edge_index[0]))

        #normalize(self.dataset)

        #self.graphLoaderValTest = self.dataset.subgraph(self.dataset.val_mask+self.dataset.test_mask).to(device)

        #self.ClusterLoaderTrain = ClusterLoader(ClusterData(self.dataset,num_parts=2,log=False), 
        #                                        batch_size=batch_size, 
        #                                        shuffle=True,
        #                                        drop_last=False)

        #self.graphLoaderTrain = NeighborLoader(self.dataset, 
        #                                       num_neighbors=[20,5],
        #                                       input_nodes=self.dataset.train_mask,
        #                                       batch_size = batch_size,
        #                                       #num_workers = 10,
        #                                       #persistent_workers = True,
        #                                       shuffle=True, 
        #                                       drop_last=False)#,num_workers=16

        #self.graphLoaderTest = NeighborLoader(self.dataset, 
        #                                       num_neighbors=[20,5],
        #                                       input_nodes=self.dataset.test_mask,
        #                                       batch_size = batch_size,
        #                                       #num_workers = 3,
        #                                       #persistent_workers = True,
        #                                       shuffle=True, 
        #                                       drop_last=False)#,num_workers=16

        #self.graphLoaderVal = NeighborLoader(self.dataset, 
        #                                       num_neighbors=[20,5],
        #                                       input_nodes=self.dataset.val_mask,
        #                                       batch_size = batch_size,
        #                                       #num_workers = 3,
        #                                       #persistent_workers = True,
        #                                       shuffle=True, 
        #                                       drop_last=False)#,num_workers=16

        #self.graphLoaderTrain = ShaDowKHopSampler(self.dataset, 
        #                                          depth=2, 
        #                                          num_neighbors=10,
        #                                          node_idx=self.dataset.train_mask,
        #                                          batch_size = batch_size,
        #                                          num_workers = 10,
        #                                          persistent_workers = True,
        #                                          shuffle=True, 
        #                                          drop_last=False)#,num_workers=16

        #self.graphLoaderTest = ShaDowKHopSampler(self.dataset, 
        #                                         depth=2, 
        #                                         num_neighbors=10,
        #                                         node_idx=self.dataset.test_mask,
        #                                         batch_size = batch_size,
        #                                         num_workers = 3,
        #                                         persistent_workers = True,
        #                                         shuffle=True, 
        #                                         drop_last=False)#,num_workers=16

        #self.graphLoaderVal = ShaDowKHopSampler(self.dataset, 
        #                                        depth=2, 
        #                                        num_neighbors=10,
        #                                        node_idx=self.dataset.val_mask,
        #                                        batch_size = batch_size,
        #                                        num_workers = 3,
        #                                        persistent_workers = True,
        #                                        shuffle=True, 
        #                                        drop_last=False)#,num_workers=16
        