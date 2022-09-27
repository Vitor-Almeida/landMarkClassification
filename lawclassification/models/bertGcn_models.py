#https://github.com/ZeroRin/BertGCN/blob/main/model/models.py

import torch
from transformers import AutoTokenizer, AutoModel
from models.gcn_models import Text_GCN
from utils.definitions import ROOT_DIR
from torch_geometric.loader import NeighborLoader, ImbalancedSampler
import pickle
import os
import copy
from torch.utils.data import DataLoader
import random
from dataset.dataset_load import deep_data
import gc
import numpy as np

class ensemble_model():
    def __init__(self,experiment):
        super(ensemble_model, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.seedVal = random.randint(0, 1000)
        random.seed(self.seedVal)
        np.random.seed(self.seedVal)
        torch.manual_seed(self.seedVal)
        torch.cuda.manual_seed_all(self.seedVal)

        f = open(os.path.join(ROOT_DIR,'data',experiment['dataname'],'pygraph.pickle'),'rb')
        #self.dataset = pickle.load(f).to(self.device)
        self.dataset = pickle.load(f)
        f.close()

        self.dataname = experiment['dataname']
        self.model_name = experiment['model_name']
        self.epochs = int(experiment['epochs'])
        self.hiddenChannels = int(experiment['hidden_channels'])
        self.problemType = experiment['problem_type']
        self.batchSize = int(experiment['batchsize'])
        self.bert_lr = experiment['bert_lr']
        self.gcn_lr = experiment['gcn_lr']
        self.flag_hierarchical = experiment['hierarchical']
        self.max_char_length = experiment['max_char_length']
        self.neigh_param = eval(str(experiment['neigh_paramater']))
        self.hier_max_seg = int(experiment['hier_max_seg'])
        self.hier_max_seg_length = int(experiment['hier_max_seg_length'])

        self.bertGcnModel = Bert_GCN(self.model_name, self.dataset.num_classes, self.hiddenChannels, self.device, m=0.7)

        ckpt = torch.load(os.path.join(ROOT_DIR,'data',self.dataname,'interm','tuned_deep_model.pth'), map_location=self.device)
        self.bertGcnModel.bert_model.load_state_dict(ckpt['bert_model'])
        self.bertGcnModel.classifier.load_state_dict(ckpt['classifier'])

        self.optimizer = torch.optim.AdamW([
            dict(params=self.bertGcnModel.bert_model.parameters(), lr=self.bert_lr),
            dict(params=self.bertGcnModel.classifier.parameters(), lr=self.bert_lr),
            dict(params=self.bertGcnModel.gcn.convs[0].parameters(), lr=self.gcn_lr, weight_decay = 5e-4),
            dict(params=self.bertGcnModel.gcn.convs[1].parameters(), lr=self.gcn_lr, weight_decay = 0) ###arrumar o weight decay
        ])

        self.warmup_size = 0.1

        self.datasetBank = deep_data(typeSplit=None, 
                                     max_length = self.max_char_length, 
                                     tokenizer = self.bertGcnModel.tokenizer, 
                                     hier_max_seg = self.hier_max_seg,
                                     hier_max_seg_length= self.hier_max_seg_length,
                                     dataname = self.dataname,
                                     problem_type = self.problemType,
                                     flag_hierarchical = self.flag_hierarchical,
                                     flag_bertgcn = True)

        self.dataloaderBank = DataLoader(dataset = self.datasetBank,
                                         batch_size = 1024,
                                         shuffle= False,
                                         drop_last = False)

        self.dataset.docmask = self.dataset.train_mask + self.dataset.test_mask + self.dataset.val_mask
        self.append_ids_to_graph()

        #sampler = ImbalancedSampler(self.dataset, input_nodes=self.dataset.train_mask)

        self.trainLoader = NeighborLoader(self.dataset, input_nodes=self.dataset.train_mask,
                                          num_neighbors = self.neigh_param, shuffle=True, 
                                          batch_size = self.batchSize
                                          #,sampler = sampler
                                          #directed = False
                                          #is_sorted = True
                                          #num_workers = 8, persistent_workers = True
                                          )

        self.subgraphLoader = NeighborLoader(copy.copy(self.dataset), input_nodes=None,
                                             num_neighbors = [-1], shuffle=False,
                                             batch_size = self.batchSize
                                             #directed = False
                                             #is_sorted = True
                                             #num_workers= 4, persistent_workers = True
                                             )

        self.total_steps = len(self.trainLoader) * self.epochs

        def lr_lambda(current_step, num_warmup_steps=int(self.warmup_size * self.total_steps) ,num_training_steps= self.total_steps):

            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda,last_epoch=-1)

        del self.subgraphLoader.data.x, self.subgraphLoader.data.y
        gc.collect()
        self.subgraphLoader.data.num_nodes = self.dataset.num_nodes
        self.subgraphLoader.data.n_id = torch.arange(self.dataset.num_nodes)


    def append_ids_to_graph(self):

        inputList = []
        attentionList = []

        for batch in self.dataloaderBank:

            inputList.append(batch['input_ids'].numpy())
            attentionList.append(batch['attention_mask'].numpy())

        inputList = np.concatenate(inputList)
        attentionList = np.concatenate(attentionList)
        batchIndexT = np.expand_dims(np.arange(0,inputList.shape[0]),axis=1)
        
        colAtList = attentionList.shape[1]
        colInList = inputList.shape[1]

        totalArr = np.concatenate((batchIndexT,inputList,attentionList),axis=1)

        tmpIdx = []
        tmpInsList = []
        tmpAtList = []

        for n in self.dataset.indexMask:
            if n == 999999:
                tmpIdx.append(np.array([[999999]]))
                tmpInsList.append(np.array([[0]*colInList]))
                tmpAtList.append(np.array([[0]*colAtList]))
            else:
                match = ((self.dataset.indexMask == n).nonzero(as_tuple=True)[0]).item()
                tmpIdx.append(np.array([[match]]))
                tmpInsList.append(np.expand_dims(totalArr[n][1:colAtList+1],axis=0))
                tmpAtList.append(np.expand_dims(totalArr[n][colAtList+1:colInList+colAtList+1],axis=0))

        tmpInsList = np.concatenate(tmpInsList)
        tmpAtList = np.concatenate(tmpAtList)

        self.dataset['input_ids'] = torch.tensor(tmpInsList,dtype=torch.int32)
        self.dataset['attention_mask'] = torch.tensor(tmpAtList,dtype=torch.int32)
            
        return None

class Bert_GCN(torch.nn.Module):
    def __init__(self, model_name, nb_class, hiddenChannels, device, m):

        super(Bert_GCN, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = torch.nn.Linear(self.feat_dim, nb_class)
        self.gcn = Text_GCN(
            in_channels=self.feat_dim,      
            hidden_channels=hiddenChannels,
            out_channels=nb_class,
            device = device
        )

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = torch.nn.Softmax(dim=1)(cls_logit)
        gcn_logit = self.gcn(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
        gcn_pred = torch.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        pred = torch.log(pred)
        return pred