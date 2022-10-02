#https://github.com/ZeroRin/BertGCN/blob/main/model/models.py

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from models.gcn_models import Text_GCN
from utils.definitions import ROOT_DIR
import json
from torch_geometric.loader import NeighborLoader, ImbalancedSampler
from utils.helper_funs import set_learning_rates
import pickle
import os
import copy
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

        f = open(os.path.join(ROOT_DIR,'data',experiment['dataname'],'pygraph_bertGcn.pickle'),'rb')
        #self.dataset = pickle.load(f).to(self.device)
        self.dataset = pickle.load(f)
        f.close()
        #not hier:
        del self.dataset.token_s_hier_id
        del self.dataset.token_s_hier_att
        del self.dataset.token_s_hier_tid
        gc.collect()

        self.dataname = experiment['dataname']
        self.model_name = experiment['model_name']
        self.model_path = os.path.join(ROOT_DIR,'lawclassification','models','external',self.model_name)
        self.epochs = int(experiment['epochs'])
        self.hiddenChannels = int(experiment['hidden_channels'])
        self.problemType = experiment['problem_type']
        self.batchSize = int(experiment['batchsize'])
        self.batchSizeGcn = int(experiment['batchsize_gcn'])
        self.bert_lr = experiment['bert_lr']
        self.gcn_lr = experiment['gcn_lr']
        self.flag_hierarchical = experiment['hierarchical']
        self.max_char_length = experiment['max_char_length']
        self.neigh_param = eval(str(experiment['neigh_paramater']))
        self.hier_max_seg = int(experiment['hier_max_seg'])
        self.hier_max_seg_length = int(experiment['hier_max_seg_length'])
        self.decay_lr =  experiment['decay_lr']
        self.weight_decay =  experiment['weight_decay']
        self.qtyFracLayers =  experiment['qty_layer_unfreeze']
        self.finetunepath = os.path.join(ROOT_DIR,'lawclassification','models','internal',self.dataname,self.model_name)

        with open(os.path.join(os.path.join(ROOT_DIR,'data',self.dataname,'interm','id2label.json'))) as f:
            self.id2label =  json.load(f)
            f.close()

        with open(os.path.join(os.path.join(ROOT_DIR,'data',self.dataname,'interm','label2id.json'))) as f:
            self.label2id = json.load(f)
            f.close()

        self.bertGcnModel = Bert_GCN(self.model_path, self.id2label, self.label2id , self.problemType, self.dataset.num_classes, self.hiddenChannels, self.device, self.finetunepath, m=0.7 )

        #ckpt = torch.load(os.path.join(ROOT_DIR,'data',self.dataname,'interm','tuned_deep_model.pth'), map_location=self.device)
        #self.bertGcnModel.bert_model.load_state_dict(ckpt['bert_model'])
        #self.bertGcnModel.classifier.load_state_dict(ckpt['classifier'])
        #del ckpt
        #gc.collect()

        self.optimizer = torch.optim.AdamW(set_learning_rates(self.bert_lr,
                                                              self.decay_lr,
                                                              self.bertGcnModel,
                                                              self.weight_decay,
                                                              self.qtyFracLayers,
                                                              self.gcn_lr)) 

        self.warmup_size = 0.1

        self.dataset.docmask = self.dataset.train_mask + self.dataset.test_mask + self.dataset.val_mask

        self.trainLoader = NeighborLoader(self.dataset, input_nodes=self.dataset.train_mask,
                                          num_neighbors = self.neigh_param, shuffle=True, 
                                          is_sorted=True,
                                          batch_size = self.batchSizeGcn
                                          )
        #precisa usar a mask?::
        self.trainLoader.data.num_nodes = self.dataset.num_nodes
        self.trainLoader.data.n_id = torch.arange(self.dataset.num_nodes)

        self.testLoader = NeighborLoader(self.dataset, input_nodes = self.dataset.test_mask,
                                          num_neighbors = self.neigh_param, shuffle = False, 
                                          is_sorted=True,
                                          batch_size = self.batchSizeGcn
                                          )

        #precisa usar a mask?::
        self.testLoader.data.num_nodes = self.dataset.num_nodes
        self.testLoader.data.n_id = torch.arange(self.dataset.num_nodes)

        self.valLoader = NeighborLoader(self.dataset, input_nodes = self.dataset.val_mask,
                                          num_neighbors = self.neigh_param, shuffle = False, 
                                          is_sorted=True,
                                          batch_size = self.batchSizeGcn
                                          )

        #precisa usar a mask?::
        self.valLoader.data.num_nodes = self.dataset.num_nodes
        self.valLoader.data.n_id = torch.arange(self.dataset.num_nodes)

        self.total_steps = self.epochs

        self.scheduler1 = torch.optim.lr_scheduler.LinearLR(self.optimizer, 
                                                            start_factor=0.1, 
                                                            end_factor=1, 
                                                            total_iters=int(self.warmup_size * self.total_steps))

        self.scheduler2 = torch.optim.lr_scheduler.LinearLR(self.optimizer, 
                                                            start_factor=1.0, 
                                                            end_factor=0.1,
                                                            total_iters=self.epochs-(int(self.warmup_size * self.total_steps))
                                                            )

        self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, 
                                                               schedulers = [self.scheduler1,self.scheduler2],
                                                               milestones= [int(self.warmup_size * self.total_steps)]
                                                               )

        self.bertGcnModel.to(self.device)
        #self.dataset.to(self.device)

        #como tirar isso aqui?
        #usar outro loader: 
        self.updateDataLoader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.dataset.token_w_hier_id[self.dataset.docmask], 
                                           self.dataset.token_w_hier_att[self.dataset.docmask],
                                           self.dataset.token_w_hier_tid[self.dataset.docmask]),
                                           batch_size=32)

class Bert_GCN(torch.nn.Module):
    def __init__(self, model_name, id2label, label2id, problemType, nb_class, hiddenChannels, device, finetunepath, m):

        super(Bert_GCN, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.problemType = problemType
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(finetunepath,                                                    
                                                                             local_files_only = True, 
                                                                             #output_scores = True,
                                                                             problem_type = problemType,
                                                                             num_labels = nb_class,
                                                                             id2label = id2label,
                                                                             label2id = label2id,
                                                                             output_attentions = False, #?
                                                                             output_hidden_states = False, #?
                                                                             ################
                                                                             torch_dtype = torch.float32)

        self.device = device
        self.gcn = Text_GCN(
            in_channels=self.feat_dim,      
            hidden_channels=hiddenChannels,
            out_channels=nb_class,
            device = device
        )

    def forward(self, graphFeats, n_id,  graphEdgeIndex, graphEdgeWieght, graphInputIds, graphAttMask, batch_size, subgraphLoader):
    #def forward(self, dataset, n_id, graphEdgeIndex, graphEdgeWieght, graphInputIds, graphAttMask, batch_size, subgraphLoader):
        input_ids, attention_mask = graphInputIds, graphAttMask
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            graphFeats[:batch_size] = cls_feats
        else:
            cls_feats = graphFeats[:batch_size]

        gcn_logit = self.gcn(graphFeats, graphEdgeIndex, graphEdgeWieght)[:batch_size]
        cls_logit = self.classifier(cls_feats)
        #checar se é sigmoid para multilabel classification
        cls_pred = torch.nn.Softmax(dim=1)(cls_logit)
        gcn_pred = torch.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)

        if self.problemType == 'single_label_classification':
            pred = torch.log(pred)
            
        return pred, cls_feats
    
    #def forward(self, g, idx):
    #    input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
    #    if self.training:
    #        cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
    #        g.ndata['x'][idx] = cls_feats
    #    else:
    #        cls_feats = g.ndata['x'][idx]
    #    cls_logit = self.classifier(cls_feats)
    #    cls_pred = torch.nn.Softmax(dim=1)(cls_logit)
    #    gcn_logit = self.gcn(g.ndata['x'], g, g.edata['edge_weight'])[idx]
    #    gcn_pred = torch.nn.Softmax(dim=1)(gcn_logit)
    #    pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
    #    pred = torch.log(pred)
    #    return pred