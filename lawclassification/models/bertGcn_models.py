#https://github.com/ZeroRin/BertGCN/blob/main/model/models.py

import torch
from transformers import AutoModelForSequenceClassification
from models.gcn_models import Text_GCN
from utils.definitions import ROOT_DIR
import json
from torch_geometric.loader import NeighborLoader
from utils.helper_funs import set_learning_rates, set_new_learning_rates
import pickle
import os
import random
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

        self.dataname = experiment['dataname']
        self.model_name = experiment['model_name']
        self.model_path = os.path.join(ROOT_DIR,'lawclassification','models','external',self.model_name)
        self.epochs = int(experiment['epochs'])
        self.hiddenChannels = int(experiment['hidden_channels'])
        self.problemType = experiment['problem_type']
        self.batchSizeGcn = int(experiment['batchsize_gcn'])
        self.batchSize = int(experiment['batchsize_gcn'])
        self.bert_lr = experiment['bert_lr']
        self.gcn_lr = experiment['gcn_lr']
        self.flag_hierarchical = experiment['hierarchical']
        self.max_char_length = experiment['max_char_length']
        self.neigh_param = eval(str(experiment['neigh_paramater']))
        self.hier_max_seg = int(experiment['hier_max_seg'])
        self.hier_max_seg_length = int(experiment['hier_max_seg_length'])
        self.decay_lr =  experiment['decay_lr']
        self.weight_decay =  experiment['weight_decay']
        self.qtyFracLayers =  int(experiment['qty_layer_unfreeze'])

        if self.flag_hierarchical:
            #self.finetunepath = os.path.join(ROOT_DIR,'lawclassification','models','internal','hier',self.dataname,self.model_name)
            self.finetunepath = os.path.join(ROOT_DIR,'lawclassification','models','internal','hier',self.dataname, self.model_name,'hier.pickle')
            del self.dataset.token_w_hier_id
            del self.dataset.token_w_hier_att
            del self.dataset.token_w_hier_tid
            del self.dataset.token_s_hier_tid
            gc.collect()
            self.dataset.token_w_hier_id = self.dataset.token_s_hier_id
            self.dataset.token_w_hier_att = self.dataset.token_s_hier_att
            #self.dataset.token_w_hier_tid = self.dataset.token_s_hier_tid
        else:
            self.finetunepath = os.path.join(ROOT_DIR,'lawclassification','models','internal','normal',self.dataname,self.model_name)
            del self.dataset.token_s_hier_id
            del self.dataset.token_s_hier_att
            del self.dataset.token_s_hier_tid
            del self.dataset.token_w_hier_tid
            gc.collect()

        with open(os.path.join(os.path.join(ROOT_DIR,'data',self.dataname,'interm','id2label.json'))) as f:
            self.id2label =  json.load(f)
            f.close()

        with open(os.path.join(os.path.join(ROOT_DIR,'data',self.dataname,'interm','label2id.json'))) as f:
            self.label2id = json.load(f)
            f.close()

        self.bertGcnModel = Bert_GCN(self.id2label, self.label2id , 
                                     self.problemType, self.dataset.num_classes, 
                                     self.hiddenChannels, self.device, self.finetunepath, self.flag_hierarchical , m=0.7)

        self.optimizer = torch.optim.AdamW(set_new_learning_rates(model = self.bertGcnModel,
                                                                  base_lr = self.bert_lr,
                                                                  decay_lr = self.decay_lr,
                                                                  weight_decay_bert = self.weight_decay,
                                                                  qtyLayers = self.qtyFracLayers,
                                                                  embeddings = False,
                                                                  gcn_lr = self.gcn_lr))

        self.warmup_size = 0.1

        self.dataset.docmask = self.dataset.train_mask + self.dataset.test_mask + self.dataset.val_mask

        self.trainLoader = NeighborLoader(self.dataset, input_nodes=self.dataset.train_mask,
                                          num_neighbors = self.neigh_param, shuffle=True, 
                                          is_sorted=True,
                                          batch_size = self.batchSizeGcn
                                          )
        #precisa usar a mask?::
        #self.trainLoader.data.num_nodes = self.dataset.num_nodes
        self.trainLoader.data.n_id = torch.arange(self.dataset.num_nodes)

        self.testLoader = NeighborLoader(self.dataset, input_nodes = self.dataset.test_mask,
                                          num_neighbors = self.neigh_param, shuffle = False, 
                                          is_sorted=True,
                                          batch_size = self.batchSizeGcn
                                          )

        #precisa usar a mask?::
        #self.testLoader.data.num_nodes = self.dataset.num_nodes
        #self.testLoader.data.n_id = torch.arange(self.dataset.num_nodes)

        self.valLoader = NeighborLoader(self.dataset, input_nodes = self.dataset.val_mask,
                                        num_neighbors = self.neigh_param, shuffle = False, 
                                        is_sorted=True,
                                        batch_size = self.batchSizeGcn
                                        )

        #precisa usar a mask?::
        #self.valLoader.data.num_nodes = self.dataset.num_nodes
        #self.valLoader.data.n_id = torch.arange(self.dataset.num_nodes)

        #self.updateDataLoader = NeighborLoader(self.dataset, input_nodes = self.dataset.docmask,
        #                                       num_neighbors = [-1], shuffle = False, 
        #                                       is_sorted=True,
        #                                       batch_size = 32
        #                                      )


        self.updateDataLoader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.dataset.token_w_hier_id[self.dataset.docmask], 
                                           self.dataset.token_w_hier_att[self.dataset.docmask]),
                                           batch_size=32)

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

class Bert_GCN(torch.nn.Module):
    def __init__(self, id2label, label2id, problemType, nb_class, hiddenChannels, device, finetunepath, flag_hierarchical, m):

        super(Bert_GCN, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.problemType = problemType

        if flag_hierarchical:
            f = open(finetunepath,'rb')
            self.bert_model = pickle.load(f)
            f.close()
            self.num_features = list(self.bert_model.modules())[-8].out_features
        else:
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(finetunepath,                                                    
                                                                                local_files_only = True, 
                                                                                #output_scores = True,
                                                                                problem_type = problemType,
                                                                                num_labels = nb_class,
                                                                                id2label = id2label,
                                                                                label2id = label2id,
                                                                                output_attentions = False,
                                                                                output_hidden_states = True,
                                                                                ################
                                                                                torch_dtype = torch.float32, trust_remote_code=True)

            self.num_features = list(self.bert_model.modules())[-4].out_features

        self.device = device
        self.gcn = Text_GCN(
            in_channels=self.num_features,      
            hidden_channels=hiddenChannels,
            out_channels=nb_class,
            device = device
        )

    def forward(self, graphFeats,graphEdgeIndex, graphEdgeWieght, graphInputIds, graphAttMask, graphTokenType, batch_size):
    #def forward(self, dataset, n_id, graphEdgeIndex, graphEdgeWieght, graphInputIds, graphAttMask, batch_size, subgraphLoader):
        if self.training:
            cls_feats = self.bert_model.bert(input_ids=graphInputIds, attention_mask=graphAttMask, token_type_ids=graphTokenType).pooler_output
            #cls_logit = self.bert_model.classifier(self.bert_model.dropout(cls_feats))
            cls_logit = self.bert_model.classifier(cls_feats)
            graphFeats[:batch_size] = cls_feats
        else:
            cls_feats = graphFeats[:batch_size]
            #cls_logit = self.bert_model.classifier(self.bert_model.dropout(cls_feats))
            cls_logit = self.bert_model.classifier(cls_feats)

        gcn_logit = self.gcn(graphFeats, graphEdgeIndex, graphEdgeWieght)[:batch_size]

        if self.problemType == 'single_label_classification':
            pred = (torch.nn.Softmax(dim=1)(gcn_logit)+1e-10) * self.m + torch.nn.Softmax(dim=1)(cls_logit) * (1 - self.m)
            pred = torch.log(pred)
        else:
            pred = (torch.nn.Sigmoid()(gcn_logit)) * self.m + torch.nn.Sigmoid()(cls_logit) * (1 - self.m)
            
        return pred, cls_feats