import torch
import pandas as pd
import os
import json
import numpy as np
from torch.utils.data import Dataset
from utils.definitions import ROOT_DIR
#import spacy
#import re

def evallist(row):

    #slow:
    ########### TESTING: (64x128 OOM) #############:
    cols = ['labels','token_s_hier_id','token_s_hier_att','token_s_hier_tid','token_w_hier_id','token_w_hier_att','token_w_hier_tid']
    #cols = ['labels','token_s_hier_id','token_s_hier_att','token_s_hier_tid']
    ############# TESTING: (64x128 OOM) ####################

    for col in cols:
        row[col] = eval(str(row[col]))

    return row

#NLP = spacy.load('en_core_web_lg')

class deep_data(Dataset):
    """
    Load raw data in a pytorch dataset class
    All data will follow the (label,text) format
    """
    def __init__(self, typeSplit,hier_max_seg,hier_max_seg_length,
                 max_length,dataname,problem_type,flag_hierarchical,flag_bertgcn):
        super(deep_data, self).__init__()

        self.name = dataname
        self.problem_type = problem_type
        self.flag_hierarchical = flag_hierarchical
        self.hier_max_seg = hier_max_seg
        self.hier_max_seg_length = hier_max_seg_length

        if flag_bertgcn:
            dataframeTrain = pd.read_csv(os.path.join(ROOT_DIR,'data',self.name,'interm','train','train.csv'))
            dataframeTest = pd.read_csv(os.path.join(ROOT_DIR,'data',self.name,'interm','test','test.csv'))
            dataframeVal = pd.read_csv(os.path.join(ROOT_DIR,'data',self.name,'interm','val','val.csv'))
            self.dataframe = pd.concat([dataframeTrain,dataframeTest,dataframeVal],ignore_index=True)
        else:
            self.dataframe = pd.read_csv(os.path.join(ROOT_DIR,'data',self.name,'interm',typeSplit,typeSplit+'.csv'))
            ############# TESTING: (64x128 OOM) ####################
            #self.dataframe.drop(columns=['token_w_hier_id','token_w_hier_att','token_w_hier_tid'],inplace=True)
            ############# TESTING: (64x128 OOM) ####################

        #self.dataframe = self.dataframe.apply(lambda row: evallist(row),axis=1)
        if self.problem_type == 'single_label_classification':
            pass
        else:
            self.dataframe['labels'] = self.dataframe['labels'].apply(lambda row: eval(row))
        
        #self.dataframe['token_s_hier_id'] = pd.eval(self.dataframe['token_s_hier_id'])

        self.dataframe['token_s_hier_id'] = self.dataframe['token_s_hier_id'].apply(lambda row: eval(row)) #.apply(eval)
        self.dataframe['token_s_hier_att'] = self.dataframe['token_s_hier_att'].apply(lambda row: eval(row))
        self.dataframe['token_s_hier_tid'] = self.dataframe['token_s_hier_tid'].apply(lambda row: eval(row))
        self.dataframe['token_w_hier_id'] = self.dataframe['token_w_hier_id'].apply(lambda row: eval(row))
        self.dataframe['token_w_hier_att'] = self.dataframe['token_w_hier_att'].apply(lambda row: eval(row))
        self.dataframe['token_w_hier_tid'] = self.dataframe['token_w_hier_tid'].apply(lambda row: eval(row))

        with open(os.path.join(os.path.join(ROOT_DIR,'data',self.name,'interm','id2label.json'))) as f:
            self.id2label =  json.load(f)
            f.close()

        with open(os.path.join(os.path.join(ROOT_DIR,'data',self.name,'interm','label2id.json'))) as f:
            self.label2id = json.load(f)
            f.close()

        self.labels = self.dataframe.iloc[:,0]
        self.text = self.dataframe.iloc[:,1]

        self.token_s_hier_id = self.dataframe.iloc[:,3]
        self.token_s_hier_att = self.dataframe.iloc[:,4]
        self.token_s_hier_tid = self.dataframe.iloc[:,5]
        ############# TESTING: (64x128 OOM) ####################
        self.token_w_id = self.dataframe.iloc[:,6] #<-------- testar tirar isso aqui
        self.token_w_att = self.dataframe.iloc[:,7] #<-------- testar tirar isso aqui
        self.token_w_tid = self.dataframe.iloc[:,8] #<-------- testar tirar isso aqui
        ############# TESTING: (64x128 OOM) ####################

        self.max_length = max_length

        self.count = 0

        if self.problem_type == 'single_label_classification':
            self.num_labels = len(np.unique(self.labels))
        else:
            self.num_labels = len(self.labels[0])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.problem_type == 'single_label_classification':
            labels1 = torch.tensor(self.labels[idx],dtype=torch.int64)
        else:
            labels1 = torch.tensor(self.labels[idx],dtype=torch.float32)

        if self.flag_hierarchical:

            token_s_hier_id1 = torch.tensor(self.token_s_hier_id[idx],dtype=torch.int32).squeeze(0)
            token_s_hier_att1 = torch.tensor(self.token_s_hier_att[idx],dtype=torch.int32).squeeze(0)
            token_s_hier_tid1 = torch.tensor(self.token_s_hier_tid[idx],dtype=torch.int32).squeeze(0)

            batch = {'input_ids' : token_s_hier_id1,
                     'attention_mask': token_s_hier_att1,
                     'token_type_ids': token_s_hier_tid1,
                     'labels': labels1
                    }

        else:

            token_w_id1 = torch.tensor(self.token_w_id[idx],dtype=torch.int32).squeeze(0)
            token_w_att1 = torch.tensor(self.token_w_att[idx],dtype=torch.int32).squeeze(0)
            token_w_tid1 = torch.tensor(self.token_w_tid[idx],dtype=torch.int32).squeeze(0)

            batch = {'input_ids':token_w_id1,
                    'attention_mask': token_w_att1,
                    #'token_type_ids': inputs["token_type_ids"].squeeze(0),# <--- fix for other models
                    'labels': labels1
                    }

        return batch