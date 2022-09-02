#from torch import is_tensor,tensor,long
import torch
import pandas as pd
import os
import json
import numpy as np
from torch.utils.data import Dataset
from utils.definitions import ROOT_DIR

#pyton -m dataset.landMarkTorchDataset => rodar testes 

class deep_data(Dataset):
    """
    Load raw data in a pytorch dataset class
    All data will follow the (label,text) format
    """
    def __init__(self, typeSplit,tokenizer,max_length,dataname,problem_type):

        super(deep_data, self).__init__()
        self.name = dataname
        self.problem_type = problem_type

        if self.problem_type == 'single_label_classification':
            self.dataframe = pd.read_csv(os.path.join(ROOT_DIR,'data',self.name,'interm',typeSplit,typeSplit+'.csv'))
        else:
            self.dataframe = pd.read_csv(os.path.join(ROOT_DIR,'data',self.name,'interm',typeSplit,typeSplit+'.csv'))
            self.dataframe['labels'] = self.dataframe['labels'].apply(lambda row: json.loads(row)) #pra transformar '[A,B,C]' str em list list

        with open(os.path.join(os.path.join(ROOT_DIR,'data',self.name,'interm','id2label.json'))) as f:
            self.id2label =  json.load(f)
            f.close()

        with open(os.path.join(os.path.join(ROOT_DIR,'data',self.name,'interm','label2id.json'))) as f:
            self.label2id = json.load(f)
            f.close()

        self.labels = self.dataframe.iloc[:,0]
        self.text = self.dataframe.iloc[:,1]
        self.max_length = max_length
        self.tokenizer = tokenizer

        if self.problem_type == 'single_label_classification':
            self.num_labels = len(np.unique(self.labels))
        else:
            self.num_labels = len(self.labels[0])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        #single_label:
        text1 = self.text[idx]
        #multi_label:

        # tem que ver se esse encode_plus eh para todos os modelos
        inputs = self.tokenizer.encode_plus(
            text1 ,
            padding='max_length',
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
            return_tensors = 'pt',
            truncation=True,
        )

        ids = inputs["input_ids"]
        #token_type_ids = inputs["token_type_ids"] => arrumar para o roberta faz diferenca esse buxo aqui?
        mask = inputs["attention_mask"]

        return {
            'input_ids': ids.squeeze(0),
            'attention_mask': mask.squeeze(0),
            #'token_type_ids': token_type_ids.squeeze(0), #faz diferenca esse buxo aqui? # tem que entender a parada de dos tokens especiais, separação de senteças
            'labels': torch.tensor(self.labels[idx])
            }


class gcn_data(Dataset):
    """
    Load raw data in a pytorch dataset class
    All data will follow the (label,text) format
    """
    def __init__(self, typeSplit,tokenizer,max_length,dataname,problem_type):

        super(deep_data, self).__init__()
        self.name = dataname
        self.problem_type = problem_type