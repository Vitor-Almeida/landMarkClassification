import torch
import pandas as pd
import os
import json
import numpy as np
from torch.utils.data import Dataset
from utils.definitions import ROOT_DIR
import spacy
import re

NLP = spacy.load('en_core_web_lg')

class deep_data(Dataset):
    """
    Load raw data in a pytorch dataset class
    All data will follow the (label,text) format
    """
    def __init__(self, typeSplit,tokenizer,hier_max_seg,hier_max_seg_length,
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

        self.dataframe['labels'] = self.dataframe['labels'].apply(lambda row: eval(str(row)))

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

        #single_label:
        text1 = self.text[idx]

        if self.flag_hierarchical:

            #esse aqui é especifico do scotus:
            #text1 = re.split('\n{2,}', text1) #achar um 'sentence' tokenizer padrão e tacar aqui <--- tem q retornar outra lista
            text1 = [str(text) for text in NLP(text1).sents]

            case_encodings = self.tokenizer(text1[:self.hier_max_seg], padding='max_length', max_length=self.hier_max_seg_length, truncation=True)

            inputIds = torch.tensor(case_encodings['input_ids'] + [[0] * self.hier_max_seg_length] * (self.hier_max_seg - len(case_encodings['input_ids'])),dtype=torch.int32)
            attentionMask = torch.tensor(case_encodings['attention_mask'] + [[0] * self.hier_max_seg_length] * (self.hier_max_seg - len(case_encodings['attention_mask'])),dtype=torch.int32)
            tokenTypeIds = torch.tensor(case_encodings['token_type_ids'] + [[0] * self.hier_max_seg_length] * (self.hier_max_seg - len(case_encodings['token_type_ids'])),dtype=torch.int32)

            batch = {'input_ids' : inputIds.squeeze(0),
                     'attention_mask': attentionMask.squeeze(0),
                     'token_type_ids': tokenTypeIds.squeeze(0),
                     'labels': torch.tensor(self.labels[idx])
                    }

        else:

            inputs = self.tokenizer.encode_plus(
                            text1 ,
                            padding='max_length',
                            add_special_tokens=True,
                            return_attention_mask=True,
                            max_length=self.max_length,
                            return_tensors = 'pt',
                            truncation=True,
                        )           

            batch = {'input_ids':inputs["input_ids"].squeeze(0).to(dtype=torch.int32),
                    'attention_mask': inputs["attention_mask"].squeeze(0).to(dtype=torch.int32),
                    #'token_type_ids': inputs["token_type_ids"].squeeze(0),# <--- fix for other models
                    'labels': torch.tensor(self.labels[idx])
                    }

        return batch