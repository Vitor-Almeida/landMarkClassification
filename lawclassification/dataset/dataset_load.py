#from torch import is_tensor,tensor,long
import torch
import pandas as pd
import os
from torch.utils.data import Dataset
from utils.definitions import ROOT_DIR

#pyton -m dataset.landMarkTorchDataset => rodar testes 

class yelpReview(Dataset):
    """
    Load raw data in a pytorch dataset class
    All data will follow the (label,text) format
    """
    def __init__(self, typeSplit,tokenizer,max_length):

        super(yelpReview, self).__init__()
        self.name = 'yelp'
        self.dataframe = pd.read_csv(os.path.join(ROOT_DIR,'data',self.name,'interm',typeSplit,typeSplit+'.csv'))
        self.labels = self.dataframe.iloc[:,0]
        self.text = self.dataframe.iloc[:,1]
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        text1 = self.text[idx]

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
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        return {
            'input_ids': ids.squeeze(0),
            'attention_mask': mask.squeeze(0),
            'token_type_ids': token_type_ids.squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }