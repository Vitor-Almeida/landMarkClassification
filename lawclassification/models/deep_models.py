import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader
import os
from utils.definitions import ROOT_DIR
from transformers import logging

class deep_models():
    def __init__(self, model_name, batchsize, max_char_length, lr, epochs, warmup_size, class_fun, dropout):
        super(deep_models, self).__init__()

        logging.set_verbosity_error() #remove annoying transformers warnings.

        self.model_name = model_name
        self.model_path = os.path.join(ROOT_DIR,'lawclassification','models','external',self.model_name)
        self.dropout = dropout

        self.batchsize = batchsize
        self.max_char_length = max_char_length
        self.warmup_size = warmup_size

        self.seed_val = random.randint(0, 1000)
        random.seed(self.seed_val)
        np.random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)
        torch.cuda.manual_seed_all(self.seed_val)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, do_lower_case=True)
        
        self.dataset_val = class_fun(typeSplit='val', max_length = self.max_char_length, tokenizer = self.tokenizer)
        self.dataset_train = class_fun(typeSplit='train', max_length = self.max_char_length, tokenizer = self.tokenizer)
        self.dataset_test = class_fun(typeSplit='test', max_length = self.max_char_length, tokenizer = self.tokenizer)

        self.val_dataloader = DataLoader(dataset=self.dataset_val,batch_size=self.batchsize,drop_last=True)
        self.train_dataloader = DataLoader(dataset=self.dataset_train,batch_size=self.batchsize,shuffle=True,drop_last=True)
        self.test_dataloader = DataLoader(dataset=self.dataset_test,batch_size=self.batchsize,drop_last=True)
        
        self.num_labels = len(np.unique(self.dataset_train.labels))
        self.lr = lr
        self.epochs = epochs
        self.total_steps = len(self.train_dataloader) * epochs

        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, num_labels = self.num_labels
                                                                                    ,output_attentions = False,output_hidden_states = False)

        ########## AQUI TEM QUE CONGELAR DEPENDENDO DO INPUT #############
        #### checar se os learning rate são de fato diferente por camada
        #### checar alguns padroes da softmax da ultima layer de classificacao desse bixo
        for _,param in enumerate(list(self.model.bert.named_parameters())):
            if param[0].find('encoder.layer.11.') != -1:
                param[1].requires_grad = True
            else:
                param[1].requires_grad = False

        self.model.dropout.p = self.dropout

        ##################################################################

        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),lr=self.lr) #faz diferenca isso aqui? colocar o optimizer para tudo?

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, #colocar puro pytorch aqui
                                    num_warmup_steps = int(self.warmup_size * self.total_steps), 
                                    num_training_steps = self.total_steps)

        self.model.to(self.device)        