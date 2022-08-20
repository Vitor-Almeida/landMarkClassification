import random
import numpy as np
from dataset.dataset_load import deep_data
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader
import os
from utils.definitions import ROOT_DIR
from transformers import logging

class deep_models():
    def __init__(self, model_name, batchsize, max_char_length, lr, epochs, warmup_size,  dropout, dataname,problem_type):
        super(deep_models, self).__init__()

        #baixar modelo e deixar offline:
        #git clone https://huggingface.co/bert-base-uncased

        logging.set_verbosity_error() #remove annoying transformers warnings.

        self.dataname = dataname
        self.model_name = model_name
        self.model_path = os.path.join(ROOT_DIR,'lawclassification','models','external',self.model_name)
        self.dropout = dropout
        self.problem_type = problem_type

        self.batchsize = batchsize
        self.max_char_length = max_char_length
        self.warmup_size = warmup_size

        self.seed_val = random.randint(0, 1000)
        random.seed(self.seed_val)
        np.random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)
        torch.cuda.manual_seed_all(self.seed_val)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, do_lower_case=True)
        
        self.dataset_val = deep_data(typeSplit='val', max_length = self.max_char_length, tokenizer = self.tokenizer, dataname = self.dataname,problem_type=self.problem_type)
        self.dataset_train = deep_data(typeSplit='train', max_length = self.max_char_length, tokenizer = self.tokenizer, dataname = self.dataname,problem_type=self.problem_type)
        self.dataset_test = deep_data(typeSplit='test', max_length = self.max_char_length, tokenizer = self.tokenizer, dataname = self.dataname,problem_type=self.problem_type)

        self.val_dataloader = DataLoader(dataset=self.dataset_val,batch_size=self.batchsize,drop_last=True)
        self.train_dataloader = DataLoader(dataset=self.dataset_train,batch_size=self.batchsize,shuffle=True,drop_last=True)
        self.test_dataloader = DataLoader(dataset=self.dataset_test,batch_size=self.batchsize,drop_last=True)
        
        self.num_labels_train = self.dataset_train.num_labels
        self.num_labels_test = self.dataset_test.num_labels
        self.num_labels_val = self.dataset_val.num_labels

        if problem_type == 'single_label_classification':
            self.num_labels = self.num_labels_train
        else:
            self.num_labels = 2
            #self.num_labels = self.num_labels_train

        self.lr = lr
        self.epochs = epochs
        self.total_steps = len(self.train_dataloader) * epochs

        #da pra colocar tipo um config.json aqui que da pra mudar as parada de dropout, requires grad:
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, 
                                                                        local_files_only = True, 
                                                                        #torch_dtype = 
                                                                        output_scores = True,
                                                                        problem_type=self.problem_type,
                                                                        num_labels = self.num_labels_train,
                                                                        id2label = self.dataset_train.id2label,
                                                                        label2id = self.dataset_train.label2id,
                                                                        output_attentions = False,
                                                                        output_hidden_states = False,
                                                                        ################
                                                                        classifier_dropout = self.dropout,
                                                                        hidden_dropout_prob = self.dropout,
                                                                        attention_probs_dropout_prob = self.dropout) #MUDOU AQUI

        ########## AQUI TEM QUE CONGELAR DEPENDENDO DO INPUT #############
        #### checar se os learning rate são de fato diferente por camada
        ####
        #### entender oq é decay, 
        #### entender se o learning rate é diferente para cada um, e se eles mudam diferente.
        for _,param in enumerate(list(self.model.named_parameters())):

            #separar 1 pra cada, nao tem muito oq fazer, talvez aqui seja o lugar onde vai setar os learning rate tmb talvez.

            if self.model.base_model_prefix == 'bert' or self.model.base_model_prefix == 'roberta' or self.model.base_model_prefix == 'longformer':

                if param[0].find(f'{self.model.base_model_prefix}.encoder.layer.11.') != -1 or param[0].find('classifier') != -1:
                    param[1].requires_grad = True #GRADIENT SERÁ CALCULADO
                else:
                    param[1].requires_grad = False #NAO SERÁ CALCULADO

            elif self.model.base_model_prefix == 'albert-base-v2':

                if param[0].find(f'albert_layer_groups.0.albert_layers.0') != -1 or param[0].find('classifier') != -1:
                    param[1].requires_grad = True #GRADIENT SERÁ CALCULADO
                else:
                    param[1].requires_grad = False #NAO SERÁ CALCULADO

            elif self.model.base_model_prefix == 'distilbert':

                if param[0].find('distilbert.transformer.layer.5') != -1 or param[0].find('classifier') != -1: #tirar a camada pre_classifier ??
                    param[1].requires_grad = True #GRADIENT SERÁ CALCULADO
                else:
                    param[1].requires_grad = False #NAO SERÁ CALCULADO

        #self.model.dropout.p = self.dropout #aonde fica os dropout?

        ##################################################################

        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),lr=self.lr) #faz diferenca isso aqui? colocar o optimizer para tudo?
        #setar um lr diferente para cada layer?

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, #colocar puro pytorch aqui
                                    num_warmup_steps = int(self.warmup_size * self.total_steps), 
                                    num_training_steps = self.total_steps)

        self.model.to(self.device)        