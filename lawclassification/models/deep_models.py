import random
import numpy as np
from dataset.dataset_load import deep_data
from models.hier_models import HierarchicalBert
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader
import os
from utils.definitions import ROOT_DIR
from transformers import logging
from utils.helper_funs import EarlyStopping, set_learning_rates

class deep_models():
    def __init__(self, model_name, batchsize, max_char_length, lr, epochs, warmup_size, dropout, dataname,
                 problem_type, weight_decay, decay_lr, qty_layer_unfreeze, hierarchical, hier_max_seg, hier_max_seg_length):
        super(deep_models, self).__init__()

        logging.set_verbosity_error() #remove transformers warnings.

        self.dataname = dataname
        self.model_name = model_name
        self.model_path = os.path.join(ROOT_DIR,'lawclassification','models','external',self.model_name)
        self.dropout = dropout
        self.problem_type = problem_type
        self.batchsize = batchsize
        self.max_char_length = max_char_length
        self.flag_hierarchical = hierarchical
        self.warmup_size = warmup_size
        self.weight_decay = weight_decay
        self.qtyFracLayers = qty_layer_unfreeze
        self.lr = lr
        self.decay_lr = decay_lr
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hier_max_seg = hier_max_seg
        self.hier_max_seg_length = hier_max_seg_length

        self.seed_val = random.randint(0, 1000)
        random.seed(self.seed_val)
        np.random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)
        torch.cuda.manual_seed_all(self.seed_val)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, do_lower_case=True)
        
        self.dataset_val = deep_data(typeSplit='val', 
                                     max_length = self.max_char_length, 
                                     tokenizer = self.tokenizer, 
                                     hier_max_seg = self.hier_max_seg,
                                     hier_max_seg_length= self.hier_max_seg_length,
                                     dataname = self.dataname,
                                     problem_type = self.problem_type,
                                     flag_hierarchical = self.flag_hierarchical,
                                     flag_bertgcn = False)

        self.dataset_train = deep_data(typeSplit='train', 
                                       max_length = self.max_char_length, 
                                       tokenizer = self.tokenizer, 
                                       hier_max_seg = self.hier_max_seg,
                                       hier_max_seg_length= self.hier_max_seg_length,
                                       dataname = self.dataname,
                                       problem_type = self.problem_type,
                                       flag_hierarchical = self.flag_hierarchical,
                                       flag_bertgcn = False)

        self.dataset_test = deep_data(typeSplit='test', 
                                      max_length = self.max_char_length, 
                                      tokenizer = self.tokenizer, 
                                      hier_max_seg = self.hier_max_seg,
                                      hier_max_seg_length= self.hier_max_seg_length,
                                      dataname = self.dataname,
                                      problem_type = self.problem_type,
                                      flag_hierarchical = self.flag_hierarchical,
                                      flag_bertgcn = False)

        self.val_dataloader = DataLoader(dataset = self.dataset_val,
                                         batch_size = self.batchsize,
                                         drop_last = True)

        self.train_dataloader = DataLoader(dataset = self.dataset_train,
                                           batch_size = self.batchsize,
                                           shuffle = True,
                                           drop_last = True)

        self.test_dataloader = DataLoader(dataset = self.dataset_test,
                                          batch_size = self.batchsize,
                                          drop_last = True)

        self.earlyStopper = EarlyStopping(patience=3, min_delta=0)                                  
        
        self.num_labels_train = self.dataset_train.num_labels
        self.num_labels_test = self.dataset_test.num_labels
        self.num_labels_val = self.dataset_val.num_labels

        self.num_labels = self.num_labels_train

        self.total_steps = len(self.train_dataloader) * epochs

        #da pra colocar tipo um config.json aqui que da pra mudar as parada de dropout, requires grad:
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path, 
                                                                        local_files_only = True, 
                                                                        output_scores = True,
                                                                        problem_type = self.problem_type,
                                                                        num_labels = self.num_labels_train,
                                                                        id2label = self.dataset_train.id2label,
                                                                        label2id = self.dataset_train.label2id,
                                                                        output_attentions = False, #?
                                                                        output_hidden_states = False, #?
                                                                        ################
                                                                        torch_dtype = torch.float32 #fica tudo 16 bytes, o que nao é bom?
                                                                        #classifier_dropout = self.dropout, ###?
                                                                        #hidden_dropout_prob = self.dropout, ###?
                                                                        #attention_probs_dropout_prob = self.dropout
                                                                        )

        #check hierarchical
        if self.flag_hierarchical:

            self.model.bert = HierarchicalBert(encoder=self.model.bert,
                                               max_segments=self.hier_max_seg,
                                               max_segment_length=self.hier_max_seg_length,
                                               device = self.device)

        self.model.to(self.device)

        ##ARRUMAR :
        ########## AQUI TEM QUE CONGELAR DEPENDENDO DO INPUT #############

        # for _,param in enumerate(list(self.model.named_parameters())):

        #     #separar 1 pra cada, nao tem muito oq fazer, talvez aqui seja o lugar onde vai setar os learning rate tmb talvez.

        #     if self.model.base_model_prefix == 'bert' or self.model.base_model_prefix == 'roberta' or self.model.base_model_prefix == 'longformer':

        #         if param[0].find(f'{self.model.base_model_prefix}.encoder.layer.11.') != -1 or param[0].find('classifier') != -1:
        #             param[1].requires_grad = True #GRADIENT SERÁ CALCULADO
        #         else:
        #             param[1].requires_grad = False #NAO SERÁ CALCULADO

        #     elif self.model.base_model_prefix == 'albert-base-v2':

        #         if param[0].find(f'albert_layer_groups.0.albert_layers.0') != -1 or param[0].find('classifier') != -1:
        #             param[1].requires_grad = True #GRADIENT SERÁ CALCULADO
        #         else:
        #             param[1].requires_grad = False #NAO SERÁ CALCULADO

        #     elif self.model.base_model_prefix == 'distilbert':

        #         if param[0].find('distilbert.transformer.layer.5') != -1 or param[0].find('classifier') != -1: #tirar a camada pre_classifier ??
        #             param[1].requires_grad = True #GRADIENT SERÁ CALCULADO
        #         else:
        #             param[1].requires_grad = False #NAO SERÁ CALCULADO

        #self.model.dropout.p = self.dropout #aonde fica os dropout?

        ##################################################################

        self.optimizer = torch.optim.AdamW(set_learning_rates(self.lr,
                                                              self.decay_lr,
                                                              self.model,
                                                              self.weight_decay,
                                                              self.qtyFracLayers,
                                                              None)) 
                
        #Slanted Triangular Learning Rates
        def lr_lambda(current_step, 
                      num_warmup_steps=int(self.warmup_size * self.total_steps),
                      num_training_steps = self.total_steps):
        
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda,last_epoch=-1)