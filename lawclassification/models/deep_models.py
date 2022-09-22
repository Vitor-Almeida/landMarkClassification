import random
import numpy as np
from dataset.dataset_load import deep_data
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader
import os
from utils.definitions import ROOT_DIR
from transformers import logging
import re
from utils.helper_funs import EarlyStopping

def set_learning_rates(base_lr,decay_lr,model,weight_decay,qtyFracLayers):

    #https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2022/03/29/discriminative-lr.html

    # horrivel, tentar fazer de uma forma melhor:

    for param in model.parameters():
        param.requires_grad = False #começar com tudo congelado.

    ############ SEMPRE CHECAR AQUI !!! NEM TODOS OS TRANSFORMERS TEM O MESMO PADRAO DE NOME DAS CAMADAS!!! ##########

    allLayers = [name[0] for name in model.named_parameters()]
    allLayersStr=' '.join(allLayers)
    tt_layers = len(np.unique(re.findall(r'layers?\.[0-9]+\.',allLayersStr)))
    tt_layers=0 if tt_layers==1 else tt_layers

    totalLayersToUnfreeze = int(qtyFracLayers*tt_layers)

    layer_names = []
    for idx, (name, param) in enumerate(model.named_parameters()):

        curLayerNum = re.search(r'layers?\.[0-9]+\.',name)

        if curLayerNum != None:
            curLayerNum = int(re.search(r'[0-9]+',curLayerNum.group() ).group() )
            if tt_layers - curLayerNum <= totalLayersToUnfreeze:
                param.requires_grad = True #descongelar
                layer_names.append(name) 
        elif re.search(r'embeddings',name) != None:
            pass
        else:
            param.requires_grad = True #descongelar
            layer_names.append(name) #o pooler fica com um LR alto, ta certo?

    layer_names.reverse()
    parameters = []

    prev_group_name = re.search(r'layers?\.[0-9]+\.',layer_names[0])
    if prev_group_name == None:
        prev_group_name = layer_names[0].split('.')[-1] # a ideia aqui é pegar todos os pares weight/bias e manter igual
        pre_layerNum = None
    else:
        prev_group_name = prev_group_name.group()
        pre_layerNum = int(re.search(r'[0-9]',prev_group_name.group()))

    #fix this:
    if prev_group_name == 'weight':
        check1 = 'weight'
        check2 = 'bias'
    elif prev_group_name == 'bias':
        check1 = 'bias'
        check2 = 'weight'

    # store params & learning rates
    for idx, name in enumerate(layer_names):
        
        # parameter group name
        if re.search(r'layers?\.[0-9]+\.',name) == None:
            cur_group_name = name.split('.')[-1] # a ideia aqui é pegar todos os pares weight/bias e manter igual
            if (prev_group_name == check1 and cur_group_name == check2) or idx==0 or re.search(r'classifier',name) != None:
                base_lr = base_lr
            else:
                base_lr *= decay_lr
                #base_lr = base_lr - 1 check
            prev_group_name = cur_group_name

        else:
            cur_layerNum = int(re.search(r'[0-9]+',(re.search(r'layers?\.[0-9]+\.',name)).group() ).group() )
            if pre_layerNum == None:
                 pre_layerNum = cur_layerNum
                 #base_lr = base_lr - 1 check
                 base_lr *= decay_lr

            if cur_layerNum == pre_layerNum:
                base_lr = base_lr
            else:
                #base_lr = base_lr - 1 check
                base_lr *= decay_lr

            pre_layerNum = cur_layerNum
        
        # display info
        print(f'{idx}: lr = {base_lr:.6f}, {name}')
        
        # append layer parameters
        parameters += [{'params': [p for n, p in model.named_parameters() if n == name and p.requires_grad],
                        'lr':     base_lr,
                        'weight_decay': weight_decay}]

    return parameters

class deep_models():
    def __init__(self, model_name, batchsize, max_char_length, lr, epochs, warmup_size, dropout, dataname, problem_type, weight_decay, decay_lr, qty_layer_unfreeze):
        super(deep_models, self).__init__()

        logging.set_verbosity_error() #remove transformers warnings.

        self.dataname = dataname
        self.model_name = model_name
        self.model_path = os.path.join(ROOT_DIR,'lawclassification','models','external',self.model_name)
        self.dropout = dropout
        self.problem_type = problem_type
        self.batchsize = batchsize
        self.max_char_length = max_char_length
        self.warmup_size = warmup_size
        self.weight_decay = weight_decay
        self.qtyFracLayers = qty_layer_unfreeze
        self.lr = lr
        self.decay_lr = decay_lr
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seed_val = random.randint(0, 1000)
        random.seed(self.seed_val)
        np.random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)
        torch.cuda.manual_seed_all(self.seed_val)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, do_lower_case=True)
        
        self.dataset_val = deep_data(typeSplit='val', 
                                     max_length = self.max_char_length, 
                                     tokenizer = self.tokenizer, 
                                     dataname = self.dataname,
                                     problem_type = self.problem_type)

        self.dataset_train = deep_data(typeSplit='train', 
                                       max_length = self.max_char_length, 
                                       tokenizer = self.tokenizer, 
                                       dataname = self.dataname,
                                       problem_type = self.problem_type)

        self.dataset_test = deep_data(typeSplit='test', 
                                      max_length = self.max_char_length, 
                                      tokenizer = self.tokenizer, 
                                      dataname = self.dataname,
                                      problem_type = self.problem_type)

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
                                                                        output_attentions = False,
                                                                        output_hidden_states = False,
                                                                        ################
                                                                        #torch_dtype = torch.float16, #fica tudo 16 bytes, o que nao é bom?
                                                                        #classifier_dropout = self.dropout, ###?
                                                                        #hidden_dropout_prob = self.dropout, ###?
                                                                        #attention_probs_dropout_prob = self.dropout
                                                                        )

        self.model = self.model.cuda() if torch.cuda.is_available() else self.model.cpu()

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
                                                              self.qtyFracLayers)) 
                
        #Slanted Triangular Learning Rates
        def lr_lambda(current_step, 
                      num_warmup_steps=int(self.warmup_size * self.total_steps),
                      num_training_steps = self.total_steps):

            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda,last_epoch=-1)