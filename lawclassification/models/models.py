import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader


class models():
    def __init__(self, model_name, batchsize, max_char_length, lr, epochs, warmup_size, class_fun):
        super(models, self).__init__()

        self.model_name = model_name

        self.batchsize = batchsize
        self.max_char_length = max_char_length
        self.warmup_size = warmup_size

        seed_val = random.randint(0, 1000)
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, do_lower_case=True)
        
        self.dataset_val = class_fun(typeSplit='val', max_length = self.max_char_length, tokenizer = self.tokenizer)
        self.dataset_train = class_fun(typeSplit='train', max_length = self.max_char_length, tokenizer = self.tokenizer)
        self.dataset_test = class_fun(typeSplit='test', max_length = self.max_char_length, tokenizer = self.tokenizer)

        self.val_dataloader = DataLoader(dataset=self.dataset_val,batch_size=self.batchsize)
        self.train_dataloader = DataLoader(dataset=self.dataset_train,batch_size=self.batchsize,shuffle=True)
        self.test_dataloader = DataLoader(dataset=self.dataset_test,batch_size=self.batchsize)
        
        self.num_labels = len(np.unique(self.dataset_train.target))
        self.lr = lr
        self.epochs = epochs
        self.total_steps = len(self.train_dataloader) * epochs

        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels = self.num_labels
                                                                                    ,output_attentions = False,output_hidden_states = False)

        ########## AQUI TEM QUE CONGELAR DEPENDENDO DO INPUT #############

        for _,param in enumerate(list(self.model.bert.named_parameters())):
            if param[0].find('encoder.layer.11.') != -1:
                param[1].requires_grad = True
                #print("destravando: ",param[0])
            else:
                param[1].requires_grad = False

        ##################################################################

        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),lr=self.lr) #faz diferenca isso aqui? colocar o optimizer para tudo?

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, #colocar puro pytorch aqui
                                    num_warmup_steps = int(self.warmup_size * self.total_steps), 
                                    num_training_steps = self.total_steps)

        self.model.to(self.device)        