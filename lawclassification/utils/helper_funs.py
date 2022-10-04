
from utils.definitions import ROOT_DIR
import os
import pandas as pd
from tokenizers import Tokenizer
from tokenizers import pre_tokenizers
from tokenizers import normalizers
from tokenizers.models import WordPiece
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer
import numpy as np
import torch
from math import floor, log10
import re

def save_model(model, epoch):

    torch.save(
    {
        'bert_model': model.model.bert.state_dict(),
        'classifier': model.model.classifier.state_dict(),
        'optimizer': model.optimizer.state_dict(),
        'epoch': epoch,
    },
            os.path.join(ROOT_DIR,'data',model.dataname,'interm', 'tuned_deep_model.pth')
    )

    return None


def read_experiments(fileName,type):

    path = os.path.join(ROOT_DIR,'lawclassification',fileName)

    df = pd.read_csv(path)

    df = df[df['type']==type]

    return df.to_dict(orient='records')

def hug_tokenizer(vocab_size:int):

    bertTokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    bertTokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])
    bertTokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Punctuation('removed'),pre_tokenizers.Whitespace()])

    bertTokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )

    trainer = WordPieceTrainer(
        vocab_size = vocab_size, 
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        #special_tokens=[],
        min_frequency = 0, 
        show_progress = True, 
        initial_alphabet  = [],
        #continuing_subword_prefix = '##'
        continuing_subword_prefix = ''
    )

    return bertTokenizer, trainer

class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            print('early stop count =',self.counter)
            if self.counter >= self.patience:
                return True
        return False


def set_new_learning_rates(model, base_lr, decay_lr, weight_decay_bert, qtyLayers, embeddings, gcn_lr):

    #https://towardsdatascience.com/advanced-techniques-for-fine-tuning-transformers-82e4e61e16e#6196

    opt_parameters = []    # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters_core_bert = [(n,p) for n,p in list(model.named_parameters()) if not any(nd in n for nd in ['seg_pos_embeddings','seg_encoder','convs'])]
    named_parameters_hierBert = [(n,p) for n,p in list(model.named_parameters()) if any(nd in n for nd in ['seg_pos_embeddings','seg_encoder'])]
    named_parameters_convs = [(n,p) for n,p in list(model.named_parameters()) if any(nd in n for nd in ['convs'])]

    #### start core bert #################### :

    qtyLayers = -1 if qtyLayers==-1 else 11 - qtyLayers
    
    # According to AAAMLP book by A. Thakur, we generally do not use any decay 
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    init_lr = base_lr
    head_lr = init_lr + 0.1/(10**-floor(log10(init_lr)))
    lr = init_lr

    for layer in named_parameters_core_bert:
        layer[1].requires_grad = False
    
    # === Pooler and regressor ======================================================  
    
    params_0 = [p for n,p in named_parameters_core_bert if ("pooler" in n or "regressor" in n or "classifier" in n) 
                and any(nd in n for nd in no_decay)]
    params_0_names = [n for n,p in named_parameters_core_bert if ("pooler" in n or "regressor" in n or "classifier" in n) 
                and any(nd in n for nd in no_decay)]
    params_1 = [p for n,p in named_parameters_core_bert if ("pooler" in n or "regressor" in n or "classifier" in n)
                and not any(nd in n for nd in no_decay)]
    params_1_names = [n for n,p in named_parameters_core_bert if ("pooler" in n or "regressor" in n or "classifier" in n)
                and not any(nd in n for nd in no_decay)]
    
    for idx,layer in enumerate(params_1):
        layer.requires_grad = True
        print(f'{params_1_names[idx]} lr: {head_lr} weight_decay: {weight_decay_bert} {layer.requires_grad}')
    for idx,layer in enumerate(params_0):
        layer.requires_grad = True
        print(f'{params_0_names[idx]} lr: {head_lr} weight_decay: 0.0 {layer.requires_grad}')

    head_params = {"params": params_0, "lr": head_lr, "weight_decay": 0.0}
    opt_parameters.append(head_params)
        
    head_params = {"params": params_1, "lr": head_lr, "weight_decay": weight_decay_bert}
    opt_parameters.append(head_params)

    # === 12 Hidden layers ==========================================================
    
    for layer in range(11,qtyLayers,-1):        
        params_0 = [p for n,p in named_parameters_core_bert if f"encoder.layer.{layer}." in n 
                    and any(nd in n for nd in no_decay)]
        params_0_names = [n for n,p in named_parameters_core_bert if f"encoder.layer.{layer}." in n 
                    and any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in named_parameters_core_bert if f"encoder.layer.{layer}." in n 
                    and not any(nd in n for nd in no_decay)]
        params_1_names = [n for n,p in named_parameters_core_bert if f"encoder.layer.{layer}." in n 
                    and not any(nd in n for nd in no_decay)]
        
        for idx,layer in enumerate(params_1):
            layer.requires_grad = True
            print(f'{params_1_names[idx]} lr: {lr} weight_decay: {weight_decay_bert} {layer.requires_grad}')
        for idx,layer in enumerate(params_0):
            layer.requires_grad = True
            print(f'{params_0_names[idx]} lr: {lr} weight_decay: 0.0 {layer.requires_grad}')

        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)   
                            
        layer_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay_bert}
        opt_parameters.append(layer_params)   

        lr *= decay_lr    
        
    # === Embeddings layer ==========================================================
    
    if embeddings:

        params_0 = [p for n,p in named_parameters_core_bert if "embeddings" in n 
                    and any(nd in n for nd in no_decay)]
        params_0_names = [n for n,p in named_parameters_core_bert if "embeddings" in n 
                    and any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in named_parameters_core_bert if "embeddings" in n
                    and not any(nd in n for nd in no_decay)]
        params_1_names = [n for n,p in named_parameters_core_bert if "embeddings" in n
                    and not any(nd in n for nd in no_decay)]

        for idx,layer in enumerate(params_1):
            layer.requires_grad = True
            print(f'{params_1_names[idx]} lr: {lr} weight_decay: {weight_decay_bert} {layer.requires_grad}')
        for idx,layer in enumerate(params_0):
            layer.requires_grad = True
            print(f'{params_0_names[idx]} lr: {lr} weight_decay: 0.0 {layer.requires_grad}')

        embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0} 
        opt_parameters.append(embed_params)
            
        embed_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay_bert} 
        opt_parameters.append(embed_params)        


    # ==== Hier Layer =====================================================================

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"] + ['norm.weight']

    if len(named_parameters_hierBert)>0:

        for layer in named_parameters_hierBert:
            layer[1].requires_grad = False

        params_0 = [p for n,p in named_parameters_hierBert if any(nd in n for nd in no_decay)]
        params_0_names = [n for n,p in named_parameters_hierBert if any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in named_parameters_hierBert if not any(nd in n for nd in no_decay)]
        params_1_names = [n for n,p in named_parameters_hierBert if not any(nd in n for nd in no_decay)]

        for idx,layer in enumerate(params_1):
            layer.requires_grad = True
            print(f'{params_1_names[idx]} lr: {head_lr} weight_decay: {weight_decay_bert} {layer.requires_grad}')
        for idx,layer in enumerate(params_0):
            layer.requires_grad = True
            print(f'{params_0_names[idx]} lr: {head_lr} weight_decay: 0.0 {layer.requires_grad}')

        embed_params = {"params": params_0, "lr": head_lr, "weight_decay": 0.0} 
        opt_parameters.append(embed_params)
            
        embed_params = {"params": params_1, "lr": head_lr, "weight_decay": weight_decay_bert} 
        opt_parameters.append(embed_params)     

    # ==== Conv Layer =====================================================================

    if len(named_parameters_convs)>0:

        for layer in named_parameters_convs:
            layer[1].requires_grad = False

        params_0 = [p for n,p in named_parameters_convs if any(nd in n for nd in no_decay)]
        params_0_names = [n for n,p in named_parameters_convs if any(nd in n for nd in no_decay)]
        params_1 = [p for n,p in named_parameters_convs if not any(nd in n for nd in no_decay)]
        params_1_names = [n for n,p in named_parameters_convs if not any(nd in n for nd in no_decay)]

        for idx,layer in enumerate(params_1):
            layer.requires_grad = True
            print(f'{params_1_names[idx]} lr: {gcn_lr} weight_decay: {5e-4} {layer.requires_grad}')
        for idx,layer in enumerate(params_0):
            layer.requires_grad = True
            print(f'{params_0_names[idx]} lr: {gcn_lr} weight_decay: 0.0 {layer.requires_grad}')

        embed_params = {"params": params_0, "lr": gcn_lr, "weight_decay": 0.0} 
        opt_parameters.append(embed_params)
            
        embed_params = {"params": params_1, "lr": gcn_lr, "weight_decay": 5e-4} 
        opt_parameters.append(embed_params)   

    #check:
    #for n,p in model.named_parameters(): print(n,p.requires_grad)

    return opt_parameters


def set_learning_rates(base_lr,decay_lr,model,weight_decay,qtyFracLayers,gcn_lr):

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

    hierLayers = ['seg_pos_embeddings','seg_encoder']

    layer_names = []
    for idx, (name, param) in enumerate(model.named_parameters()):

        if 'seg_pos_embeddings' in name or 'seg_encoder' in name or 'gcn.convs' in name or 'gcn.batch_' in name:
            #aqui vai ter layer duplicada
            param.requires_grad = True #descongelar
            if 'gcn.batch_' in name:
                param.requires_grad = False
                continue
            continue
            #layer_names.append(name)

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

    extraNames = [name for (name, param) in model.named_parameters() if 'seg_pos_embeddings' in name or 'seg_encoder' in name]
    gcnNames = [name for (name, param) in model.named_parameters() if 'gcn.convs' in name]

    layer_names.reverse()
    #layer_names = [layer for layer in layer_names if 'seg_pos_embeddings' not in layer and 'seg_encoder' not in layer]
    layer_names = layer_names.copy() + extraNames.copy()
    #layer_names=list(set(layer_names))
    parameters = []
    namesLayer = []
    namesLayerGcn = []

    prev_group_name = re.search(r'layers?\.[0-9]+\.',layer_names[0])
    if prev_group_name == None:
        prev_group_name = layer_names[0].split('.')[-1] # a ideia aqui é pegar todos os pares weight/bias e manter igual
        pre_layerNum = None
    else:
        prev_group_name = prev_group_name.group()
        pre_layerNum = int(re.search(r'[0-9]',prev_group_name).group())

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
        
        # append layer parameters
        namesLayer += [{'names' : [n for n, p in model.named_parameters() if n == name and p.requires_grad]}]

        parameters += [{'params': [p for n, p in model.named_parameters() if n == name and p.requires_grad],
                        'lr':     base_lr,
                        'weight_decay': weight_decay}]

        #parameters += [{'params': [p for n, p in model.named_parameters() if n == name and p.requires_grad],
        #                'lr':     base_lr,
        #                'weight_decay': weight_decay}]
    
    if gcn_lr:

        parametersGCN = []
        for idx, name in enumerate(gcnNames):
            if 'convs.1' in name:
                namesLayerGcn += [{'names' : [n for n, p in model.named_parameters() if n == name and p.requires_grad]}]
                parametersGCN += [{'params': [p for n, p in model.named_parameters() if n == name and p.requires_grad],
                                'lr':     gcn_lr,
                                'weight_decay': 0}]
            elif 'batch_' in name:
                continue

            else:
                namesLayerGcn += [{'names' : [n for n, p in model.named_parameters() if n == name and p.requires_grad]}]
                parametersGCN += [{'params': [p for n, p in model.named_parameters() if n == name and p.requires_grad],
                    'lr':     gcn_lr,
                    'weight_decay': 5e-4}]


        parameters = parameters + parametersGCN
        namesLayer = namesLayer + namesLayerGcn

    for idx,n in enumerate(parameters):
        print(f'{namesLayer[idx]["names"][0]}, baselr: {n["lr"]}, weight_decay: {n["weight_decay"]}, grad: {True}')


    return parameters