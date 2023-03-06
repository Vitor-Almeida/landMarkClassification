import pandas as pd
from transformers import BertTokenizerFast
import os
from utils.definitions import ROOT_DIR
import spacy
import re

def _easy_sentencesplit(text,maxlenghsen):

    tokenList = text.split()
    innerList = []

    if len(tokenList) > maxlenghsen+2:
        rangeWin = range(0, len(tokenList)-maxlenghsen+1, maxlenghsen)
    else:
        rangeWin = range(0, 1, maxlenghsen)

    for i in rangeWin:
        innerList.append(' '.join(tokenList[i:i+maxlenghsen]))

    return innerList


def _hiersentencetokenizer(df:pd.DataFrame,hier_max_seg:int,hier_max_seg_length:int,dataname:str,modelname:str):

    bertTokenizer = BertTokenizerFast.from_pretrained(os.path.join(ROOT_DIR,'lawclassification','models','external',modelname))
    #bertTokenizer = BertTokenizer.from_pretrained(os.path.join(ROOT_DIR,'lawclassification','models','external','bert-base-uncased'))
    #bertTokenizer = BertTokenizer.from_pretrained(os.path.join(ROOT_DIR,'lawclassification','models','external','ulysses-camara.legal-bert-pt-br'))
    #bertTokenizer = BertTokenizer.from_pretrained(os.path.join(ROOT_DIR,'lawclassification','models','external','neuralmind.bert-base-portuguese-cased'))

    NLP = spacy.load('en_core_web_lg')

    input_idslist = []
    attention_masklist = []
    token_type_idslist = []

    def hier(row,hier_max_seg_length):

        if dataname in ['scotus_lexbench']:
            #text1 = re.split('\n{2,}', row)
            text1 = _easy_sentencesplit(row,hier_max_seg_length)
        elif dataname in ['ecthr_a_lexbench','ecthr_b_lexbench']:
            text1 = _easy_sentencesplit(row,hier_max_seg_length)
        else:
            text1 = _easy_sentencesplit(row,hier_max_seg_length)

        case_encodings = bertTokenizer(text1[:hier_max_seg], padding='max_length', max_length = hier_max_seg_length, add_special_tokens=True , truncation=True, return_attention_mask=True)

        inputIds = case_encodings['input_ids'] + [[0] * hier_max_seg_length] * (hier_max_seg - len(case_encodings['input_ids']))
        attentionMask = case_encodings['attention_mask'] + [[0] * hier_max_seg_length] * (hier_max_seg - len(case_encodings['attention_mask']))
        tokenTypeIds = case_encodings['token_type_ids'] + [[0] * hier_max_seg_length] * (hier_max_seg - len(case_encodings['token_type_ids']))

        #return [inputIds,attentionMask,tokenTypeIds]

        return {'input_ids' : inputIds,
                'attention_mask': attentionMask,
                'token_type_ids': tokenTypeIds
                }

    for row in df['text']:
        dict = hier(row,hier_max_seg_length)
        input_idslist.append(dict['input_ids'])
        attention_masklist.append(dict['attention_mask'])
        token_type_idslist.append(dict['token_type_ids'])

    return [input_idslist,attention_masklist,token_type_idslist]

def _deeptokenizer(df:pd.DataFrame,modelname:str):

    bertTokenizer = BertTokenizerFast.from_pretrained(os.path.join(ROOT_DIR,'lawclassification','models','external',modelname))
    #bertTokenizer = BertTokenizer.from_pretrained(os.path.join(ROOT_DIR,'lawclassification','models','external','bert-base-uncased'))
    #bertTokenizer = BertTokenizer.from_pretrained(os.path.join(ROOT_DIR,'lawclassification','models','external','ulysses-camara.legal-bert-pt-br'))
    #bertTokenizer = BertTokenizer.from_pretrained(os.path.join(ROOT_DIR,'lawclassification','models','external','neuralmind.bert-base-portuguese-cased'))

    input_idslist = []
    attention_masklist = []
    token_type_idslist = []

    for row in df['text']:

        batch = bertTokenizer(row,
                              padding='max_length',
                              add_special_tokens=True,
                              max_length=512,
                              return_attention_mask=True,
                              truncation=True)

        input_idslist.append(batch['input_ids'])
        attention_masklist.append(batch['attention_mask'])
        token_type_idslist.append(batch['token_type_ids'])

    return [input_idslist,attention_masklist,token_type_idslist]

def append_token_hier(dataname:str,modelname:str,hier_max_seg:int,hier_max_seg_length:int):
    
    dfTrain = pd.read_csv(os.path.join(ROOT_DIR,'data',dataname,'interm','train','train.csv'))
    dfTrain['split'] = 'train'
    dfTest = pd.read_csv(os.path.join(ROOT_DIR,'data',dataname,'interm','test','test.csv'))
    dfTest['split'] = 'test'
    dfVal = pd.read_csv(os.path.join(ROOT_DIR,'data',dataname,'interm','val','val.csv'))
    dfVal['split'] = 'val'

    df = pd.concat([dfTrain,dfTest,dfVal],ignore_index=True)

    encodedSenDeep = _hiersentencetokenizer(df,hier_max_seg,hier_max_seg_length,dataname,modelname)
    df['token_s_hier_id'] = encodedSenDeep[0]
    df['token_s_hier_att'] = encodedSenDeep[1]
    df['token_s_hier_tid'] = encodedSenDeep[2]

    encodedTextDeep = _deeptokenizer(df,modelname)
    df['token_w_hier_id'] = encodedTextDeep[0]
    df['token_w_hier_att'] = encodedTextDeep[1]
    df['token_w_hier_tid'] = encodedTextDeep[2]

    df = df.reset_index(drop=True)
    df = df.reset_index()
    df['dataset_index'] = df['index']
    df.drop(columns=['index'],inplace=True)

    df[df['split']=='train'].to_csv(os.path.join(ROOT_DIR,'data',dataname,'interm','train','train.csv'),index=False)
    df[df['split']=='test'].to_csv(os.path.join(ROOT_DIR,'data',dataname,'interm','test','test.csv'),index=False)
    df[df['split']=='val'].to_csv(os.path.join(ROOT_DIR,'data',dataname,'interm','val','val.csv'),index=False)

    df = df[['text','split','dataset_index']]

    df.to_csv(os.path.join(ROOT_DIR,'data',dataname,'interm','index_text_lookup.csv'),index=False)

    return None