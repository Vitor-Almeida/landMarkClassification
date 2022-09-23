import pandas as pd
import xgboost as xgb
from utils.helper_funs import hug_tokenizer
from typing import Dict, Tuple, List
import gc
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from utils.definitions import ROOT_DIR

def _dummytfidf(doc):
    return doc

def _tokenizer(df:pd.DataFrame, vocab_size:int):

    bertTokenizer , trainer = hug_tokenizer(vocab_size)

    bertTokenizer.train_from_iterator(df['text'], trainer=trainer)

    vocabToId = bertTokenizer.get_vocab()
    vocabVec = list(bertTokenizer.get_vocab().keys())
    idToVocab = {idx:label for label,idx in vocabToId.items()}

    encodedText = df['text'].apply(lambda row: bertTokenizer.encode(row,add_special_tokens=False).ids)

    vocabMaps = [vocabVec,vocabToId,idToVocab]

    return encodedText, vocabMaps

def csv_to_dmMatrix(dataname:str,problem_type:str):

    model = 'xgb'
    vectorizer = 'tfidf'

    model_name = model + '_' + vectorizer
    dataname = dataname
    problem_type = problem_type

    pathDM = os.path.join(ROOT_DIR,'data',dataname,'interm',model_name+'_tt.buffer')
    pathTestDM = os.path.join(ROOT_DIR,'data',dataname,'interm','test',model_name+'_test.buffer')
    pathTrainDM = os.path.join(ROOT_DIR,'data',dataname,'interm','train',model_name+'_train.buffer')
    pathValDM = os.path.join(ROOT_DIR,'data',dataname,'interm','val',model_name+'_val.buffer')

    pathTestCSV = os.path.join(ROOT_DIR,'data',dataname,'interm','test','test.csv')
    pathTrainCSV= os.path.join(ROOT_DIR,'data',dataname,'interm','train','train.csv')
    pathValDMCSV = os.path.join(ROOT_DIR,'data',dataname,'interm','val','val.csv')

    numLabelsMetrics = 0

    if not os.path.exists(pathDM):

        print('Creating new .buffer for vectorized xgboost files, this might take a while ...')

        dfTestCSV = pd.read_csv(pathTestCSV)
        dfTrainCSV = pd.read_csv(pathTrainCSV)
        dfValCSV = pd.read_csv(pathValDMCSV)

        if problem_type == 'multi_label_classification':
            dfTestCSV['labels'] = dfTestCSV['labels'].apply(eval)
            dfTrainCSV['labels'] = dfTrainCSV['labels'].apply(eval)
            dfValCSV['labels'] = dfValCSV['labels'].apply(eval)

        dfDicLen = {'test':{'start':0,'end':len(dfTestCSV)},
        'train':{'start':len(dfTestCSV),'end':len(dfTestCSV)+len(dfTrainCSV)},
        'val':{'start':len(dfTestCSV)+len(dfTrainCSV),'end':len(dfTestCSV)+len(dfTrainCSV)+len(dfValCSV)}}

        dfCorpus = pd.concat([dfTestCSV,dfTrainCSV,dfValCSV],ignore_index=True)

        del dfTestCSV
        del dfTrainCSV
        del dfValCSV
        gc.collect()

        dfCorpus['text_token'], vocabMaps = _tokenizer(dfCorpus,vocab_size=200000)

        tfidfVector = TfidfVectorizer(#ngram_range=(1,1), #artigo lexbench usa (1,3)
                                      dtype = np.float32,
                                      preprocessor=_dummytfidf,
                                      tokenizer=_dummytfidf,
                                      token_pattern=None)

        if problem_type == 'multi_label_classification':
            labelMulti = np.zeros((1,len(dfCorpus['labels'][0])))
            #fix => dont loop over dataframes
            for row in dfCorpus['labels']:
                labelMulti=np.append(labelMulti,np.array([row]),axis=0)
            label = labelMulti[1:]
            numLabels = 1
            numLabelsMetrics = len(dfCorpus['labels'][0])
        else:
            label = dfCorpus['labels']
            numLabels = len(np.unique(dfCorpus['labels']))
            numLabelsMetrics = numLabels

        DMatrixTT = xgb.DMatrix(tfidfVector.fit_transform(dfCorpus['text_token']),
                    label=label,
                    feature_names=[str(n) for n in tfidfVector.get_feature_names_out().tolist()]
                    )

        del dfCorpus
        gc.collect()

        DMatrixTT.save_binary(pathDM)
        DMatrixTrain = DMatrixTT.slice(range(dfDicLen['train']['start'],dfDicLen['train']['end']))
        DMatrixTest = DMatrixTT.slice(range(dfDicLen['test']['start'],dfDicLen['test']['end']))
        DMatrixVal = DMatrixTT.slice(range(dfDicLen['val']['start'],dfDicLen['val']['end']))

        DMatrixTrain.save_binary(pathTrainDM)
        DMatrixTest.save_binary(pathTestDM)
        DMatrixVal.save_binary(pathValDM)

    else:

        DMatrixTT = xgb.DMatrix(pathDM)
        DMatrixTest = xgb.DMatrix(pathTestDM)
        DMatrixTrain = xgb.DMatrix(pathTrainDM)
        DMatrixVal = xgb.DMatrix(pathValDM)

        if problem_type == 'single_label_classification':
            numLabels = len(np.unique(DMatrixTT.get_label()))
            numLabelsMetrics = numLabels
        else:
            numLabels = 1
            numLabelsMetrics = int(len(DMatrixTT.get_label())/DMatrixTT.num_row())

    
    return DMatrixTest, DMatrixTrain, DMatrixVal, numLabels, numLabelsMetrics