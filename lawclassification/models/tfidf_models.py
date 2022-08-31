import spacy
import pandas as pd
import xgboost as xgb
import random
import json
import torch
from typing import Dict, Tuple, List
import gc
import numpy as np
import torchmetrics
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from utils.definitions import ROOT_DIR

class xgb_tfidf():
    def __init__(self,expDic):
        super(xgb_tfidf, self).__init__()

        self.model = 'xgb'
        self.vectorizer = 'tfidf'

        self.model_name = self.model + '_' + self.vectorizer
        self.dataname = expDic['dataname']
        self.problem_type = expDic['problem_type']

        self.path = os.path.join(ROOT_DIR,'lawclassification','data',self.dataname,'interm')
        self.pathDM = os.path.join(ROOT_DIR,'data',self.dataname,'interm',self.model_name+'_tt.buffer')
        self.pathTestDM = os.path.join(ROOT_DIR,'data',self.dataname,'interm','test',self.model_name+'_test.buffer')
        self.pathTrainDM = os.path.join(ROOT_DIR,'data',self.dataname,'interm','train',self.model_name+'_train.buffer')
        self.pathValDM = os.path.join(ROOT_DIR,'data',self.dataname,'interm','val',self.model_name+'_val.buffer')

        self.pathTestCSV = os.path.join(ROOT_DIR,'data',self.dataname,'interm','test','test.csv')
        self.pathTrainCSV= os.path.join(ROOT_DIR,'data',self.dataname,'interm','train','train.csv')
        self.pathValDMCSV = os.path.join(ROOT_DIR,'data',self.dataname,'interm','val','val.csv')

        self.accuracy = torchmetrics.Accuracy(average='micro', 
                                              threshold = 0.5, 
                                              subset_accuracy = True).to('cuda:0')

        self.nlp = spacy.load('en_core_web_lg')
        #setar token

    def train(self):

        DMatrixTT = xgb.DMatrix(self.pathDM)
        DMatrixTest = xgb.DMatrix(self.pathTestDM)
        DMatrixTrain = xgb.DMatrix(self.pathTrainDM)
        DMatrixVal = xgb.DMatrix(self.pathValDM)

        if self.problem_type == 'single_label_classification':
            self.num_labels = len(np.unique(DMatrixTT.get_label()))
        else:
            self.num_labels = 1
            #try to fix:
            #dfTrainCSV = pd.read_csv(self.pathTrainCSV,nrows=2)
            #dfTrainCSV['labels'] = dfTrainCSV['labels'].apply(lambda row: json.loads(row))
            #self.num_labels = len(dfTrainCSV['labels'][0]) #???
            #self.num_labels = 1
            #del dfTrainCSV
            #gc.collect()

        del DMatrixTT
        gc.collect()

        #fazer cross validation?

        evallist = [(DMatrixTrain, 'train'), (DMatrixVal, 'val'), (DMatrixTest, 'test')] #last one is used for early stopping

        boosterParams = {'max_depth': 2, #more, more overfit
                        'min_split_loss' : 0, #larger, less overfitting
                        'learning_rate': 0.3, #larger, less overfitting
                        'seed':random.randint(0, 1000),
                        'reg_lambda':1, #larger, less overfitting, L2 regu
                        'alpha':1, #larger, less overfitting, L1 regu
                        #'num_class' : self.num_labels, #colocar aqui quando for multi-label
                        'num_target' : self.num_labels,
                        'max_bin' : 256, #Increasing this number improves the optimality of splits at the cost of higher computation time.
                        'tree_method': 'gpu_hist',
                        'predictor': 'gpu_predictor',
                        #'objective': 'multi:softprob',  #multi:softmax
                        #'eval_metric': ['mlogloss','merror']
                        'objective': 'binary:logistic',  
                        #'eval_metric': ['logloss','error']
                        } #last one is used for early stopping

        evals_result = {}

        #colocar optuna:
        bst = xgb.train(params=boosterParams,
                        dtrain=DMatrixTrain,
                        num_boost_round=200,
                        custom_metric=self._sub_acc, #tirar para multiclass
                        evals=evallist,
                        early_stopping_rounds=10,
                        evals_result=evals_result)

        preds = bst.predict(DMatrixTest, iteration_range=(0, bst.best_iteration + 1))
        #preds = bst.predict(DMatrixTest)
        labels = DMatrixTest.get_label()

    def _sub_acc(self, predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
        y = dtrain.get_label().reshape(predt.shape)
        acc = self.accuracy(torch.tensor(predt).cuda(),torch.tensor(y).int().cuda())
        return "acc_py_metrics", acc

    def _tokenizer(self,text):

        text = text[0:3000] #~512 words

        newText = []

        #slow:
        for word in self.nlp(text):
            if (not word.is_stop or not word.is_punct) and (word.is_alpha and word.is_ascii and not word.is_digit and word.ent_iob_ == 'O'):
                newText.append(word.lemma_.lower())

        return newText

    def csv_to_dm(self):

        if os.path.exists(self.pathDM):
            print('Found .buffer file!')
            return None

        print('Creating new .buffer for vectorized xgboost files, this might take a while ...')

        dfTestCSV = pd.read_csv(self.pathTestCSV)
        dfTrainCSV = pd.read_csv(self.pathTrainCSV)
        dfValCSV = pd.read_csv(self.pathTrainCSV)

        if self.problem_type == 'multi_label_classification':
            dfTestCSV['labels'] = dfTestCSV['labels'].apply(lambda row: json.loads(row))
            dfTrainCSV['labels'] = dfTrainCSV['labels'].apply(lambda row: json.loads(row))
            dfValCSV['labels'] = dfValCSV['labels'].apply(lambda row: json.loads(row))

        dfDicLen = {'test':{'start':0,'end':len(dfTestCSV)},
                    'train':{'start':len(dfTestCSV),'end':len(dfTestCSV)+len(dfTrainCSV)},
                    'val':{'start':len(dfTestCSV)+len(dfTrainCSV),'end':len(dfTestCSV)+len(dfTrainCSV)+len(dfValCSV)}}

        dfCorpus = pd.concat([dfTestCSV,dfTrainCSV,dfValCSV],ignore_index=True)

        del dfTestCSV
        del dfTrainCSV
        del dfValCSV
        gc.collect()

        tfidf_vector = TfidfVectorizer(tokenizer = self._tokenizer,
                                       #stop_words = 'english', 
                                       #sublinear_tf = True, 
                                       ngram_range=(1,1),
                                       lowercase = True,
                                       strip_accents = 'unicode',
                                       encoding = 'utf-8',
                                       decode_error = 'strict',
                                       dtype = np.float32,
                                       max_df = 1.0, #tirar 'outliers' palavras muito repetidas entre documentos
                                       min_df = 1, #tirar 'outliers' palavras muito raras entre documentos#'cut-off'
                                       max_features = None, #numero maximo de features
                                       vocabulary = None) #da subir o vocab do bert aqui

        if self.problem_type == 'multi_label_classification':
            labelMulti = np.zeros((1,len(dfCorpus['labels'][0])))
            #fix => dont loop over dataframes
            for row in dfCorpus['labels']:
                labelMulti=np.append(labelMulti,np.array([row]),axis=0)
            labelMulti = labelMulti[1:]
            label = labelMulti
        else:
            label = dfCorpus['labels']

        DMatrixTT = xgb.DMatrix(tfidf_vector.fit_transform(dfCorpus['text']),
                                label=label,
                                feature_names=tfidf_vector.get_feature_names_out())

        del dfCorpus
        gc.collect()

        DMatrixTT.save_binary(self.pathDM)
        DMatrixTT.slice(range(dfDicLen['train']['start'],dfDicLen['train']['end'])).save_binary(self.pathTrainDM)
        DMatrixTT.slice(range(dfDicLen['test']['start'],dfDicLen['test']['end'])).save_binary(self.pathTestDM)
        DMatrixTT.slice(range(dfDicLen['val']['start'],dfDicLen['val']['end'])).save_binary(self.pathValDM)