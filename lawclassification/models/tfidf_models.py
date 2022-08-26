import spacy
import pandas as pd
import xgboost as xgb
import random
import json
import gc
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import os
from utils.definitions import ROOT_DIR

class xgb_tfidf():
    def __init__(self,expDic):
        super(xgb_tfidf, self).__init__()

        self.model = 'xgb'
        self.vectorizer = 'tfidf'

        self.model_name = self.model + self.vectorizer
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

        self.nlp = spacy.load('en_core_web_lg')

    def train(self):

        DMatrixTT = xgb.DMatrix(self.pathDM)
        DMatrixTest = xgb.DMatrix(self.pathTestDM)
        DMatrixTrain = xgb.DMatrix(self.pathTrainDM)
        DMatrixVal = xgb.DMatrix(self.pathValDM)

        if self.problem_type == 'single_label_classification':
            self.num_labels = len(np.unique(DMatrixTT.get_label()))
        else:
            self.num_labels = len(DMatrixTT.get_label()[0]) #???

        del DMatrixTT
        gc.collect()

        evallist = [(DMatrixTest, 'test'), (DMatrixTrain, 'train')]

        #classificadores:
        #multi:softmax => set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes)
        #mlogloss => logloss multiclass
        #binary:logistic => logistic regression for binary classification, output probability
        #logloss

        boosterParams = {'max_depth': 2, #more, more overfit
                        'min_split_loss' : 0, #larger, less overfitting
                        'learning_rate': 0.3, #larger, less overfitting
                        'objective': 'multi:softmax', 
                        'reg_lambda':1, #larger, less overfitting, L2 regu
                        'alpha':1, #larger, less overfitting, L1 regu
                        'num_class' : self.num_labels, #colocar aqui quando for multi-label
                        'max_bin' : 256, #Increasing this number improves the optimality of splits at the cost of higher computation time.
                        'tree_method': 'gpu_hist',
                        'predictor': 'gpu_predictor',
                        'eval_metric': 'mlogloss'}

        evals_result = {}

        #colocar optuna e mlflow:
        bst = xgb.train(params=boosterParams,
                        dtrain=DMatrixTrain,
                        num_boost_round=1000,
                        evals=evallist,
                        early_stopping_rounds=10,
                        evals_result=evals_result)

        preds = bst.predict(DMatrixTest)
        labels = DMatrixTest.get_label()
        print('error=%f' %
            (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) /
            float(len(preds))))

    def _tokenizer(self,text):

        newText = []

        for word in self.nlp(text):
            if (not word.is_stop or not word.is_punct or not word.is_oov) and (word.is_alpha and word.is_ascii):
                newText.append(word.lemma_.lower())

        return newText

    def csv_to_dm(self):

        if os.path.exists(self.pathDM):
            print('Found .buffer file!')
            return None

        print('Creating new .buffer for xgboost files vectorized might take a while ...')

        if self.problem_type == 'single_label_classification':
            dfTestCSV = pd.read_csv(self.pathTestCSV)
            dfTrainCSV = pd.read_csv(self.pathTrainCSV)
            dfValCSV = pd.read_csv(self.pathTrainCSV)
        else:
            dfTestCSV = pd.read_csv(self.pathTestCSV)
            dfTrainCSV = pd.read_csv(self.pathTrainCSV)
            dfValCSV = pd.read_csv(self.pathTrainCSV)
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
                                       ngram_range=(1,1),
                                       lowercase = True,
                                       strip_accents = 'unicode',
                                       encoding = 'utf-8',
                                       decode_error = 'strict',
                                       max_df = 1.0, #tirar 'outliers' palavras muito repetidas
                                       min_df = 1, #tirar 'outliers' palavras muito raras #'cut-off'
                                       max_features = None, #numero maximo de features
                                       vocabulary = None) #da subir o vocab do bert aqui

        DMatrixTT = xgb.DMatrix(tfidf_vector.fit_transform(dfCorpus['text']),
                                label=dfCorpus['labels'],
                                feature_names=tfidf_vector.get_feature_names_out())

        del dfCorpus
        gc.collect()

        DMatrixTT.save_binary(self.pathDM)
        DMatrixTT.slice(range(dfDicLen['train']['start'],dfDicLen['train']['end'])).save_binary(self.pathTrainDM)
        DMatrixTT.slice(range(dfDicLen['test']['start'],dfDicLen['test']['end'])).save_binary(self.pathTestDM)
        DMatrixTT.slice(range(dfDicLen['val']['start'],dfDicLen['val']['end'])).save_binary(self.pathValDM)