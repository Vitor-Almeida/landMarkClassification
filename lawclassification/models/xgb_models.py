import random
from typing import Dict, Tuple, List
from dataset.xgbdata_load import csv_to_dmMatrix

class xgb_models():
    def __init__(self, expDicBoost):
        super(xgb_models, self).__init__()

        self.dataname = expDicBoost['dataname']
        self.problemType = expDicBoost['problem_type']
        self.DMatrixTest, self.DMatrixTrain, self.DMatrixVal, self.numLabels, self.numLabelsMetrics = csv_to_dmMatrix(self.dataname,self.problemType)
        self.TestSize = self.DMatrixTest.num_row()
        self.TrainSize = self.DMatrixTrain.num_row()
        self.ValSize = self.DMatrixVal.num_row()

        #gambiarra para conseguir logar coisas custom mlflow com o xgboost:
        assert(self.TestSize != self.TrainSize and self.TestSize != self.TrainSize and self.TrainSize != self.ValSize)

        #fazer cross validation?

        boosterParams = {'single_label_classification':{'max_depth': 2, #more, more overfit
                                                        'min_split_loss' : 0, #larger, less overfitting
                                                        'learning_rate': 0.3, #larger, less overfitting
                                                        'seed':random.randint(0, 1000),
                                                        'reg_lambda':1, #larger, less overfitting, L2 regu
                                                        'alpha':1, #larger, less overfitting, L1 regu
                                                        'disable_default_eval_metric':1,
                                                        'num_class' : self.numLabels,
                                                        'max_bin' : 256, #Increasing this number improves the optimality of splits at the cost of higher computation time.
                                                        'tree_method': 'gpu_hist',
                                                        'predictor': 'gpu_predictor',
                                                        'objective': 'multi:softprob'  #multi:softprob
                                                        #'eval_metric': ['mlogloss','merror'] #accuracia
                                                        }, #last one is used for early stopping

                        'multi_label_classification':{  'max_depth': 2, #more, more overfit
                                                        'min_split_loss' : 0, #larger, less overfitting
                                                        'learning_rate': 0.3, #larger, less overfitting
                                                        'seed':random.randint(0, 1000),
                                                        'reg_lambda':1, #larger, less overfitting, L2 regu
                                                        'alpha':1, #larger, less overfitting, L1 regu
                                                        'disable_default_eval_metric':1,
                                                        #'num_class' : self.numLabels, #colocar aqui quando for multi-label
                                                        'num_target' : self.numLabels,
                                                        'max_bin' : 256, #Increasing this number improves the optimality of splits at the cost of higher computation time.
                                                        'tree_method': 'gpu_hist',
                                                        'predictor': 'gpu_predictor',
                                                        #'objective': 'multi:softprob',  #multi:softmax
                                                        #'eval_metric': ['mlogloss','merror']
                                                        'objective': 'binary:logitraw' 
                                                        #'eval_metric': ['logloss','error']
                                                        }} #last one is used for early stopping

        self.boosterParams = boosterParams[self.problemType]