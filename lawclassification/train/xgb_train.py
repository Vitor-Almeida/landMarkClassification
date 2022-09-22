from sys import maxsize
import xgboost as xgb
from typing import Dict, Tuple, List
from models.xgb_models import xgb_models
import numpy as np
import torch
import torchmetrics
import mlflow
from utils.deep_metrics import metrics_config

class xgb_train():
    def __init__(self,expDicBoost):
        super(xgb_train, self).__init__()

        self.numBoostRound = expDicBoost['num_boost_round']
        self.goalMetricName = expDicBoost['goal_metric']

        self.xgbModel = xgb_models(expDicBoost)

        self.evallist = [(self.xgbModel.DMatrixTrain, 'Train_Epoch'), 
                         (self.xgbModel.DMatrixVal, 'Val_Epoch'), 
                         (self.xgbModel.DMatrixTest, 'Test_Epoch')] #last one is used for early stopping
        
        self.evals_result = {}

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        _ = metrics_config(num_labels = self.xgbModel.numLabelsMetrics,
                           device = self.device,
                           problem_type = self.xgbModel.problemType)

        self.metricsGoal = _.clone('Goal_')

        self.metricsGoal = self.metricsGoal[self.goalMetricName]
        self.metricsTrainEpoch = _.clone(prefix='Train_Epoch_')
        self.metricsTestEpoch = _.clone(prefix='Test_Epoch_')
        self.metricsValEpoch = _.clone(prefix='Val_Epoch_')

        #colocar optuna:

    def fit_and_eval(self):

        bst = xgb.train(params=self.xgbModel.boosterParams,
                        dtrain=self.xgbModel.DMatrixTrain,
                        num_boost_round=self.numBoostRound,
                        verbose_eval=False,
                        evals=self.evallist,
                        custom_metric=self._custom_goal_metric, #slow
                        maximize=True,
                        early_stopping_rounds=10,
                        evals_result=self.evals_result)

        preds = bst.predict(self.xgbModel.DMatrixTest, iteration_range=(0, bst.best_iteration + 1))
        #preds = bst.predict(DMatrixTest)
        labels = self.xgbModel.DMatrixTest.get_label()

    def _custom_goal_metric(self, predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
        metric = self.metricsGoal.reset()
        predt = torch.tensor(predt).to(self.device)

        if self.xgbModel.problemType == 'multi_label_classification':
            y = torch.tensor(dtrain.get_label().reshape(predt.shape)).int().to(self.device)
        else:
            y = torch.tensor(dtrain.get_label()).int().to(self.device)
        
        metric = self.metricsGoal(predt,y)
        self._calculate_metrics(dtrain,predt,y)

        return ('GOAL'+self.goalMetricName, metric.item())

    def _calculate_metrics(self,dtrain,predt,y):
        self.metricsTestEpoch.reset()
        self.metricsTrainEpoch.reset()
        self.metricsValEpoch.reset()
        
        if dtrain.num_row() == self.xgbModel.TestSize:
            metric = self.metricsTestEpoch(predt,y)
            mlflow.log_metrics({label:round(value.item(),4) for label, value in metric.items()})

        elif dtrain.num_row() == self.xgbModel.TrainSize:
            metric = self.metricsTrainEpoch(predt,y)
            mlflow.log_metrics({label:round(value.item(),4) for label, value in metric.items()})

        elif dtrain.num_row() == self.xgbModel.ValSize:
            metric = self.metricsValEpoch(predt,y)
            mlflow.log_metrics({label:round(value.item(),4) for label, value in metric.items()})

        return None
