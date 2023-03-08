from train.bertGcn_train import bertGcn_Train
from utils.helper_funs import read_experiments
from train.deep_train import deep_train
from train.gcn_train import gcn_train
from train.xgb_train import xgb_train
import torch
import gc
import mlflow
import pandas as pd
from cleanlab.filter import find_label_issues
#import shap

def _find_label_export_all(predictions_labels, supportDf, tokens):

    ordered_label_issues = find_label_issues(
        labels = supportDf['real_labels_ID'],
        pred_probs = predictions_labels,  # predicted probabilities from any model (ideally out-of-sample predictions) <==out-of-sample = fazer CV
        return_indices_ranked_by='self_confidence',
    )

    ordered_label_issues_df = pd.DataFrame(ordered_label_issues,columns= ['dataset_index'])
    ordered_label_issues_df['label_problem'] = "label_problem"
    ordered_label_issues_df = ordered_label_issues_df.reset_index()
    ordered_label_issues_df.rename(columns={'index':'label_error_ranking'},inplace=True)
    #colocar um index aq
    supportDf = supportDf.merge(ordered_label_issues_df, how='left', on=['dataset_index'])
    supportDf['label_problem'] = supportDf['label_problem'].fillna('OK')
    
    return supportDf

#mlflow ui --backend-store-uri sqlite:///mlflow.db
#tensorboard --logdir=./log/Hier_debug


def main():

    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    expDicDeep = read_experiments('run_experiments_deep.csv','deep_learning')
    expDicBoost = read_experiments('run_experiments_boost.csv','boost')
    expDicGCN = read_experiments('run_experiments_gcn.csv','gcn_learning')
    expDicBertGCN = read_experiments('run_experiments_bertgcn.csv','bertgc_learning')
    
    #mlflow.delete_experiment(experiment_id=3)
    #mlflow gc --backend-store-uri sqlite:///mlflow.db

    #deep training:
    for idx,experiment in enumerate(expDicDeep):

        print(f'Starting deep learning experiments: {idx+1}/{len(expDicDeep)}')
        print(f'Results are being logged in mlflow ...')

        expSubName = '[Deep_Hier]' if bool(experiment['hierarchical']) else '[Deep]'

        mlflow.set_experiment(expSubName + experiment['dataname'])

        with mlflow.start_run(run_name=experiment['descripton']):

            mlflow.log_params(experiment)

            train = deep_train(experiment)
            train.fit_and_eval()

            #multi-label might not work here#############:
            print("total infere started:")
            predictions_labels, supportDf, _ = train.infere_total()
            _find_label_export_all(predictions_labels, supportDf, _).to_csv(expSubName + experiment['dataname']+'.csv',index=False,encoding='Latin')
            ##############################################

        #explainer = shap.Explainer(train.model.model)
        #shap_values = explainer([1])

        del train
        gc.collect()
        torch.cuda.empty_cache()

    #xgboost training
    for idx,experiment in enumerate(expDicBoost):
        print(f'Starting xgboost experiments: {idx+1}/{len(expDicBoost)}')
        print(f'Results are being logged in mlflow ...')

        mlflow.set_experiment('[Boost]' + experiment['dataname'])

        boostExp = xgb_train(experiment)
        

        with mlflow.start_run(run_name=experiment['descripton']):
            mlflow.log_params(experiment)
            #colocar optuna:
            boostExp.fit_and_eval()

        del boostExp
        gc.collect()
        torch.cuda.empty_cache()

    #gcn training
    for idx,experiment in enumerate(expDicGCN):
        print(f'Starting GNN experiments: {idx+1}/{len(expDicGCN)}')
        print(f'Results are being logged in mlflow ...')

        mlflow.set_experiment('[GCN]'+experiment['dataname'])

        with mlflow.start_run(run_name=experiment['descripton']):
            gcnExp = gcn_train(experiment)
            gcnExp.fit_and_eval()

        del gcnExp
        gc.collect()
        torch.cuda.empty_cache()

    #bert gcn
    for idx,experiment in enumerate(expDicBertGCN):
        print(f'Starting Bert+GNN experiments: {idx+1}/{len(expDicBertGCN)}')
        print(f'Results are being logged in mlflow ...')

        expSubName = '[(H_B)GCN]' if bool(experiment['hierarchical']) else '[(B)GCN]'

        mlflow.set_experiment(expSubName+experiment['dataname'])

        with mlflow.start_run(run_name=experiment['descripton']):
            gcnExp = bertGcn_Train(experiment)
            gcnExp.fit_and_eval()

        del gcnExp
        gc.collect()
        torch.cuda.empty_cache()

    return None

if __name__ == '__main__':
    main()