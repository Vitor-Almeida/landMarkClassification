from utils.helper_funs import read_experiments
from train.deep_train import deep_train
import models.tfidf_models as boost
import gc
import mlflow

#mlflow ui --backend-store-uri sqlite:///mlflow.db

def main():

    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    expDicDeep = read_experiments('run_experiments_deep.csv','deep_learning')
    expDicBoost = read_experiments('run_experiments_boost.csv','boost')
    
    #mlflow.delete_experiment(experiment_id=3)
    #mlflow gc --backend-store-uri sqlite:///mlflow.db

    #deep training:
    for idx,experiment in enumerate(expDicDeep):

        print(f'begin of deep experiment: {idx+1}/{len(expDicDeep)}')
        print(f'Olhar resultados no mlflow !')

        #outro tipo de experimento poderia ser dataname+modelo+'busca hyper'
        mlflow.set_experiment(experiment['dataname'])

        with mlflow.start_run(run_name=experiment['descripton']):

            mlflow.log_params(experiment)

            train = deep_train(experiment)
            train.fit_and_eval()

            #da um log no melhor modelo:
            #mlflow.pytorch.log_model(train.model.model, "model")

        del train
        gc.collect()

    
    for idx,experiment in enumerate(expDicBoost):
        print(f'begin of boost experiment: {idx+1}/{len(expDicBoost)}')
        print(f'Olhar resultados no mlflow !')

        boostExp = boost.xgb_tfidf(experiment)
        boostExp.csv_to_dm()
        boostExp.train()

        del boostExp
        gc.collect()

    return None

if __name__ == '__main__':
    main()