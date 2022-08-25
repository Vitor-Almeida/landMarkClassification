from utils.helper_funs import read_experiments
from train.deep_train import deep_train
import gc
import mlflow

#mlflow ui --backend-store-uri sqlite:///mlflow.db

def main():

    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    experimentsDic = read_experiments('run_experiments.csv')
    
    #mlflow.delete_experiment(experiment_id=3)
    #mlflow gc --backend-store-uri sqlite:///mlflow.db

    for idx,experiment in enumerate(experimentsDic):

        print(f'begin of experiment: {idx+1}/{len(experimentsDic)}')
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

    return None

if __name__ == '__main__':
    main()