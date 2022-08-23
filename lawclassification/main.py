from utils.helper_funs import read_experiments
from train.deep_train import deep_train
import gc
import mlflow

#possible models:
#bert-base-uncased
#albert-base-v2
#allenai.longformer-base-4096
#distilbert-base-uncased
#nlpaueb.legal-bert-base-uncased
#roberta-base
#saibo.legal-roberta-base
#CaseLaw-BERT => testar

#possible datasets:
#yelp;single_label_classification
#SemEval2018-Task1-all-data;multi_label_classification
#eurlex57k;multi_label_classification
#echr;multi_label_classification
#dmoz_4090;single_label_classification
#dmoz_1500;single_label_classification
#dmoz_510_1500;single_label_classification
#
#ecthr_a_lexbench;multi_label_classification
#ecthr_b_lexbench;multi_label_classification
#eurlex_lexbench;multi_label_classification
#ledgar_lexbench;single_label_classification
#scotus_lexbench;single_label_classification
#unfair_lexbench;multi_label_classification

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