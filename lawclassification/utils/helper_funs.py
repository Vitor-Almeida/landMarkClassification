
from utils.definitions import ROOT_DIR
import os
import pandas as pd

def print_step_log(idx,cur_epoch,model,metricsViews):

    logInterval = int(model.total_steps/50)

    if idx % logInterval == 0 and idx > 0:
        qtyToFormat = len(metricsViews)
        str_to_format = []

        for pos in range(0,qtyToFormat):
            string = ['{',str(pos),'}']
            string = ''.join(string)
            str_to_format.append(string)
        str_to_format = '   '.join(str_to_format)

        str_gamb = []

        for views in metricsViews:
            string = "'"+str(views)+": "+str(metricsViews[views])+"'"
            str_gamb.append(string)
        str_gamb = ','.join(str_gamb)

        str_end = str_to_format.format(*eval(str_gamb))

        print(f'| epoch:{cur_epoch+1} | {idx}/{len(model.train_dataloader)} batches | {str_end}')
    else:
        return None


def read_experiments():

    path = os.path.join(ROOT_DIR,'lawclassification','run_experiments.csv')

    df = pd.read_csv(path)

    return df.to_dict(orient='records')

def print_batch_log(cur_epoch,model,metricsViews):

    qtyToFormat = len(metricsViews)
    str_to_format = []

    for pos in range(0,qtyToFormat):
        string = ['{',str(pos),'}']
        string = ''.join(string)
        str_to_format.append(string)
    str_to_format = '   '.join(str_to_format)

    str_gamb = []

    for views in metricsViews:
        string = "'"+str(views)+": "+str(metricsViews[views])+"'"
        str_gamb.append(string)
    str_gamb = ','.join(str_gamb)

    str_end = str_to_format.format(*eval(str_gamb))

    print(f'|end of epoch:{cur_epoch+1} | {len(model.train_dataloader)}/{len(model.train_dataloader)} batches | {str_end}')


def print_params_terminal(model):

    dataset_length = len(model.dataset_test) + len(model.dataset_train) + len(model.dataset_val)
    train_labels = model.num_labels_train
    test_labels = model.num_labels_test
    val_labels = model.num_labels_val

    print('-'*59)
    print('Parameters:')
    print(f'modelo: {model.model_name} | batchsize: {model.batchsize} | max_tokens = {model.max_char_length} | learning_rate = {model.lr}')
    print(f'epochs = {model.epochs} | warmup_size = {model.warmup_size} | dropout = {model.dropout}')
    print(f'num_labels = {train_labels} | dataset_length = {dataset_length} | dataset_name = {model.dataname}')
    print(f'random_seed = {model.seed_val} | train_length = {len(model.dataset_train)} | train_labels = {train_labels}')
    print(f'test_length = {len(model.dataset_test)} | test_labels = {test_labels} | val_length = {len(model.dataset_val)} | test_labels = {val_labels}')
    print('-'*59)

    return None