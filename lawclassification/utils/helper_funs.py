from torchmetrics import Accuracy,ConfusionMatrix,Precision,F1Score

def metrics_config(model):

    accuracy_accu = Accuracy(num_classes=model.num_labels, average='micro', threshold = 0.5, mdmc_average = 'global',top_k=1).to(model.device)
    conf_matrix = ConfusionMatrix(num_classes=model.num_labels,normalize = None, threshold  = 0.5).to(model.device)
    precision = Precision(num_classes=model.num_labels, average='micro', threshold  = 0.5, mdmc_average = 'global',top_k=1).to(model.device)
    f1score = F1Score(num_classes=model.num_labels, average='micro', threshold  = 0.5, mdmc_average = 'global',top_k=1).to(model.device)

    mDict = {'accuracy_accu':accuracy_accu,'conf_matrix':conf_matrix,'precision':precision,'f1score':f1score}

    return mDict

def printLog(idx,cur_epoch,model,accuracy):

    log_interval = int(model.total_steps/50)

    if idx == -1:
        print('| end of epoch {:3d} '
        '| test accuracy {:8.3f}'.format(cur_epoch+1, accuracy))

    if idx % log_interval == 0 and idx > 0:
        print('| epoch {:3d} | {:5d}/{:5d} batches '
                '| train accuracy {:8.3f}'.format(cur_epoch+1, idx, len(model.train_dataloader),
                                            accuracy))

    return None

def printParamsTerminal(model):

    print('-'*59)
    print('Parameters:')
    print(f'modelo: {model.model_name} | batchsize: {model.batchsize} | max_tokens = {model.max_char_length} | learning_rate = {model.lr}')
    print(f'epochs = {model.epochs} | warmup_size = {model.warmup_size} | dropout = {model.dropout}')
    print(f'num_labels = {model.num_labels} | dataset_length = {len(model.dataset_train.labels)} | dataset_name = {model.dataname}')
    print(f'random_seed = {model.seed_val}')
    print('-'*59)

    return None

def printEvalResults(model,output_batch_list_eval,label_batch_list_eval):

    print('-'*59)
    accu = metrics_config(model)['accuracy_accu'](output_batch_list_eval, label_batch_list_eval)
    print(f"EVAL Accuracy: {accu}")
    conf_matrix = metrics_config(model)['conf_matrix'](output_batch_list_eval,label_batch_list_eval)
    print(f'matriz de confusão: {conf_matrix}')
    precision=metrics_config(model)['precision'](output_batch_list_eval,label_batch_list_eval)
    print(f'precision: {precision}')
    f1score=metrics_config(model)['f1score'](output_batch_list_eval,label_batch_list_eval)
    print(f'f1score: {f1score}')
    print('-'*59)

    return None