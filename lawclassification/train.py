from tqdm.auto import tqdm
import torch
from dataset.dataset_load import yelpReview
from models.deep_models import deep_models
from torchmetrics import Accuracy

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

def trainLoop(cur_epoch,model,dataloader,progress_bar):

    model.model.train()

    accuracy_accu = Accuracy(num_classes=model.num_labels, average='micro', threshold = 0.5, mdmc_average = 'global').to(model.device)

    for idx,batch in enumerate(dataloader):
        model.optimizer.zero_grad()
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model.model(**batch)
        loss = outputs.loss # como é a camada classifier desse modelo, qual valor definido de 'treshold'?
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), 0.1) # exploding gradients ?
        model.optimizer.step()
        model.scheduler.step() #diferença aqui entre o tutorial do pytorch ?

        ### metricas:
        if idx == 0:
            output_batch_list = outputs.logits.unsqueeze(2)
            label_batch_list = batch['labels'].unsqueeze(1)
        else:
            output_batch_list = torch.cat((output_batch_list,outputs.logits.unsqueeze(2)),dim=2)
            label_batch_list = torch.cat((label_batch_list,batch['labels'].unsqueeze(1)),dim=1)

        accuracy_accu_result = round(accuracy_accu(output_batch_list, label_batch_list).item(),2)

        ###
        
        printLog(idx,cur_epoch,model,accuracy_accu_result)

        progress_bar.update(1)

    return output_batch_list,label_batch_list

def testLoop(cur_epoch,model,dataloader):

    model.model.eval()

    accuracy_accu = Accuracy(num_classes=model.num_labels, average='micro', threshold = 0.5, mdmc_average = 'global').to(model.device)

    with torch.no_grad():
        for idx,batch in enumerate(dataloader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model.model(**batch)

            ### metricas:

            if idx == 0:
                output_batch_list = outputs.logits.unsqueeze(2)
                label_batch_list = batch['labels'].unsqueeze(1)
            else:
                output_batch_list = torch.cat((output_batch_list,outputs.logits.unsqueeze(2)),dim=2)
                label_batch_list = torch.cat((label_batch_list,batch['labels'].unsqueeze(1)),dim=1)

            accuracy_accu_result = round(accuracy_accu(output_batch_list, label_batch_list).item(),2)

            ###

        printLog(-1,cur_epoch,model,accuracy_accu_result)

    return output_batch_list,label_batch_list

def printParamsTerminal(model):

    print('-'*22,'Parameters:','-'*22)
    print(f'modelo: {model.model_name} | batchsize: {model.batchsize} | max_tokens = {model.max_char_length} | learning_rate = {model.lr}')
    print(f'epochs = {model.epochs} | warmup_size = {model.warmup_size} | dropout = {model.dropout}')
    print(f'num_labels = {model.num_labels} | dataset_length = {len(model.dataset_train.labels)} | dataset_name = {model.dataset_train.name}')
    print(f'random_seed = {model.seed_val}')
    print('-'*59)

    return None

def metrics_config(model):

    mDict = {}

    #
    accuracy_accu = Accuracy(num_classes=model.num_labels, average='micro', threshold = 0.5, mdmc_average = 'global').to(model.device)


    mDict = {'accuracy_accu':accuracy_accu}

    return mDict


def main():

    #baixar modelo e deixar offline:
    #git clone https://huggingface.co/bert-base-uncased

    #adicionar matriz confusao
    #adicionar f1
    #adicionar sumario
    #adicionar leitor de parametros
    #fazer AUC
    #adicionar mlflow
    #adicionar gridsearch
    #fazer um exportador pra saber onde ta errando

    model_name = 'bert-base-uncased'
    batchsize = 16
    max_char_length = 128
    lr = 5e-5
    epochs = 5
    warmup_size = 0.1
    dropout = 0.2
    
    model = deep_models(model_name, batchsize, max_char_length, lr, epochs, warmup_size, yelpReview, dropout)
    progress_bar = tqdm(range(model.total_steps))

    metrics_config(model)
    printParamsTerminal(model)

    for epoch_i in range(0, model.epochs):

        output_batch_train,label_batch_train = trainLoop(epoch_i,model,model.train_dataloader,progress_bar)

        output_batch_test,label_batch_test = testLoop(epoch_i,model,model.test_dataloader)

        ## get all metric data:

        if epoch_i == 0:
            output_batch_list_train = output_batch_train
            label_batch_list_train = label_batch_train

            output_batch_list_test = output_batch_test
            label_batch_list_test = label_batch_test
        else:
            output_batch_list_train = torch.cat((output_batch_list_train,output_batch_train),dim=2)
            label_batch_list_train = torch.cat((label_batch_list_train,label_batch_train),dim=1)

            output_batch_list_test = torch.cat((output_batch_list_test,output_batch_test),dim=2)
            label_batch_list_test = torch.cat((label_batch_list_test,label_batch_test),dim=1)


    print('-'*22,'EVAL Accuracy:','-'*22)
    testLoop(epoch_i,model,model.val_dataloader)

    return None

if __name__ == '__main__':
    main()