from tqdm.auto import tqdm
import torch
from dataset.dataset_load import yelpReview
from models.deep_models import deep_models
from torchmetrics import Accuracy,ConfusionMatrix,Precision,F1Score
from utils.helperfuns import printLog, printParamsTerminal

def trainLoop(cur_epoch,model,dataloader,progress_bar):

    model.model.train()

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

        accuracy_accu_result = round(metrics_config(model)['accuracy_accu'](output_batch_list, label_batch_list).item(),2)

        ###
        
        printLog(idx,cur_epoch,model,accuracy_accu_result)

        progress_bar.update(1)

    return output_batch_list,label_batch_list

def testLoop(model,dataloader):

    model.model.eval()

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

            ###

    return output_batch_list,label_batch_list

def metrics_config(model):

    accuracy_accu = Accuracy(num_classes=model.num_labels, average='micro', threshold = 0.5, mdmc_average = 'global',top_k=1).to(model.device)
    conf_matrix = ConfusionMatrix(num_classes=model.num_labels,normalize = None, threshold  = 0.5).to(model.device)
    precision = Precision(num_classes=model.num_labels, average='micro', threshold  = 0.5, mdmc_average = 'global',top_k=1).to(model.device)
    f1score = F1Score(num_classes=model.num_labels, average='micro', threshold  = 0.5, mdmc_average = 'global',top_k=1).to(model.device)

    mDict = {'accuracy_accu':accuracy_accu,'conf_matrix':conf_matrix,'precision':precision,'f1score':f1score}

    return mDict

def main():

    model_name = 'bert-base-uncased'
    batchsize = 16
    max_char_length = 128
    lr = 5e-5
    epochs = 3
    warmup_size = 0.1
    dropout = 0.2
    
    model = deep_models(model_name, batchsize, max_char_length, lr, epochs, warmup_size, yelpReview, dropout)
    progress_bar = tqdm(range(model.total_steps))

    printParamsTerminal(model)

    for epoch_i in range(0, model.epochs):

        output_batch_train,label_batch_train = trainLoop(epoch_i,model,model.train_dataloader,progress_bar)

        output_batch_test,label_batch_test = testLoop(model,model.test_dataloader)
        acc_test = round(metrics_config(model)['accuracy_accu'](output_batch_test, label_batch_test).item(),2)
        printLog(-1,epoch_i,model,acc_test)

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

        ##

    print('-'*59)
    output_batch_list_eval,label_batch_list_eval=testLoop(model,model.val_dataloader)
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

if __name__ == '__main__':
    main()