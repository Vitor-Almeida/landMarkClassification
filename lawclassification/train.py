from tqdm.auto import tqdm
import torch
from dataset.dataset_load import deepData
from models.deep_models import deep_models
from utils.helper_funs import printLog, printParamsTerminal,metrics_config,printEvalResults
import mlflow

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
            outputBatch = outputs.logits.unsqueeze(2)
            labelBatch = batch['labels'].unsqueeze(1)
        else:
            outputBatch = torch.cat((outputBatch,outputs.logits.unsqueeze(2)),dim=2)
            labelBatch = torch.cat((labelBatch,batch['labels'].unsqueeze(1)),dim=1)

        acc = round(metrics_config(model)['accuracy_accu'](outputBatch, labelBatch).item(),2)

        ###
        
        printLog(idx,cur_epoch,model,acc)

        progress_bar.update(1)

    return outputBatch,labelBatch

def testLoop(model,dataloader):

    model.model.eval()

    with torch.no_grad():
        for idx,batch in enumerate(dataloader):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model.model(**batch)

            ### metricas:

            if idx == 0:
                outputBatch = outputs.logits.unsqueeze(2)
                labelBatch = batch['labels'].unsqueeze(1)
            else:
                outputBatch = torch.cat((outputBatch,outputs.logits.unsqueeze(2)),dim=2)
                labelBatch = torch.cat((labelBatch,batch['labels'].unsqueeze(1)),dim=1)

            ###

    return outputBatch,labelBatch

def main():

    model_name = 'bert-base-uncased'
    batchsize = 32
    max_char_length = 512
    lr = 3e-5
    epochs = 6
    warmup_size = 0.1
    dropout = 0.1
    dataname = 'dmoz_510_1500'
    
    model = deep_models(model_name, batchsize, max_char_length, lr, epochs, warmup_size, deepData, dropout, dataname=dataname)
    progress_bar = tqdm(range(model.total_steps))

    printParamsTerminal(model)

    for epoch_i in range(0, model.epochs):

        outBatTrain,labBatTrain = trainLoop(epoch_i,model,model.train_dataloader,progress_bar)

        outBatTest,labBatTest = testLoop(model,model.test_dataloader)

        acc_test = round(metrics_config(model)['accuracy_accu'](outBatTest, labBatTest).item(),2)
        printLog(-1,epoch_i,model,acc_test)

        ## get all metric data:

        if epoch_i == 0:
            outBatTrain_List = outBatTrain
            labBatTrain_List = labBatTrain

            outBatTest_List = outBatTest
            labBatTest_List = labBatTest
        else:
            outBatTrain_List = torch.cat((outBatTrain_List,outBatTrain),dim=2)
            labBatTrain_List = torch.cat((labBatTrain_List,labBatTrain),dim=1)

            outBatTest_List = torch.cat((outBatTest_List,outBatTest),dim=2)
            labBatTest_List = torch.cat((labBatTest_List,labBatTest),dim=1)

        ##

    outBatEval,labBatEval=testLoop(model,model.val_dataloader)
    printEvalResults(model,outBatEval,labBatEval)

    return None

if __name__ == '__main__':
    main()