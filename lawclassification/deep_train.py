from tqdm.auto import tqdm
import torch
from models.deep_models import deep_models
from utils.helper_funs import printParamsTerminal,print_step_log
from utils.deep_metrics import deep_metrics, execute_metrics_type, compute_metrics_type
import mlflow

def trainLoop(epoch_i,model,progressBar,metrics):

    model.model.train()

    for idx,batch in enumerate(model.train_dataloader):
        model.optimizer.zero_grad()
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model.model(**batch)
        loss = outputs.loss 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
        model.optimizer.step()
        model.scheduler.step() #diferença aqui entre o tutorial do pytorch ?

        metricsResults = execute_metrics_type(metrics.metricDic['Train'],outputs.logits, batch['labels'].int())

        print_step_log(idx,epoch_i,model,metricsResults['Batch']) #fica dando o resultado acumulado durante a epoca

        progressBar.update(1)

    compute_metrics_type(metrics.metricDic['Train']['Batch'],action='compute')
    compute_metrics_type(metrics.metricDic['Train']['Batch'],action='reset')

    return None

def testLoop(model,dataloader,metrics):

    model.model.eval()

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model.model(**batch)

            ### metricas:

            accBatch = metrics.accuracyTestBatch(outputs.logits, batch['labels'].int())
            accEpoch = metrics.accuracyTestEpoch(outputs.logits, batch['labels'].int())
            f1Batch = metrics.f1scoreTestBatch(outputs.logits, batch['labels'].int())
            f1Epoch = metrics.f1scoreTestEpoch(outputs.logits, batch['labels'].int())
            precisionBatch = metrics.precisionTestBatch(outputs.logits, batch['labels'].int())
            precisionEpoch = metrics.precisionTestEpoch(outputs.logits, batch['labels'].int())
            aurocBatch = metrics.aurocTestBatch(outputs.logits, batch['labels'].int())
            aurocEpoch = metrics.aurocTestEpoch(outputs.logits, batch['labels'].int())

    accBatch_Result = metrics.accuracyTestBatch.compute()
    f1Batch_Result = metrics.f1scoreTestBatch.compute()
    precisionBatch_Result = metrics.precisionTestBatch.compute()
    aurocBatch_Result = metrics.aurocTestBatch.compute()

    print('acuracia do batch test: ', round(accBatch_Result.item(),4))
    print('f1 do batch test: ', round(f1Batch_Result.item(),4))
    print('precision do batch test: ', round(precisionBatch_Result.item(),4))
    print('auroc do batch test: ', round(aurocBatch_Result.item(),4))

    metrics.accuracyTestBatch.reset()
    metrics.f1scoreTestBatch.reset()
    metrics.precisionTestBatch.reset()
    metrics.aurocTestBatch.reset()

    return None

def valLoop(model,dataloader,metrics):

    model.model.eval()

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model.model(**batch)

            ### metricas:

            accBatch = metrics.accuracyValBatch(outputs.logits, batch['labels'].int())
            accEpoch = metrics.accuracyValEpoch(outputs.logits, batch['labels'].int())
            f1Batch = metrics.f1scoreValBatch(outputs.logits, batch['labels'].int())
            f1Epoch = metrics.f1scoreValEpoch(outputs.logits, batch['labels'].int())
            precisionBatch = metrics.precisionValBatch(outputs.logits, batch['labels'].int())
            precisionEpoch = metrics.precisionValEpoch(outputs.logits, batch['labels'].int())
            aurocBatch = metrics.aurocValBatch(outputs.logits, batch['labels'].int())
            aurocEpoch = metrics.aurocValEpoch(outputs.logits, batch['labels'].int())

    accBatch_Result = metrics.accuracyValBatch.compute()
    f1Batch_Result = metrics.f1scoreValBatch.compute()
    precisionBatch_Result = metrics.precisionValBatch.compute()
    aurocBatch_Result = metrics.aurocValBatch.compute()

    metrics.accuracyValBatch.reset()
    metrics.f1scoreValBatch.reset()
    metrics.precisionValBatch.reset()
    metrics.aurocValBatch.reset()

    return None

def main():

    model_name = 'bert-base-uncased'
    #model_name = 'albert-base-v2'
    #model_name = 'roberta-base'
    #model_name = 'nlpaueb.legal-bert-base-uncased'
    #model_name = 'distilbert-base-uncased'
    #model_name = 'saibo.legal-roberta-base' 
    #model_name = 'allenai.longformer-base-4096'
    
    batchsize = 16
    max_char_length = 256
    lr = 3e-5
    epochs = 6
    warmup_size = 0.1
    dropout = 0.1 #o dropout fica só no classificador? lugar do dropout fica mudando entre os modelos.
    #dataname = 'echr'
    #dataname = 'SemEval2018-Task1-all-data'
    dataname = 'dmoz_510_1500'
    #problem_type = 'multi_label_classification'
    problem_type = 'single_label_classification'
    
    model = deep_models(model_name, batchsize, max_char_length, lr, epochs, warmup_size, dropout, dataname,problem_type)
    metrics = deep_metrics(model)
    progress_bar = tqdm(range(model.total_steps))

    printParamsTerminal(model)

    for epoch_i in range(0, model.epochs):

        trainLoop(epoch_i,model,progress_bar,metrics)
        accEpoch_train_Result = metrics.accuracyTrainEpoch.compute()
        f1Epoch_train_Result = metrics.f1scoreTrainEpoch.compute()
        precisionEpoch_train_Result = metrics.precisionTrainEpoch.compute()
        aurocEpoch_train_Result = metrics.aurocTrainEpoch.compute()

        print('acuracia do train da epoca: ', round(accEpoch_train_Result.item(),4))
        print('f1 do train da epoca: ', round(f1Epoch_train_Result.item(),4))
        print('precision do train da epoca: ', round(precisionEpoch_train_Result.item(),4))
        print('auroc do train da epoca: ', round(aurocEpoch_train_Result.item(),4))

        metrics.accuracyTrainEpoch.reset()
        metrics.f1scoreTrainEpoch.reset()
        metrics.precisionTrainEpoch.reset()
        metrics.aurocTrainEpoch.reset()

        testLoop(model,model.test_dataloader,metrics)
        accEpoch_Test_Result = metrics.accuracyTestEpoch.compute()
        f1Epoch_Test_Result = metrics.f1scoreTestEpoch.compute()
        precisionEpoch_Test_Result = metrics.precisionTestEpoch.compute()
        aurocEpoch_Test_Result = metrics.aurocTestEpoch.compute()

        print('acuracia do Test da epoca: ', round(accEpoch_Test_Result.item(),4))
        print('f1 do Test da epoca: ', round(f1Epoch_Test_Result.item(),4))
        print('precision do Test da epoca: ', round(precisionEpoch_Test_Result.item(),4))
        print('auroc do Test da epoca: ', round(aurocEpoch_Test_Result.item(),4))

        metrics.accuracyTestEpoch.reset()
        metrics.f1scoreTestEpoch.reset()
        metrics.precisionTestEpoch.reset()
        metrics.aurocTestEpoch.reset()

    valLoop(model,model.val_dataloader,metrics)
    accEpoch_Val_Result = metrics.accuracyValEpoch.compute()
    f1Epoch_Val_Result = metrics.f1scoreValEpoch.compute()
    precisionEpoch_Val_Result = metrics.precisionValEpoch.compute()
    aurocEpoch_Val_Result = metrics.aurocValEpoch.compute()

    print('acuracia do Val da epoca: ', round(accEpoch_Val_Result.item(),4))
    print('f1 do Val da epoca: ', round(f1Epoch_Val_Result.item(),4))
    print('precision do Val da epoca: ', round(precisionEpoch_Val_Result.item(),4))
    print('auroc do Val da epoca: ', round(aurocEpoch_Val_Result.item(),4))

    metrics.accuracyValEpoch.reset()
    metrics.f1scoreValEpoch.reset()
    metrics.precisionValEpoch.reset()
    metrics.aurocValEpoch.reset()

    return None

if __name__ == '__main__':
    main()