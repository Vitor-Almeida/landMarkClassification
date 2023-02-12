from utils.definitions import ROOT_DIR
import os
import pandas as pd
from datasets import Dataset

from setfit import SetFitModel
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitTrainer

def load_prepare_data(datasetName):
    dftest = pd.read_csv(os.path.join(ROOT_DIR,'data',datasetName,'interm','test','test.csv'))
    dftrain = pd.read_csv(os.path.join(ROOT_DIR,'data',datasetName,'interm','train','train.csv'))
    dfval = pd.read_csv(os.path.join(ROOT_DIR,'data',datasetName,'interm','val','val.csv'))

    dftest = dftest[['text','labels']]
    dftrain = dftrain[['text','labels']]
    dfval = dfval[['text','labels']]

    #truncando os textos em 512 tokens ~5000 chars
    dftest['text'] = dftest['text'].str.slice(0,10*512)
    dftrain['text'] = dftrain['text'].str.slice(0,10*512)
    dfval['text'] = dfval['text'].str.slice(0,10*512)

    dftest['labels'] = dftest['labels'].apply(lambda row: eval(row))
    dftrain['labels'] = dftrain['labels'].apply(lambda row: eval(row))
    dfval['labels'] = dfval['labels'].apply(lambda row: eval(row))

    labelExample=dftest['labels'][0]
    labelLen = len(labelExample)

    testLabels = dftest['labels'].to_list()
    trainLabels = dftrain['labels'].to_list()
    valLabels = dfval['labels'].to_list()

    featuresCols = []
    for idx,n in enumerate(labelExample):
        featuresCols.append('col_'+str(idx))  #<---------- pegar os nomes corretos aqui

    dfLTest = pd.DataFrame(data=testLabels, columns=featuresCols)
    dfLTrain = pd.DataFrame(data=trainLabels, columns=featuresCols)
    dfLVal = pd.DataFrame(data=valLabels, columns=featuresCols)

    dftest = pd.concat([dftest, dfLTest.astype('int64')], axis = 1)
    dftrain = pd.concat([dftrain, dfLTrain.astype('int64')], axis = 1)
    dfval = pd.concat([dfval, dfLVal.astype('int64')], axis = 1)

    Hugtest=Dataset.from_pandas(dftest)
    Hugtrain=Dataset.from_pandas(dftrain)
    Hugval=Dataset.from_pandas(dfval)

    dicHugData = {'test':Hugtest,'train':Hugtrain,'val':Hugval}

    #    with open(os.path.join(ROOT_DIR,'data','big_tj','interm','id2label.json'),'w') as f:
    #        json.dump(id2label,f)
    #        f.close()
    #    with open(os.path.join(ROOT_DIR,'data','big_tj','interm','label2id.json'),'w') as f:
    #        json.dump(label2id,f)
    #        f.close()

    return dicHugData

def trainModel(modelName, dataDic):

    modelPath = os.path.join(ROOT_DIR,'lawclassification','models','external',modelName)
    model = SetFitModel.from_pretrained(modelPath, multi_target_strategy="one-vs-rest")

    #setence size: model.model_body.max_seq_length

    trainer = SetFitTrainer(
        model=model,
        train_dataset=dataDic['train'],
        use_amp = True,
        eval_dataset=dataDic['test'],
        loss_class=CosineSimilarityLoss,
        num_iterations=20,
        column_mapping={"text": "text", "labels": "label"},
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)


    #fazer teste na validação:
    #preds = model(
    #[
    #    "Jewish people often don't eat pork.",
    #    "Is this lipstick suitable for people with dark skin?"
    #]
    #)
    #preds

    return None


def main():
    dataDic = load_prepare_data(datasetName="big_tj")
    trainModel(modelName = "rufimelo.Legal-BERTimbau-sts-base-ma-v2", dataDic = dataDic)

if __name__ == '__main__':
    main()