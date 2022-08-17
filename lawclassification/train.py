from tqdm.auto import tqdm
from torch import no_grad, argmax, nn, mean, Tensor, cat
import os
from dataset.dataset_load import yelpReview
from models.models import models
from utils.definitions import ROOT_DIR
from torchmetrics import Accuracy

def main():

    #git clone https://huggingface.co/bert-base-uncased
    
    #arrumar dropout
    #dividir eval e test
    #deixar os print as https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
    #adicionar mlflow
    #adicionar gridsearch
    #adicionar matriz confusao
    #adicionar f1
    #adicionar sumario
    #adicionar leitor de parametros

    model_name='bert-base-uncased'
    batchsize=16
    max_char_length = 512
    lr = 5e-5
    epochs = 5
    warmup_size = 0.1
    dropout = 0.1 #nao faz nada, vem do parâmetro do BERT, entrar na camada model.dropout.parameters[xx] = 0.1 etc

    model_path = os.path.join(ROOT_DIR,'lawclassification','models','external',model_name)

    train = models(model_path, batchsize, max_char_length, lr, epochs, warmup_size, yelpReview)

    progress_bar = tqdm(range(train.total_steps))

    for epoch_i in range(0, train.epochs):
        train.model.train()

        for batch in train.train_dataloader:
            
            batch = {k: v.to(train.device) for k, v in batch.items()}

            #train.model.zero_grad() #nao tem no tutorial do transofmers

            outputs = train.model(**batch)
            loss = outputs.loss
            loss.backward()
            nn.utils.clip_grad_norm_(train.model.parameters(), 1.0) # exploding gradients
            train.optimizer.step()
            train.scheduler.step()
            train.optimizer.zero_grad() #?
            
            progress_bar.update(1)

        train.model.eval()

        #dividir entre eval e test:
        #colocar as observações vindo do site do pytorch de class https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
        #adicionar o mlflow
        #adicionar o gridsearch msm q nao vá usar agora

        acc_list = Tensor([0]).to(train.device)
        for batch in train.test_dataloader:
            batch = {k: v.to(train.device) for k, v in batch.items()}
            with no_grad():
                outputs = train.model(**batch)

            metric = Accuracy(num_classes = train.num_labels, threshold = 0.5, average='micro')
            metric.to(train.device)
            predictions = argmax(outputs.logits, dim=-1)
            result_metric = metric(predictions,batch["labels"]).unsqueeze(0)
            acc_list = cat((acc_list,result_metric))
            #print('predicao: ', predictions)
            #print('label: ', batch["labels"])
        #print(acc_list[1:])
        print(f'accuracia da {epoch_i}: {mean(acc_list[1:])}')
        
    return None


if __name__ == '__main__':
    main()