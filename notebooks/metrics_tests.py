import torch
from utils.deep_metrics import execute_metrics_type, compute_metrics_type
import torchmetrics

#single_label_classification
#multi_label_classification

#average type:
#micro = é só somar e dividir todos os acertos
#macro = é somar e dividir o acerto de cada classe e depois fazer a media burra (pesos iguais) entre eles, é bom caso exista inbalance nas classes.
#none = ele retorna o resultado abrindo por classe (vai precisar desse quando for fazer o summario)
#sample = ele faz para cada linha do batch e depois calcula a media burra com o batch


class model_test():

    def __init__(self,num_labels,problem_type,device="cpu"):
        self.num_labels = num_labels
        self.device = device
        self.problem_type = problem_type

        dataset_type = ['Train']
        all_metrics_multi = ['accuracy','auroc','precision','f1score','rP']
        all_metrics_single = ['accuracy','auroc','precision','f1score']
        all_views = ['Batch']
        dic_datasets = {}
        dic_metrics = {}
        dic_views = {}

        all_metrics = all_metrics_multi if problem_type == 'multi_label_classification' else all_metrics_single

        for dataset in dataset_type:
            for view in all_views:
                for metrics in all_metrics:
                    dic_metrics.update({metrics:metrics_config_test(num_labels = self.num_labels,
                                                            device = self.device,
                                                            problem_type = self.problem_type)[metrics]}
                    )
                dic_views.update({view:dic_metrics})
                dic_metrics = {}
            dic_datasets.update({dataset:dic_views})
            dic_views = {}

        self.metricDic = dic_datasets

def metrics_config_test(num_labels,device,problem_type):

    if problem_type == 'single_label_classification':

        accuracy = torchmetrics.Accuracy(num_classes=num_labels, 
                                            average='micro', 
                                            threshold = 0.5).to(device)

        precision = torchmetrics.Precision(num_classes=num_labels,
                                           average='micro', 
                                           threshold = 0.5).to(device)     

        f1score = torchmetrics.F1Score(num_classes=num_labels,
                                           average='micro', 
                                           threshold = 0.5).to(device) 

        auroc = torchmetrics.AUROC(num_classes=num_labels).to(device)

        #metricas precisam ta iguais nos dois.. 
        rP = torchmetrics.RetrievalPrecision()

        return {'accuracy':accuracy,'auroc':auroc,'precision':precision,'f1score':f1score}

    else:
        #colocar: recall
        #colcar o k dinamico

        #ok
        accuracy = torchmetrics.Accuracy(average='micro', 
                                        threshold = 0.5, 
                                        subset_accuracy = True).to(device)

        #eh pra ta ok
        precision = torchmetrics.Precision(average='micro', 
                                           threshold = 0.5).to(device)

        #ok
        f1score = torchmetrics.F1Score(average='micro', 
                                           threshold = 0.5).to(device)                                              

        #testar micro vs macro
        auroc = torchmetrics.AUROC(num_classes=num_labels,
                                   pos_label = 1,
                                   average='micro').to(device) #micro ou macro no auroc multi-label?

        rP = torchmetrics.RetrievalPrecision(empty_target_action='neg',k=1)

        return {'accuracy':accuracy,'auroc':auroc,'precision':precision,'f1score':f1score,'rP':rP}


    #return {'accuracy':accuracy,'auroc':auroc,'precision':precision,'f1score':f1score,'rP':rP}


def multi_test():

    #tipo de input do pytorch:
    #Multi-label

    ok = 0.6
    nok = 0.0

    outputs = torch.Tensor([[ok,nok,nok,ok],[ok,ok,nok,ok]])
    labels = torch.Tensor([[1,0,0,1],[1,1,0,0]]).int()

    modelDummy = model_test(labels.size()[1],"multi_label_classification",device="cpu")
    examinando = 'f1score'

    print('tetando o: ',examinando)
    metricsResults = execute_metrics_type(modelDummy.metricDic['Train'],outputs,labels.int())
    print('step1: ', metricsResults['Batch'][examinando])

    outputs = torch.Tensor([[ok,nok,nok,ok],[ok,ok,nok,ok]])
    labels = torch.Tensor([[1,0,0,1],[1,1,0,0]]).int()

    metricsResults = execute_metrics_type(modelDummy.metricDic['Train'],outputs,labels.int())
    print('step2: ', metricsResults['Batch'][examinando])

    outputs = torch.Tensor([[ok,nok,nok,ok],[ok,ok,nok,nok]])
    labels = torch.Tensor([[1,0,0,1],[1,1,0,0]]).int()

    metricsResults = execute_metrics_type(modelDummy.metricDic['Train'],outputs,labels.int())
    print('step3: ', metricsResults['Batch'][examinando])

    trainTmpResult = compute_metrics_type(modelDummy.metricDic['Train']['Batch'],action='compute')
    print('batch ', trainTmpResult[examinando])

def single_test():

    #tipo de input do pytorch:
    #Multi-class with logits or probabilities

    maior_1 = 0.5
    maior_2 = 0.3
    maior_3 = 0.15
    maior_4 = 0.05

    outputs = torch.Tensor([[maior_2,maior_1,maior_3,maior_4],[maior_3,maior_4,maior_1,maior_2]])
    labels = torch.Tensor([1,2]).int() #len aqui eh o tamanho do batch, e os numeros so podem ir de 0 ate total_labels-1

    modelDummy = model_test(outputs.size()[1],"single_label_classification",device="cpu")
    examinando = 'accuracy'

    metricsResults = execute_metrics_type(modelDummy.metricDic['Train'],outputs,labels.int())
    print('step1: ', metricsResults['Batch'][examinando])

    outputs = torch.Tensor([[maior_2,maior_1,maior_3,maior_4],[maior_4,maior_2,maior_3,maior_1]])
    labels = torch.Tensor([0,3]).int()

    metricsResults = execute_metrics_type(modelDummy.metricDic['Train'],outputs,labels.int())
    print('step2: ', metricsResults['Batch'][examinando])

    outputs = torch.Tensor([[maior_2,maior_1,maior_3,maior_4],[maior_2,maior_3,maior_1,maior_4]])
    labels = torch.Tensor([2,3]).int()

    metricsResults = execute_metrics_type(modelDummy.metricDic['Train'],outputs,labels.int())
    print('step3: ', metricsResults['Batch'][examinando])

    trainTmpResult = compute_metrics_type(modelDummy.metricDic['Train']['Batch'],action='compute')
    print('batch ', trainTmpResult['accuracy'])

def main():

    print('---multi test----:')
    multi_test()
    print('---single test----:')
    single_test()
    print('----end-----')

    return None

if __name__ == '__main__':
    main()
