import torchmetrics
from torch import zeros, long

def execute_metrics_type(metricsDic,pred,label):

    metrics_result = {}
    tmp_result = {}

    #arrumar:
    indexes = zeros(label.size(),dtype=long).to('cuda') #zz aq

    for views in metricsDic:
        for metrics in metricsDic[views]:

            if metrics == 'nDCG': #information retrival metrics requires indexes
                #tmp_result.update({metrics : round(metricsDic[views][metrics](pred, label, indexes).item(),4)})
                tmp_result.update({metrics : metricsDic[views][metrics](pred, label, indexes)})
            else:
                #tmp_result.update({metrics : round(metricsDic[views][metrics](pred, label).item(),4)})
                tmp_result.update({metrics : metricsDic[views][metrics](pred, label)})
        metrics_result.update({views:tmp_result})
        tmp_result = {}

    return metrics_result

def reset_all(metricsDic):

    #fix:
    for views in metricsDic:
        for metrics in metricsDic[views]:
            for m in metricsDic[views][metrics]:
                metricsDic[views][metrics][m].reset()

    return None

def compute_metrics_type(metricsViews,action):

    metrics_result = {}

    if action == 'reset':
        for metrics in metricsViews:
            metricsViews[metrics].reset()
            metrics_result.update({metrics:0.0})

        return metrics_result
    else:
        for metrics in metricsViews:
            result = metricsViews[metrics].compute()
            metrics_result.update({metrics:result})
            #metrics_result.update({metrics:round(result.item(),4)})

        return metrics_result

def metrics_config(num_labels,device,problem_type):

        if problem_type == 'single_label_classification':

            #num_labels aqui vai dar ruim se testar no dataset test e tiver menos label q o dataset train (?)

            accuracy = torchmetrics.Accuracy(num_classes=num_labels, 
                                             average='micro', 
                                             threshold = 0.5, 
                                             subset_accuracy = False).to(device)

            f1score_macro = torchmetrics.F1Score(num_classes=num_labels, 
                                           average='macro', 
                                           threshold = 0.5, 
                                           subset_accuracy = False).to(device)

            f1score_micro = torchmetrics.F1Score(num_classes=num_labels, 
                                           average='micro', 
                                           threshold = 0.5, 
                                           subset_accuracy = False).to(device)

            precision = torchmetrics.Precision(num_classes=num_labels, 
                                               average='micro', 
                                               threshold = 0.5, 
                                               subset_accuracy = False).to(device)

            recall = torchmetrics.Recall(num_classes=num_labels, 
                                               average='micro', 
                                               threshold = 0.5, 
                                               subset_accuracy = False).to(device)

            auroc = torchmetrics.AUROC(num_classes=num_labels, 
                                       average='macro',
                                       subset_accuracy = False).to(device)

            return torchmetrics.MetricCollection({'accuracy':accuracy,
                                                  'f1score_micro':f1score_micro,
                                                  'f1score_macro':f1score_macro,
                                                  'recall':recall,
                                                  'precision':precision,
                                                  'auroc':auroc})

        else: 

            accuracy = torchmetrics.Accuracy(average='micro', 
                                             threshold = 0.5, 
                                             subset_accuracy = True).to(device)

            f1score_micro = torchmetrics.F1Score(average='micro', 
                                           threshold = 0.5).to(device)

            #testar:
            f1score_macro = torchmetrics.F1Score(num_classes=num_labels, 
                                                 average='macro', 
                                                 threshold = 0.5).to(device)

            #testar:
            precision = torchmetrics.Precision(average='micro', 
                                               threshold = 0.5).to(device)

            #testar:
            recall = torchmetrics.Recall(average='micro', 
                                         threshold = 0.5).to(device)                          

            #testar:
            nDCG = torchmetrics.RetrievalNormalizedDCG()

            #nao bate:
            auroc_micro = torchmetrics.AUROC(num_classes=num_labels,
                                       average='micro').to(device)

            #testar
            auroc_macro = torchmetrics.AUROC(num_classes=num_labels,
                                       average='macro').to(device)


            return torchmetrics.MetricCollection({'accuracy':accuracy,
                                                  'f1score_micro':f1score_micro,
                                                  'f1score_macro': f1score_macro,
                                                  'precision':precision,
                                                  'recall': recall,
                                                  #'nDCG':nDCG,
                                                  'auroc_micro':auroc_micro,
                                                  'auroc_macro':auroc_macro})