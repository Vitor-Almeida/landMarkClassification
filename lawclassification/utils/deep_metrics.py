import torchmetrics
from torch import zeros, long

def execute_metrics_type(metricsDic,pred,label):

    metrics_result = {}
    tmp_result = {}

    #arrumar:
    indexes = zeros(label.size(),dtype=long).to('cuda') #zz aq

    for views in metricsDic:
        for metrics in metricsDic[views]:

            if metrics == 'rP': #information retrival metrics requires indexes
                tmp_result.update({metrics : round(metricsDic[views][metrics](pred, label, indexes).item(),4)})
            else:
                tmp_result.update({metrics : round(metricsDic[views][metrics](pred, label).item(),4)})
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
            metrics_result.update({metrics:round(result.item(),4)})

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

            return {'accuracy':accuracy,
                    'f1score_micro':f1score_micro,
                    'f1score_macro':f1score_macro,
                    'recall':recall,
                    'precision':precision,
                    'auroc':auroc}

        else: 

            #colocar a nDCG@x (x?)

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
            rP = torchmetrics.RetrievalNormalizedDCG()

            #nao bate:
            auroc = torchmetrics.AUROC(num_classes=num_labels,
                                       average='micro').to(device)

            return {'accuracy':accuracy,
                    'f1score_micro':f1score_micro,
                    'f1score_macro': f1score_macro,
                    'precision':precision,
                    'recall': recall,
                    'rP':rP,
                    'auroc':auroc}


class deep_metrics():
    """
    Persistent metrics of pytorchmetrics
    """
    def __init__(self,model):

        super(deep_metrics, self).__init__()

        dataset_type = ['Train','Test','Val']

        all_metrics_multi = ['accuracy','f1score_micro','f1score_macro','precision','recall','rP','auroc']
        all_metrics_single = ['accuracy','f1score_micro','f1score_macro','recall','precision','auroc']

        all_metrics = all_metrics_multi if model.problem_type == 'multi_label_classification' else all_metrics_single

        all_views = ['Batch','Epoch']
        dic_datasets = {}
        dic_metrics = {}
        dic_views = {}

        for dataset in dataset_type:
            for view in all_views:
                for metrics in all_metrics:
                    dic_metrics.update({metrics:metrics_config(num_labels = model.num_labels,
                                                               device = model.device,
                                                               problem_type = model.problem_type)[metrics]}
                    )
                dic_views.update({view:dic_metrics})
                dic_metrics = {}
            dic_datasets.update({dataset:dic_views})
            dic_views = {}

        self.metricDic = dic_datasets