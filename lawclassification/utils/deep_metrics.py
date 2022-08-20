import torchmetrics

def execute_metrics_type(metricsDic,pred,label):

    metrics_result = {}
    tmp_result = {}

    for views in metricsDic:
        for metrics in metricsDic[views]:
            tmp_result.update({metrics : round(metricsDic[views][metrics](pred, label).item(),4)})
        metrics_result.update({views:tmp_result})
        tmp_result = {}

    return metrics_result

def compute_metrics_type(metricsViews,action):

    if action == 'reset':
        for metrics in metricsViews:
            metrics.reset()
    else:
        for metrics in metricsViews:
            metrics.compute()

    return None

def metrics_config(num_labels,device,problem_type):

        if problem_type == 'single_label_classification':

            accuracy = torchmetrics.Accuracy(num_classes=num_labels, 
                                             average='macro', 
                                             threshold = 0.5, 
                                             mdmc_average = 'global',
                                             multiclass = True, 
                                             subset_accuracy = False).to(device)

            f1score = torchmetrics.F1Score(num_classes=num_labels, 
                                           average='macro', 
                                           threshold = 0.5, 
                                           mdmc_average = 'global',
                                           multiclass = True, 
                                           subset_accuracy = False).to(device)

            precision = torchmetrics.Precision(num_classes=num_labels, 
                                           average='macro', 
                                           threshold = 0.5, 
                                           mdmc_average = 'global',
                                           multiclass = True, 
                                           subset_accuracy = False).to(device)

            auroc = torchmetrics.AUROC(num_classes=num_labels, 
                                           average='macro',
                                           multiclass = True, 
                                           subset_accuracy = False).to(device)

        else: 

            accuracy = torchmetrics.Accuracy(num_classes=num_labels, 
                                             average='micro', 
                                             threshold = 0.5, 
                                             mdmc_average = 'samplewise',
                                             multiclass = True, 
                                             subset_accuracy = True).to(device)

            f1score = torchmetrics.F1Score(num_classes=num_labels, 
                                           average='micro', 
                                           threshold = 0.5, 
                                           mdmc_average = 'global',
                                           multiclass = True, 
                                           subset_accuracy = True).to(device)

            precision = torchmetrics.Precision(num_classes=num_labels, 
                                           average='micro', 
                                           threshold = 0.5, 
                                           mdmc_average = 'samplewise',
                                           multiclass = True, 
                                           subset_accuracy = True).to(device)

            auroc = torchmetrics.AUROC(num_classes=num_labels, 
                                           average='micro',
                                           multiclass = True, 
                                           subset_accuracy = True).to(device)

        return {'accuracy':accuracy,
                'f1score':f1score,
                'precision':precision,
                'auroc':auroc}

class deep_metrics():
    """
    Persistent metrics of pytorchmetrics
    """
    def __init__(self,model):

        super(deep_metrics, self).__init__()

        dataset_type = ['Train','Test','Val']
        all_metrics = ['accuracy','f1score','precision','auroc']
        all_views = ['Batch','Epoch']
        dic_datasets = {}
        dic_metrics = {}
        dic_views = {}

        for dataset in dataset_type:
            for view in all_views:
                for metrics in all_metrics:
                    dic_metrics.update({metrics:metrics_config(model.num_labels,
                                                          model.device,
                                                          model.problem_type)[metrics]}
                    )
                dic_views.update({view:dic_metrics})
                dic_metrics = {}
            dic_datasets.update({dataset:dic_views})
            dic_views = {}

        self.metricDic = dic_datasets