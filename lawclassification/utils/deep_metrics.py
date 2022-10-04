import torchmetrics
import torch

def metrics_config(num_labels,device,problem_type):

    if problem_type == 'single_label_classification':

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

        #auroc = torchmetrics.AUROC(num_classes=num_labels, 
        #                           average='macro',
        #                           subset_accuracy = False).to(device)

        return torchmetrics.MetricCollection({'accuracy':accuracy,
                                                'f1score_micro':f1score_micro,
                                                'f1score_macro':f1score_macro,
                                                'recall':recall,
                                                'precision':precision
                                                #'auroc':auroc
                                                })

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
        #auroc_micro = torchmetrics.AUROC(num_classes=num_labels,
        #                                 average='micro').to(device)

        #testar
        #auroc_macro = torchmetrics.AUROC(num_classes=num_labels,
        #                                 average='macro').to(device)


        return torchmetrics.MetricCollection({'accuracy':accuracy,
                                                'f1score_micro':f1score_micro,
                                                'f1score_macro': f1score_macro,
                                                'precision':precision,
                                                'recall': recall
                                                #'nDCG':nDCG,
                                                #'auroc_micro':auroc_micro,
                                                #'auroc_macro':auroc_macro
                                                })

def metrics_config_special(num_labels,device):

    f1score_macro = torchmetrics.F1Score(num_classes=num_labels, 
                                            average='macro', 
                                            threshold = 0.5, 
                                            subset_accuracy = False).to(device)

    f1score_micro = torchmetrics.F1Score(num_classes=num_labels, 
                                            average='micro', 
                                            threshold = 0.5, 
                                            subset_accuracy = False).to(device)

    return torchmetrics.MetricCollection({'f1score_micro_special':f1score_micro,
                                            'f1score_macro_special':f1score_macro
                                            #'auroc':auroc
                                            })

def f1ajust_lexglue(outputs, batch, device, flagBertGCN):

    if flagBertGCN:
        ajTensor = (outputs > torch.tensor([0.5],device=device,dtype=torch.float32))*1.0
    else:
        ajTensor = (torch.sigmoid(outputs) > torch.tensor([0.5],device=device,dtype=torch.float32))*1.0
    nOut = torch.empty((1,ajTensor.size()[1]+1),dtype=torch.float32,device=device)
    nLab = torch.empty((1,batch.size()[1]+1),dtype=torch.int32,device=device)
    for out,lab in zip(ajTensor,batch):
        if out.sum() > 0:
            out=torch.cat((torch.tensor([0.0],device=device,dtype=torch.float32),out),dim=0)
        else:
            out=torch.cat((torch.tensor([1.0],device=device,dtype=torch.float32),out),dim=0)

        if lab.sum() > 0:
            lab = torch.cat((torch.tensor([0],device=device,dtype=torch.int32),lab),dim=0)
        else:
            lab = torch.cat((torch.tensor([1],device=device,dtype=torch.int32),lab),dim=0)

        nOut = torch.cat((nOut,out.unsqueeze(dim=0)))
        nLab = torch.cat((nLab,lab.unsqueeze(dim=0)))

    return nOut[1:,:],nLab[1:,:]