import copy
import os
import torch
from torch_geometric.loader import NeighborLoader, ImbalancedSampler
from models.gcn_models import Text_GCN
from utils.definitions import ROOT_DIR
from utils.deep_metrics import metrics_config, metrics_config_special, f1ajust_lexglue
from utils.helper_funs import EarlyStopping
import mlflow
import gc
import pickle
from torch_geometric import utils as U
import random
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime

class gcn_train():

    def __init__(self,experiment):

        #torch.cuda.mem_get_info(self.device)

        super(gcn_train, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.seedVal = random.randint(0, 1000)
        random.seed(self.seedVal)
        np.random.seed(self.seedVal)
        torch.manual_seed(self.seedVal)
        torch.cuda.manual_seed_all(self.seedVal)

        self.earlyStopper = EarlyStopping(patience=5, min_delta=0)

        self.starttime = datetime.now()
        self.dataname = experiment['dataname']
    
        f = open(os.path.join(ROOT_DIR,'data',self.dataname,'pygraph.pickle'),'rb')
        self.dataset = pickle.load(f).to(self.device)
        f.close()

        self.epochs = int(experiment['epochs'])
        self.hiddenChannels = int(experiment['hidden_channels'])
        self.problemType = experiment['problem_type']
        self.batchSize = int(experiment['batchsize'])
        self.problemType = experiment['problem_type']
        self.lr = experiment['lr']
        self.neigh_param = eval(experiment['neigh_paramater'])

        self.model = Text_GCN(self.dataset.num_features, self.hiddenChannels, self.dataset.num_classes, self.device).to(self.device)

        self.optimizer = torch.optim.AdamW([
            dict(params=self.model.convs[0].parameters(), weight_decay=5e-4),
            dict(params=self.model.convs[1].parameters(), weight_decay=0)
        ], lr=self.lr)

        self.warmup_size = 0.1

        if self.problemType == 'single_label_classification':
            self.criterion = torch.nn.CrossEntropyLoss()
            self.logCrit = torch.nn.CrossEntropyLoss()
            #samplerTrain = ImbalancedSampler(self.dataset, input_nodes=self.dataset.train_mask)
            #samplerTest = ImbalancedSampler(self.dataset, input_nodes=self.dataset.test_mask)
            #samplerVal = ImbalancedSampler(self.dataset, input_nodes=self.dataset.val_mask)
        else:
            self.criterion = torch.nn.BCEWithLogitsLoss()
            self.logCrit = torch.nn.BCEWithLogitsLoss()
            #samplerTrain = ImbalancedSampler(self.dataset.homoLabels, input_nodes=self.dataset.train_mask)
            #samplerTest = ImbalancedSampler(self.dataset.homoLabels, input_nodes=self.dataset.test_mask)
            #samplerVal = ImbalancedSampler(self.dataset.homoLabels, input_nodes=self.dataset.val_mask)


        self.trainLoader = NeighborLoader(self.dataset, input_nodes=self.dataset.train_mask,
                                          num_neighbors = self.neigh_param, 
                                          shuffle=True,
                                          #is_sorted=True,
                                          batch_size = self.batchSize
                                          )

        self.testLoader = NeighborLoader(self.dataset, input_nodes=self.dataset.test_mask,
                                         num_neighbors = self.neigh_param, 
                                         shuffle=False, 
                                         #is_sorted=True,
                                         batch_size = self.batchSize
                                         )

        self.valLoader = NeighborLoader(self.dataset, input_nodes=self.dataset.val_mask,
                                        num_neighbors = self.neigh_param, 
                                        shuffle=False, 
                                        #is_sorted=True,
                                        batch_size = self.batchSize
                                        )

        #self.subgraphLoader = NeighborLoader(copy.copy(self.dataset), input_nodes=None,
        #                                     num_neighbors = [-1], shuffle=False,
        #                                     batch_size = self.batchSize
        #                                     #directed = False
        #                                     #is_sorted = True
        #                                     #num_workers= 4, persistent_workers = True
        #                                     )

        #self.total_steps = len(self.trainLoader) * self.epochs
        self.total_steps = self.epochs

        self.scheduler1 = torch.optim.lr_scheduler.LinearLR(self.optimizer, 
                                                            start_factor=0.1, 
                                                            end_factor=1, 
                                                            total_iters=int(self.warmup_size * self.total_steps))

        self.scheduler2 = torch.optim.lr_scheduler.LinearLR(self.optimizer, 
                                                            start_factor=1.0, 
                                                            end_factor=0.1,
                                                            total_iters=self.epochs-(int(self.warmup_size * self.total_steps))
                                                            )

        self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, 
                                                               schedulers = [self.scheduler1,self.scheduler2],
                                                               milestones= [int(self.warmup_size * self.total_steps)]
                                                               )

        #del self.subgraphLoader.data.x, self.subgraphLoader.data.y
        #gc.collect()
        #self.subgraphLoader.data.num_nodes = self.dataset.num_nodes
        #self.subgraphLoader.data.n_id = torch.arange(self.dataset.num_nodes)

        _ = metrics_config(num_labels = self.dataset.num_classes,
                           device = self.model.device,
                           problem_type = self.problemType)

        _special = metrics_config_special(num_labels = self.dataset.num_classes+1,
                                          device = self.model.device)

        self.metricsTrainBatch = _.clone(prefix='Train_Batch_')
        self.metricsTrainEpoch = _.clone(prefix='Train_Epoch_')
        self.metricsTestEpoch = _.clone(prefix='Test_Epoch_')
        self.metricsValEpoch = _.clone(prefix='Val_Epoch_')

        self.metricsTestEpochSpecial = _special.clone(prefix='Test_Special_')
        self.metricsValEpochSpecial = _special.clone(prefix='Val_Special_')

        self.datasetParams = {'num_nodes':self.dataset.x.size()[0],
                              'num_edges':self.dataset.edge_index.size()[1],
                              'random_seed':self.seedVal,
                              'num_classes':self.dataset.num_classes,
                              'num_node_docs': sum(self.dataset.test_mask).item()+sum(self.dataset.train_mask).item()+sum(self.dataset.val_mask).item(),
                              'test_length':sum(self.dataset.test_mask).item(),
                              'train_length':sum(self.dataset.train_mask).item(),
                              'lr':self.lr,
                              'neighboards': self.neigh_param,
                              'batch_size':self.batchSize,
                              'val_length':sum(self.dataset.val_mask).item(),
                              'num_node_words': self.dataset.x.size()[0] - (sum(self.dataset.test_mask).item()+sum(self.dataset.train_mask).item()+sum(self.dataset.val_mask).item()),
                              'avg_degree': round(torch.mean(U.degree(self.dataset.edge_index[0])).item(),4),
                              'homophily':self.dataset.homophily}

        mlflow.log_params(self.datasetParams)

    def train(self,epoch_i):
        self.model.train()

        lossTrain = total_examples = 0

        for batch in self.trainLoader:
            self.optimizer.zero_grad()

            y = batch.y[:batch.batch_size]
            y_hat = self.model(batch.x, batch.edge_index, batch.edge_weight)[:batch.batch_size]

            loss = self.criterion(y_hat, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            
            self.metricsTrainBatch(y_hat, y.int())

            lossTrain += loss.item() * batch.batch_size
            total_examples += batch.batch_size

        self.scheduler.step()

        metric = self.metricsTrainBatch.compute()
        mlflow.log_metrics({label:round(value.item(),4) for label, value in metric.items()},epoch_i+1)
        mlflow.log_metrics({'Train_Batch_loss':round(lossTrain/total_examples,4)},epoch_i+1)
        metric = self.metricsTrainBatch.reset()

        return None

    def test(self,epoch_i):

        self.model.eval()

        lossTest = total_examples = 0

        with torch.no_grad():
            for batch in self.testLoader:
                y = batch.y[:batch.batch_size]
                y_hat = self.model(batch.x, batch.edge_index, batch.edge_weight)[:batch.batch_size]

                loss = self.criterion(y_hat, y)

                self.metricsTestEpoch(y_hat, y.int())

                if self.dataname in ['ecthr_b_lexbench','ecthr_a_lexbench','unfair_lexbench']:
                    out,lab = f1ajust_lexglue(y_hat, y, self.model.device, False)
                    self.metricsTestEpochSpecial(out, lab.int())

                lossTest += loss.item() * batch.batch_size
                total_examples += batch.batch_size

        metric = self.metricsTestEpoch.compute()
        mlflow.log_metrics({label:round(value.item(),4) for label, value in metric.items()},epoch_i+1)
        mlflow.log_metrics({'Test_Epoch_loss':round(lossTest/total_examples,4)},epoch_i+1)
        metric = self.metricsTestEpoch.reset()

        if self.dataname in ['ecthr_b_lexbench','ecthr_a_lexbench','unfair_lexbench']:
            metric = self.metricsTestEpochSpecial.compute()
            mlflow.log_metrics({label:round(value.item(),4) for label, value in metric.items()},epoch_i+1)
            metric = self.metricsTestEpochSpecial.reset()

        return lossTest/total_examples

    def val(self,epoch_i):

        self.model.eval()

        lossVal = total_examples = 0

        with torch.no_grad():
            for batch in self.valLoader:
                y = batch.y[:batch.batch_size]
                y_hat = self.model(batch.x, batch.edge_index, batch.edge_weight)[:batch.batch_size]

                loss = self.criterion(y_hat, y)

                if self.dataname in ['ecthr_b_lexbench','ecthr_a_lexbench','unfair_lexbench']:
                    out,lab = f1ajust_lexglue(y_hat, y, self.model.device, False)
                    self.metricsValEpochSpecial(out, lab.int())

                self.metricsValEpoch(y_hat, y.int())

                lossVal += loss.item() * batch.batch_size
                total_examples += batch.batch_size

        metric = self.metricsValEpoch.compute()
        mlflow.log_metrics({label:round(value.item(),4) for label, value in metric.items()},epoch_i+1)
        mlflow.log_metrics({'Test_Epoch_loss':round(lossVal/total_examples,4)},epoch_i+1)
        metric = self.metricsValEpoch.reset()

        if self.dataname in ['ecthr_b_lexbench','ecthr_a_lexbench','unfair_lexbench']:
            metric = self.metricsValEpochSpecial.compute()
            mlflow.log_metrics({label:round(value.item(),4) for label, value in metric.items()},epoch_i+1)
            metric = self.metricsValEpochSpecial.reset()

        return None

    def fit_and_eval(self):

        for epoch_i in tqdm(range(self.epochs)):
            self.train(epoch_i)

            curTestLoss = self.test(epoch_i)

            if epoch_i > 3:
                
                earlyStopCriteria = curTestLoss

                if self.earlyStopper.early_stop(earlyStopCriteria):
                    self.val(epoch_i)
                    break

        mlflow.log_metrics({'Minute Duration':round((datetime.now() - self.starttime).total_seconds()/60,0)},self.epochs)