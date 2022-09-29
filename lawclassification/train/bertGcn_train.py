import torch
from models.bertGcn_models import ensemble_model
from utils.deep_metrics import metrics_config
from utils.helper_funs import EarlyStopping
import mlflow
from torch_geometric import utils as U
from tqdm.auto import tqdm
from datetime import datetime

class bertGcn_Train():

    def __init__(self,experiment):
        super(bertGcn_Train, self).__init__()

        self.earlyStopper = EarlyStopping(patience=3, min_delta=0)

        self.starttime = datetime.now()

        self.model = ensemble_model(experiment)
    
        if self.model.problemType == 'single_label_classification':
            #self.criterion = torch.nn.CrossEntropyLoss()
            #self.logCrit = torch.nn.CrossEntropyLoss()
            self.criterion = torch.nn.NLLLoss()
            self.logCrit = torch.nn.NLLLoss()
        else:
            #self.criterion = torch.nn.BCEWithLogitsLoss()
            #self.logCrit = torch.nn.BCEWithLogitsLoss()
            self.criterion = torch.nn.BCELoss()
            self.logCrit = torch.nn.BCELoss()

        _ = metrics_config(num_labels = self.model.dataset.num_classes,
                           device = self.model.device,
                           problem_type = self.model.problemType)

        self.metricsTrainBatch = _.clone(prefix='Train_Batch_')
        self.metricsTrainEpoch = _.clone(prefix='Train_Epoch_')
        self.metricsTestEpoch = _.clone(prefix='Test_Epoch_')
        self.metricsValEpoch = _.clone(prefix='Val_Epoch_')

        self.datasetParams = {'num_nodes':self.model.dataset.x.size()[0],
                              'num_edges':self.model.dataset.edge_index.size()[1],
                              'random_seed':self.model.seedVal,
                              'num_classes':self.model.dataset.num_classes,
                              'num_node_docs': sum(self.model.dataset.test_mask).item()+sum(self.model.dataset.train_mask).item()+sum(self.model.dataset.val_mask).item(),
                              'test_length':sum(self.model.dataset.test_mask).item(),
                              'train_length':sum(self.model.dataset.train_mask).item(),
                              'bert_lr':self.model.bert_lr,
                              'gcn_lr':self.model.gcn_lr,
                              'neighboards': self.model.neigh_param,
                              'batch_size':self.model.batchSize,
                              'val_length':sum(self.model.dataset.val_mask).item(),
                              'num_node_words': self.model.dataset.x.size()[0] - (sum(self.model.dataset.test_mask).item()+sum(self.model.dataset.train_mask).item()+sum(self.model.dataset.val_mask).item()),
                              'avg_degree': round(torch.mean(U.degree(self.model.dataset.edge_index[0])).item(),4),
                              'homophily':self.model.dataset.homophily}

        mlflow.log_params(self.datasetParams)

    def update_doc_features(self):

        with torch.no_grad():
            #model = model.to(self.device)
            self.model.bertGcnModel.eval()
            cls_list = []
            for i, batch in enumerate(self.model.updateDataLoader):
                input_ids = batch[0].to(self.model.device)
                attention_mask = batch[1].to(self.model.device)

                with torch.autocast(device_type=self.model.device.type, dtype=torch.float16, enabled=True):
                    output = self.model.bertGcnModel.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]

                cls_list.append(output.cpu())
            cls_feat = torch.cat(cls_list, axis=0)
        self.model.dataset = self.model.dataset.to('cpu')
        self.model.dataset.x[self.model.dataset.docmask] = cls_feat
        #self.model.dataset['x'].to_sparse()
        torch.cuda.empty_cache()
        #g.ndata['cls_feats'][self.model.dataset.train_mask] = cls_feat
        return None

    def train(self):
        self.model.bertGcnModel.train()

        lossTrain = total_examples = 0

        for batch in self.model.trainLoader:
            self.model.optimizer.zero_grad()

            y = batch.y[:batch.batch_size]
            y_hat, cls_feats = self.model.bertGcnModel(batch.x,
                                            batch.n_id,
                                            batch.edge_index,
                                            batch.edge_weight, 
                                            batch.input_ids[:batch.batch_size], 
                                            batch.attention_mask[:batch.batch_size], 
                                            batch.batch_size,
                                            None)

            loss = self.criterion(y_hat, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.bertGcnModel.parameters(), 1.0)

            self.model.optimizer.step()
            self.model.scheduler.step()

            self.metricsTrainBatch(y_hat, y.int())

            lossTrain += float(loss) * batch.batch_size
            total_examples += batch.batch_size

            self.model.dataset.x[batch.n_id[:batch.batch_size]] = cls_feats.detach().clone()

        metric = self.metricsTrainBatch.compute()
        mlflow.log_metrics({label:round(value.item(),4) for label, value in metric.items()})
        mlflow.log_metrics({'Train_Batch_loss':round(lossTrain/total_examples,4)})
        metric = self.metricsTrainBatch.reset()

        return None

    def test_test(self,epoch_i):
        
        self.model.bertGcnModel.eval()
        with torch.no_grad():

            lossTest = total_examples = 0

            for batch in self.model.testLoader:
                self.model.optimizer.zero_grad()

                y = batch.y[:batch.batch_size]
                y_hat, _ = self.model.bertGcnModel(batch.x,
                                                batch.n_id,
                                                batch.edge_index,
                                                batch.edge_weight, 
                                                batch.input_ids[:batch.batch_size], 
                                                batch.attention_mask[:batch.batch_size], 
                                                batch.batch_size,
                                                None)

                loss = self.criterion(y_hat, y)

                self.metricsTestEpoch(y_hat, y.int())

                lossTest += float(loss) * batch.batch_size
                total_examples += batch.batch_size

            metric = self.metricsTestEpoch.compute()
            mlflow.log_metrics({label:round(value.item(),4) for label, value in metric.items()},epoch_i+1)
            mlflow.log_metrics({'Test_Epoch_loss':round(lossTest/total_examples,4)},epoch_i+1)
            metric = self.metricsTestEpoch.reset()

        return lossTest/total_examples

    def test(self,epoch_i):
        with torch.no_grad():
            self.model.bertGcnModel.eval()
            #y_hat = self.model.bertGcnModel.inference(self.model.dataset.x, self.model.subgraphLoader).to(self.device)
            y_hat, _ = self.model.bertGcnModel(self.model.dataset.x, None, None, None, None, None, None, self.model.subgraphLoader)
            y = self.model.dataset.y.to(y_hat.device)

            lossTrain = self.logCrit(y_hat[self.model.dataset.train_mask], y[self.model.dataset.train_mask])
            lossTest = self.logCrit(y_hat[self.model.dataset.test_mask], y[self.model.dataset.test_mask])
            lossVal = self.logCrit(y_hat[self.model.dataset.val_mask], y[self.model.dataset.val_mask])

            self.metricsTrainEpoch(y_hat[self.model.dataset.train_mask], y[self.model.dataset.train_mask].int())
            metric = self.metricsTrainEpoch.compute()
            mlflow.log_metrics({label:round(value.item(),4) for label, value in metric.items()},epoch_i+1)
            mlflow.log_metrics({'Train_Epoch_loss':round(lossTrain.item(),4)},epoch_i+1)
            metric = self.metricsTrainEpoch.reset()

            self.metricsTestEpoch(y_hat[self.model.dataset.test_mask], y[self.model.dataset.test_mask].int())
            metric = self.metricsTestEpoch.compute()
            mlflow.log_metrics({label:round(value.item(),4) for label, value in metric.items()},epoch_i+1)
            mlflow.log_metrics({'Test_Epoch_loss':round(lossTest.item(),4)},epoch_i+1)

            earlyStopCriteria = lossTest.item()

            metric = self.metricsTestEpoch.reset()

            self.metricsValEpoch(y_hat[self.model.dataset.val_mask], y[self.model.dataset.val_mask].int())
            metric = self.metricsValEpoch.compute()
            mlflow.log_metrics({label:round(value.item(),4) for label, value in metric.items()},epoch_i+1)
            mlflow.log_metrics({'Val_Epoch_loss':round(lossVal.item(),4)},epoch_i+1)
            metric = self.metricsValEpoch.reset()

            return earlyStopCriteria

    def fit_and_eval(self):

        for epoch_i in tqdm(range(self.model.epochs)):

            #update features:
            self.update_doc_features()

            self.model.dataset = self.model.dataset.to(self.model.device)
            self.train()

            curTestLoss = self.test_test(epoch_i)

            if epoch_i > 3:
                
                earlyStopCriteria = curTestLoss

                if self.earlyStopper.early_stop(earlyStopCriteria):
                    break

        mlflow.log_metrics({'Minute Duration':round((datetime.now() - self.starttime).total_seconds()/60,0)},self.model.epochs)