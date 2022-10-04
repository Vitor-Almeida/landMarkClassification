import torch
from models.bertGcn_models import ensemble_model
from utils.deep_metrics import metrics_config, f1ajust_lexglue, metrics_config_special
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

        _special = metrics_config_special(num_labels = self.model.dataset.num_classes+1,
                                          device = self.model.device)

        self.metricsTestEpochSpecial = _special.clone(prefix='Test_Special_')
        self.metricsValEpochSpecial = _special.clone(prefix='Val_Special_')

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
                              'model_name':self.model.model_name,
                              'epochs':self.model.epochs,
                              'problemType':self.model.problemType,
                              'hierarchical':self.model.flag_hierarchical,
                              'decay_lr':self.model.decay_lr,
                              'weight_decay':self.model.weight_decay,
                              'qty_layer_unfreeze':self.model.qtyFracLayers,
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
            self.model.bertGcnModel.eval()
            cls_list = []
            for batch in self.model.updateDataLoader:
                input_ids = batch[0].to(self.model.device)
                #input_ids = batch.token_w_hier_id[:batch.batch_size]
                attention_mask = batch[1].to(self.model.device)
                #attention_mask = batch.token_w_hier_att[:batch.batch_size]
                token_type_ids = batch[2].to(self.model.device)
                #token_type_ids = batch.token_w_hier_tid[:batch.batch_size]

                #colocar só train?
                with torch.autocast(device_type=self.model.device.type, dtype=torch.float16, enabled=not(self.model.flag_hierarchical)):
                    output = self.model.bertGcnModel.bert_model.bert(input_ids=input_ids, 
                                                                     attention_mask=attention_mask, 
                                                                     token_type_ids=token_type_ids).pooler_output

                cls_list.append(output.cpu())
            cls_feat = torch.cat(cls_list, axis=0)
        self.model.dataset = self.model.dataset.to('cpu')
        self.model.dataset.x[self.model.dataset.docmask] = cls_feat.to(torch.float32) #sera que ta na mesma ordem?
        torch.cuda.empty_cache()
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
                                                       batch.token_w_hier_id[:batch.batch_size], 
                                                       batch.token_w_hier_att[:batch.batch_size], 
                                                       batch.token_w_hier_tid[:batch.batch_size], 
                                                       batch.batch_size,
                                                       None)

            loss = self.criterion(y_hat, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.bertGcnModel.parameters(), 1.0)

            self.model.optimizer.step()
            
            self.metricsTrainBatch(y_hat, y.int())

            lossTrain += loss.item() * batch.batch_size
            total_examples += batch.batch_size

            self.model.dataset.x[batch.n_id[:batch.batch_size]] = cls_feats.detach()

        self.model.scheduler.step()

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
                                                   batch.token_w_hier_id[:batch.batch_size], 
                                                   batch.token_w_hier_att[:batch.batch_size], 
                                                   batch.token_w_hier_tid[:batch.batch_size], 
                                                   batch.batch_size,
                                                   None)

                if self.model.dataname in ['ecthr_b_lexbench','ecthr_a_lexbench','unfair_lexbench']:
                    out,lab = f1ajust_lexglue(y_hat, y, self.model.device, True)
                    self.metricsTestEpochSpecial(out, lab.int())

                loss = self.criterion(y_hat, y)

                self.metricsTestEpoch(y_hat, y.int())

                lossTest += loss.item() * batch.batch_size
                total_examples += batch.batch_size

            metric = self.metricsTestEpoch.compute()
            mlflow.log_metrics({label:round(value.item(),4) for label, value in metric.items()},epoch_i+1)
            mlflow.log_metrics({'Test_Epoch_loss':round(lossTest/total_examples,4)},epoch_i+1)
            metric = self.metricsTestEpoch.reset()

            if self.model.dataname in ['ecthr_b_lexbench','ecthr_a_lexbench','unfair_lexbench']:
                metric = self.metricsTestEpochSpecial.compute()
                mlflow.log_metrics({label:round(value.item(),4) for label, value in metric.items()},epoch_i+1)
                metric = self.metricsTestEpochSpecial.reset()

        return lossTest/total_examples

    def val_val(self,epoch_i):
        
        self.model.bertGcnModel.eval()
        with torch.no_grad():

            lossTest = total_examples = 0

            for batch in self.model.valLoader:
                self.model.optimizer.zero_grad()

                y = batch.y[:batch.batch_size]
                y_hat, _ = self.model.bertGcnModel(batch.x,
                                                   batch.n_id,
                                                   batch.edge_index,
                                                   batch.edge_weight, 
                                                   batch.token_w_hier_id[:batch.batch_size], 
                                                   batch.token_w_hier_att[:batch.batch_size], 
                                                   batch.token_w_hier_tid[:batch.batch_size], 
                                                   batch.batch_size,
                                                   None)

                if self.model.dataname in ['ecthr_b_lexbench','ecthr_a_lexbench','unfair_lexbench']:
                    out,lab = f1ajust_lexglue(y_hat, y, self.model.device, True)
                    self.metricsValEpochSpecial(out, lab.int())

                loss = self.criterion(y_hat, y)

                self.metricsValEpoch(y_hat, y.int())

                lossTest += loss.item() * batch.batch_size
                total_examples += batch.batch_size

            metric = self.metricsValEpoch.compute()
            mlflow.log_metrics({label:round(value.item(),4) for label, value in metric.items()},epoch_i+1)
            mlflow.log_metrics({'Test_Epoch_loss':round(lossTest/total_examples,4)},epoch_i+1)
            metric = self.metricsValEpoch.reset()

            if self.model.dataname in ['ecthr_b_lexbench','ecthr_a_lexbench','unfair_lexbench']:
                metric = self.metricsValEpochSpecial.compute()
                mlflow.log_metrics({label:round(value.item(),4) for label, value in metric.items()},epoch_i+1)
                metric = self.metricsValEpochSpecial.reset()

        return lossTest/total_examples

    def fit_and_eval(self):

        for epoch_i in tqdm(range(self.model.epochs)):

            #update features:
            self.update_doc_features()

            self.model.dataset = self.model.dataset.to(self.model.device)
            self.train()

            curTestLoss = self.test_test(epoch_i)

            #torch.cuda.empty_cache()

            if epoch_i > 3:
                
                earlyStopCriteria = curTestLoss

                if self.earlyStopper.early_stop(earlyStopCriteria):
                    break
    
        self.val_val(self.model.epochs)
        mlflow.log_metrics({'Minute Duration':round((datetime.now() - self.starttime).total_seconds()/60,0)},self.model.epochs)