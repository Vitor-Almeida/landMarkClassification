from tqdm.auto import tqdm
import torch
import gc
from models.deep_models import deep_models
from utils.deep_metrics import metrics_config
from utils.helper_funs import save_model
import mlflow

class deep_train():
    """
    Train diferent deep models
    """

    def __init__(self,experiment):

        super(deep_train, self).__init__()
        
        self.model = deep_models(experiment['model_name'], 
                                 int(experiment['batchsize']),
                                 int(experiment['max_char_length']),
                                 experiment['lr'],
                                 int(experiment['epochs']),
                                 experiment['warmup_size'],
                                 experiment['dropout'],
                                 experiment['dataname'],
                                 experiment['problem_type'],
                                 experiment['weight_decay'],
                                 experiment['decay_lr'],
                                 experiment['qty_layer_unfreeze'],
                                 bool(experiment['hierarchical']),
                                 int(experiment['hier_max_seg']),
                                 int(experiment['hier_max_seg_length'])
                                 )
        
        self.flag_mixed_precision = not(bool(experiment['hierarchical']))
        self.log_every_n_steps = 50
        self.logInterval = int(self.model.total_steps/self.log_every_n_steps)

        _ = metrics_config(num_labels = self.model.num_labels,
                           device = self.model.device,
                           problem_type = self.model.problem_type)

        self.metricsTrainEpoch = _.clone(prefix='Train_Epoch_')
        #self.metricsTrainBatch = _.clone(prefix='Train_Batch_')

        self.metricsTestEpoch = _.clone(prefix='Test_Epoch_')
        #self.metricsTestBatch = _.clone(prefix='Test_Batch_')

        self.metricsValEpoch = _.clone(prefix='Val_Epoch_')
        #self.metricsValBatch = _.clone(prefix='Val_Batch_')

        self.datasetParams = {'train_labels':self.model.num_labels_train,
                              'dataset_length':len(self.model.dataset_test) + len(self.model.dataset_train) + len(self.model.dataset_val),
                              'random_seed':self.model.seed_val,
                              'dataset_train_lenght':len(self.model.dataset_train),
                              'num_labels_train':self.model.num_labels_train,
                              'dataset_test_lenght':len(self.model.dataset_test),
                              'num_labels_test':self.model.num_labels_test,
                              'dataset_val_lenght':len(self.model.dataset_val),
                              'num_labels_val':self.model.num_labels_val}

        mlflow.log_params(self.datasetParams)

        #print_params_terminal(self.model)

    def train_loop(self):

        self.model.model.train()
        scaler = torch.cuda.amp.GradScaler(enabled=True)

        epochLoss = batchItens = 0 

        for idx,batch in enumerate(self.model.train_dataloader):
            
            batch = {k: v.to(self.model.device) for k, v in batch.items()}

            #mixed precision training:
            with torch.autocast(device_type=self.model.device.type, dtype=torch.float16, enabled=True):
                outputs = self.model.model(**batch)
                loss = outputs.loss 

            scaler.scale(loss).backward() #loss.backward()
            scaler.unscale_(self.model.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 1.0)
            
            scaler.step(self.model.optimizer) #self.model.optimizer.step()
            self.model.scheduler.step()

            #torch.isnan(x)
            #torch.isinf(x)
            
            scaler.update()
            self.model.optimizer.zero_grad(set_to_none = True)

            self.metricsTrainEpoch(outputs.logits, batch['labels'].int())

            epochLoss += loss * batch['input_ids'].size()[1]
            batchItens += batch['input_ids'].size()[1]

        #self.model.scheduler.step()

        epochLoss = epochLoss / batchItens

        return epochLoss.item()

    def test_loop(self):

        self.model.model.eval()

        epochLoss = batchItens = 0 

        with torch.no_grad():
            for idx,batch in enumerate(self.model.test_dataloader):
                batch = {k: v.to(self.model.device) for k, v in batch.items()}

                with torch.autocast(device_type=self.model.device.type, dtype=torch.float16, enabled=self.flag_mixed_precision):
                #with torch.autocast(device_type=self.model.device.type, dtype=torch.float16, enabled=True):
                    outputs = self.model.model(**batch)

                #lexGlue tem q fazer um if self.model.dataname == 'unfair-tos', cat([1] ou [0] no começo do vetor se a label for [0,0,0..0])
                #problema q talvez vai bugar a definição das métricas

                self.metricsTestEpoch(outputs.logits, batch['labels'].int())

                epochLoss += outputs.loss * batch['input_ids'].size()[1]
                batchItens += batch['input_ids'].size()[1]

        epochLoss = epochLoss / batchItens

        return epochLoss.item()

    def val_loop(self):

        self.model.model.eval()

        epochLoss = batchItens = 0 

        with torch.no_grad():
            for _,batch in enumerate(self.model.val_dataloader):
                batch = {k: v.to(self.model.device) for k, v in batch.items()}

                with torch.autocast(device_type=self.model.device.type, dtype=torch.float16, enabled=self.flag_mixed_precision):
                    outputs = self.model.model(**batch)

                #self.metricsValBatch(outputs.logits, batch['labels'].int())
                self.metricsValEpoch(outputs.logits, batch['labels'].int())

                epochLoss += outputs.loss * batch['input_ids'].size()[1]
                batchItens += batch['input_ids'].size()[1]

        epochLoss = epochLoss / batchItens

        return epochLoss.item()

    def fit_and_eval(self):

        for epoch_i in tqdm(range(0, self.model.epochs)):

            trainLoss = self.train_loop()
            metric = self.metricsTrainEpoch.compute()
            mlflow.log_metrics({label:round(value.item(),4) for label, value in metric.items()},epoch_i+1)
            mlflow.log_metrics({'Train_Epoch_loss':round(trainLoss,4)},epoch_i+1)
            metric = self.metricsTrainEpoch.reset()

            testLoss = self.test_loop()
            metric = self.metricsTestEpoch.compute()
            mlflow.log_metrics({label:round(value.item(),4) for label, value in metric.items()},epoch_i+1)
            mlflow.log_metrics({'Test_Epoch_loss':round(testLoss,4)},epoch_i+1)
            metric = self.metricsTestEpoch.reset()

            if self.model.earlyStopper.early_stop(testLoss):
                save_model(self.model, epoch_i)
                break

        valLoss = self.val_loop()
        metric = self.metricsValEpoch.compute()
        mlflow.log_metrics({label:round(value.item(),4) for label, value in metric.items()},self.model.epochs)
        mlflow.log_metrics({'Val_Epoch_loss':round(valLoss,4)},epoch_i+1)
        metric = self.metricsValEpoch.reset()
