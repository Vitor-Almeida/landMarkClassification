from tqdm.auto import tqdm
import torch
import gc
from models.deep_models import deep_models
from utils.deep_metrics import metrics_config
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
                                 experiment['qty_layer_unfreeze'])
        
        self.log_every_n_steps = 50
        self.logInterval = int(self.model.total_steps/self.log_every_n_steps)

        _ = metrics_config(num_labels = self.model.num_labels,
                           device = self.model.device,
                           problem_type = self.model.problem_type)

        self.metricsTrainEpoch = _.clone(prefix='Train_Epoch_')
        self.metricsTrainBatch = _.clone(prefix='Train_Batch_')

        self.metricsTestEpoch = _.clone(prefix='Test_Epoch_')
        self.metricsTestBatch = _.clone(prefix='Test_Batch_')

        self.metricsValEpoch = _.clone(prefix='Val_Epoch_')
        self.metricsValBatch = _.clone(prefix='Val_Batch_')

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

        self.progress_bar = tqdm(range(self.model.total_steps))

        #print_params_terminal(self.model)

    def train_loop(self):

        self.model.model.train()
        scaler = torch.cuda.amp.GradScaler(enabled=True)

        for idx,batch in enumerate(self.model.train_dataloader):
            self.model.optimizer.zero_grad()
            batch = {k: v.to(self.model.device) for k, v in batch.items()}

            #mixed precision training:
            with torch.autocast(device_type=self.model.device.type, dtype=torch.float16, enabled=True):
                outputs = self.model.model(**batch)
                loss = outputs.loss 

            scaler.scale(loss).backward() #loss.backward()
            scaler.unscale_(self.model.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 1.0)
            
            scaler.step(self.model.optimizer) #self.model.optimizer.step()
            scaler.update()
            
            self.model.scheduler.step()

            self.metricsTrainBatch(outputs.logits, batch['labels'].int())
            self.metricsTrainEpoch(outputs.logits, batch['labels'].int())

            if idx % self.logInterval == 0 and idx > 0:
                metric = self.metricsTrainBatch.compute()
                mlflow.log_metrics({label:round(value.item(),4) for label, value in metric.items()}
                )
                metric = self.metricsTrainBatch.reset()

            self.progress_bar.update(1)

        return None

    def test_loop(self):

        self.model.model.eval()

        with torch.no_grad():
            for idx,batch in enumerate(self.model.test_dataloader):
                batch = {k: v.to(self.model.device) for k, v in batch.items()}

                with torch.autocast(device_type=self.model.device.type, dtype=torch.float16, enabled=True):
                    outputs = self.model.model(**batch)

                #lexGlue tem q fazer um if self.model.dataname == 'unfair-tos', cat([1] ou [0] no começo do vetor se a label for [0,0,0..0])
                #problema q talvez vai bugar a definição das métricas

                self.metricsTestEpoch(outputs.logits, batch['labels'].int())

        return None

    def val_loop(self):

        self.model.model.eval()

        with torch.no_grad():
            for _,batch in enumerate(self.model.val_dataloader):
                batch = {k: v.to(self.model.device) for k, v in batch.items()}

                with torch.autocast(device_type=self.model.device.type, dtype=torch.float16, enabled=True):
                    outputs = self.model.model(**batch)

                #self.metricsValBatch(outputs.logits, batch['labels'].int())
                self.metricsValEpoch(outputs.logits, batch['labels'].int())

        return None

    def fit_and_eval(self):

        for epoch_i in range(0, self.model.epochs):

            self.train_loop()
            metric = self.metricsTrainEpoch.compute()
            mlflow.log_metrics({label:round(value.item(),4) for label, value in metric.items()},epoch_i+1)
            metric = self.metricsTrainEpoch.reset()

            self.test_loop()
            metric = self.metricsTestEpoch.compute()
            mlflow.log_metrics({label:round(value.item(),4) for label, value in metric.items()},epoch_i+1)
            metric = self.metricsTestEpoch.reset()

        self.val_loop()
        metric = self.metricsValEpoch.compute()
        mlflow.log_metrics({label:round(value.item(),4) for label, value in metric.items()},self.model.epochs)
        metric = self.metricsValEpoch.reset()

        #self.progress_bar.refresh()
        self.progress_bar.reset()
        torch.cuda.empty_cache()
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

        ### tentar limpar tudo aqui