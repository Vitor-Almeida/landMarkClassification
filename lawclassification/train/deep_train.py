from tqdm.auto import tqdm
import torch
import gc
from models.deep_models import deep_models
from utils.helper_funs import print_params_terminal,print_step_log,print_batch_log
from utils.deep_metrics import deep_metrics, execute_metrics_type, compute_metrics_type, reset_all
import mlflow

class deep_train():
    """
    Train diferent deep models
    """

    #mlflow ui --backend-store-uri sqlite:///mlflow.db

    def __init__(self,experiment):

        super(deep_train, self).__init__()
        
        self.model = deep_models(experiment['model_name'], 
                                 experiment['batchsize'],
                                 experiment['max_char_length'],
                                 experiment['lr'],
                                 experiment['epochs'],
                                 experiment['warmup_size'],
                                 experiment['dropout'],
                                 experiment['dataname'],
                                 experiment['problem_type'],
                                 experiment['weight_decay'],
                                 experiment['decay_lr'])
        
        self.metrics = deep_metrics(self.model)
        self.progress_bar = tqdm(range(self.model.total_steps))

        print_params_terminal(self.model)

    def train_loop(self,epoch_i):

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

            metricsResults = execute_metrics_type(self.metrics.metricDic['Train'],outputs.logits, batch['labels'].int())
            print_step_log(idx,epoch_i,self.model,metricsResults['Batch']) #fica dando o resultado de acuracia do step idx, 'nao eh acumulado'
            self.progress_bar.update(1)

        return None

    def test_loop(self):

        self.model.model.eval()

        with torch.no_grad():
            for batch in self.model.test_dataloader:
                batch = {k: v.to(self.model.device) for k, v in batch.items()}

                with torch.autocast(device_type=self.model.device.type, dtype=torch.float16, enabled=True):
                    outputs = self.model.model(**batch)

                #lexGlue tem q fazer um if self.model.dataname == 'unfair-tos', cat([1] ou [0] no começo do vetor se a label for [0,0,0..0])
                #problema q talvez vai bugar a definição das métricas

                metricsResults = execute_metrics_type(self.metrics.metricDic['Test'],outputs.logits, batch['labels'].int())

        return None

    def val_loop(self):

        self.model.model.eval()

        with torch.no_grad():
            for batch in self.model.val_dataloader:
                batch = {k: v.to(self.model.device) for k, v in batch.items()}

                with torch.autocast(device_type=self.model.device.type, dtype=torch.float16, enabled=True):
                    outputs = self.model.model(**batch)

                metricsResults = execute_metrics_type(self.metrics.metricDic['Val'],outputs.logits, batch['labels'].int())

        return None

    def fit_and_eval(self):

        for epoch_i in range(0, self.model.epochs):

            self.train_loop(epoch_i)
            trainTmpResult = compute_metrics_type(self.metrics.metricDic['Train']['Batch'],action='compute')
            print('------------Train results:-------------------------')
            print_batch_log(epoch_i,self.model,trainTmpResult)
            trainTmpResult = compute_metrics_type(self.metrics.metricDic['Train']['Batch'],action='reset')
            print('Starting test loop...')
            self.test_loop()
            testTmpResult = compute_metrics_type(self.metrics.metricDic['Test']['Batch'],action='compute')
            print('------------Test results:-----------------------')
            print_batch_log(epoch_i,self.model,testTmpResult)
            testTmpResult = compute_metrics_type(self.metrics.metricDic['Test']['Batch'],action='reset')

        self.val_loop()
        valTmpResult = compute_metrics_type(self.metrics.metricDic['Val']['Batch'],action='compute')
        print('------------Validation results:----------------')
        print_batch_log(self.model.epochs-1,self.model,valTmpResult)
        valTmpResult = compute_metrics_type(self.metrics.metricDic['Val']['Batch'],action='reset')

        #self.progress_bar.refresh()
        self.progress_bar.reset()
        reset_all(self.metrics.metricDic)
        torch.cuda.empty_cache()
        del self.model
        del self.metrics
        gc.collect()
        torch.cuda.empty_cache()

        ### tentar limpar tudo aqui