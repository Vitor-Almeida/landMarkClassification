from tqdm.auto import tqdm
import torch
from models.deep_models import deep_models
from utils.helper_funs import print_params_terminal,print_step_log,print_batch_log
from utils.deep_metrics import deep_metrics, execute_metrics_type, compute_metrics_type

class deep_train():
    """
    Train diferent deep models
    """
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
                                 experiment['problem_type'])
        
        self.metrics = deep_metrics(self.model)
        self.progress_bar = tqdm(range(self.model.total_steps))

        print_params_terminal(self.model)

    def train_loop(self,epoch_i):

        self.model.model.train()

        for idx,batch in enumerate(self.model.train_dataloader):
            self.model.optimizer.zero_grad()
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            outputs = self.model.model(**batch)
            loss = outputs.loss 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 1.0)
            self.model.optimizer.step()
            self.model.scheduler.step() #diferença aqui entre o tutorial do pytorch ?

            metricsResults = execute_metrics_type(self.metrics.metricDic['Train'],outputs.logits, batch['labels'].int())

            print_step_log(idx,epoch_i,self.model,metricsResults['Batch']) #fica dando o resultado de acuracia do step idx, 'nao eh acumulado'

            self.progress_bar.update(1)

        return None

    def test_loop(self):

        self.model.model.eval()

        with torch.no_grad():
            for batch in self.model.test_dataloader:
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                outputs = self.model.model(**batch)

                metricsResults = execute_metrics_type(self.metrics.metricDic['Test'],outputs.logits, batch['labels'].int())

        return None

    def val_loop(self):

        self.model.model.eval()

        with torch.no_grad():
            for batch in self.model.val_dataloader:
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                outputs = self.model.model(**batch)

                metricsResults = execute_metrics_type(self.metrics.metricDic['Val'],outputs.logits, batch['labels'].int())

        return None

    def fit_and_eval(self):

        for epoch_i in range(0, self.model.epochs):

            self.train_loop(epoch_i)
            trainTmpResult = compute_metrics_type(self.metrics.metricDic['Train']['Batch'],action='compute')
            print('------------Train results:----------------')
            print_batch_log(epoch_i,self.model,trainTmpResult)
            trainTmpResult = compute_metrics_type(self.metrics.metricDic['Train']['Batch'],action='reset')

            self.test_loop()
            testTmpResult = compute_metrics_type(self.metrics.metricDic['Test']['Batch'],action='compute')
            print('------------Test results:----------------')
            print_batch_log(epoch_i,self.model,testTmpResult)
            testTmpResult = compute_metrics_type(self.metrics.metricDic['Test']['Batch'],action='reset')

        self.val_loop()
        valTmpResult = compute_metrics_type(self.metrics.metricDic['Val']['Batch'],action='compute')
        print('------------Validation results:----------------')
        print_batch_log(self.model.epochs-1,self.model,valTmpResult)
        valTmpResult = compute_metrics_type(self.metrics.metricDic['Val']['Batch'],action='reset')