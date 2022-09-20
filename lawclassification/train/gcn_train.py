import copy
import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from models.gcn_models import Text_GCN
from utils.definitions import ROOT_DIR
import gc
import pickle


class gcn_train():

    def __init__(self,experiment):

        super(gcn_train, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        f = open(os.path.join(ROOT_DIR,'data',experiment['dataname'],'pygraph.pickle'),'rb')
        self.dataset = pickle.load(f).to(self.device)
        f.close()

        self.epochs = experiment['epochs']

        self.model = Text_GCN(self.dataset.num_features, experiment['hidden_channels'], self.dataset.num_classes, self.device).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=experiment['lr'])

        self.trainLoader = NeighborLoader(self.dataset, input_nodes=self.dataset.train_mask,
                                          num_neighbors=[-1, 10], shuffle=True, 
                                          batch_size = experiment['batchsize']
                                          #num_workers = 10, persistent_workers = True
                                          )

        self.subgraphLoader = NeighborLoader(copy.copy(self.dataset), input_nodes=None,
                                             num_neighbors=[-1], shuffle=False,
                                             batch_size = experiment['batchsize']
                                             #num_workers= 6, persistent_workers = True
                                             )


        del self.subgraphLoader.data.x, self.subgraphLoader.data.y
        gc.collect()
        self.subgraphLoader.data.num_nodes = self.dataset.num_nodes
        self.subgraphLoader.data.n_id = torch.arange(self.dataset.num_nodes)

    def train(self):
        self.model.train()

        total_loss = total_correct = total_examples = 0
        for batch in self.trainLoader:
            self.optimizer.zero_grad()
            y = batch.y[:batch.batch_size]
            y_hat = self.model(batch.x, batch.edge_index, batch.edge_weight)[:batch.batch_size]
            loss = F.cross_entropy(y_hat, y)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss) * batch.batch_size
            total_correct += int((y_hat.argmax(dim=-1) == y).sum())
            total_examples += batch.batch_size

        return total_loss / total_examples, total_correct / total_examples


    def test(self):
        with torch.no_grad():
            self.model.eval()
            y_hat = self.model.inference(self.dataset.x, self.subgraphLoader).argmax(dim=-1)
            y = self.dataset.y.to(y_hat.device)

            accs = []
            for mask in [self.dataset.train_mask, self.dataset.val_mask, self.dataset.test_mask]:
                accs.append(int((y_hat[mask] == y[mask]).sum()) / int(mask.sum()))
            return accs

    def fit_and_eval(self):

        best_test_acc = best_val_acc = 0

        for epoch_i in range(0, self.epochs):
            loss, _ = self.train()
            train_acc, val_acc, test_acc = self.test()
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_val_acc = val_acc

            print(f'Epoch: {epoch_i:02d}, Train Loss : {loss:.4f} Train: {train_acc:.4f}, Val: {best_val_acc:.4f}, Test: {best_test_acc:.4f}')

        torch.cuda.empty_cache()