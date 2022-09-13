import os
import torch
#from torch_geometric.data import Dataset
from torch_geometric.data import InMemoryDataset
from utils.definitions import ROOT_DIR

class deep_graph(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        super(deep_graph,self).__init__(root, name, transform, pre_transform)

        self.name = name

    @property
    def raw_file_names(self):
        return [os.path.join(ROOT_DIR,'data',self.name,'interm','graph.csv')]

    @property
    def processed_file_names(self):
        #return [os.path.join(ROOT_DIR,'data',self.name,'interm','graphProcessed.csv')]
        return [os.path.join(ROOT_DIR,'data',self.name,'interm','graph.pt')]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = [...]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    #def len(self):
    #    return len(self.processed_file_names)

    #def get(self, idx):
    #    data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
    #    return data


x = deep_graph(name='ohsumed')