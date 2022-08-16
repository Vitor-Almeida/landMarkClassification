import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

#from utils.definitions import ROOT_DIR

ROOT_DIR = '/home/jaco/Projetos/landMarkClassification'

#pyton -m dataset.landMarkTorchDataset => rodar testes 

class yelpReview(Dataset):
    """
    Load raw data in a pytorch dataset class
    All data will follow the (label,text) format
    """

    def __init__(self, transform=None):
        #self.landmarks_frame = pd.read_json(os.path.join(ROOT_DIR,'/data/yelp/raw/yelp_academic_dataset_review.json'), lines=True)
        self.landmarks_frame = pd.read_json(ROOT_DIR+'/data/yelp/raw/yelp_academic_dataset_review.json', lines=True)
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        text = self.landmarks_frame.iloc[idx,0]
        label = self.landmarks_frame.iloc[idx,1]

        sample = (label,text)

        if self.transform:
            sample = self.transform(sample)

        return sample

def main():

    data = yelpReview()
    print('cu')

    #filePath = '/home/jaco/Projetos/landMarkClassification/data/onlyLandMarkWSyllabus.csv'
    
    #land_dataset = FaceLandmarksDataset(csv_file=filePath)

    #train_iter = iter(FaceLandmarksDataset(csv_file=filePath))

    return None


if __name__ == '__main__':
    main()