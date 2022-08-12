import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class FaceLandmarksDataset(Dataset):
    """Landmarks dataset"""

    def __init__(self, csv_file, transform=None):

        self.landmarks_frame = pd.read_csv(csv_file)
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

#def main():

    #filePath = '/home/jaco/Projetos/landMarkClassification/data/onlyLandMarkWSyllabus.csv'
    
    #land_dataset = FaceLandmarksDataset(csv_file=filePath)

    #train_iter = iter(FaceLandmarksDataset(csv_file=filePath))

    #return None


#if __name__ == '__main__':
    #main()