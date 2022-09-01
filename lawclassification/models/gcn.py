import os
import pandas as pd
from utils.definitions import ROOT_DIR

def main(dataname:str) -> None:

    pathFile = os.path.join(ROOT_DIR,'lawclassification','data',dataname,'graph.csv')
    df = pd.read_csv(pathFile)

if __name__ == '__main__':
    main(dataname='yelp')