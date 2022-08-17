import pandas as pd
import os
import gc
from sklearn.model_selection import train_test_split
from utils.definitions import ROOT_DIR

#pyton -m dataset.landMarkTorchDataset => rodar testes 

def yelp_review(max_data_size, test_split):

    """
    Load raw data in a pytorch dataset class
    All data will follow the (label,text) format
    """

    df = pd.read_json(os.path.join(ROOT_DIR,'data','yelp','raw','yelp_academic_dataset_review.json'), lines=True, nrows = max_data_size)

    #gamb class:
    df = df[(df['stars']==1) | (df['stars']==5) | (df['stars']==3)]
    df = df[df["text"].apply(lambda x: len(x) > 200)] 

    dataset = df[['stars','text']]

    del df
    gc.collect()

    dataset = dataset.sample(frac=1) # shuffle data
    encodedClass = pd.factorize(dataset['stars']) # encode the category target label [1...n]
    dataset['labels'] = encodedClass[0]#+1 ## pytorch needs to begins with a 1 (?)
    dataset.drop(columns=['stars'],inplace=True)

    X_train, X_testval, y_train, y_testval = train_test_split(dataset['text'],dataset['labels'], test_size=test_split, stratify= dataset['labels'])

    del dataset
    gc.collect()

    XTestval = pd.DataFrame()
    XTestval['labels'] = y_testval.to_frame()
    XTestval['text'] = X_testval.to_frame()

    X_test, X_val, y_test, y_val = train_test_split(XTestval['text'],XTestval['labels'], test_size=test_split, stratify= XTestval['labels'] )

    del XTestval
    gc.collect()

    XTrain = pd.DataFrame()
    XTrain['labels'] = y_train.to_frame()
    XTrain['text'] = X_train.to_frame()

    XTest = pd.DataFrame()
    XTest['labels'] = y_test.to_frame()
    XTest['text'] = X_test.to_frame()

    Xval = pd.DataFrame()
    Xval['labels'] = y_val.to_frame()
    Xval['text'] = X_val.to_frame()
    
    XTest.to_csv(f'{ROOT_DIR}/data/yelp/interm/test/test.csv',index=False)
    XTrain.to_csv(f'{ROOT_DIR}/data/yelp/interm/train/train.csv',index=False)
    Xval.to_csv(f'{ROOT_DIR}/data/yelp/interm/val/val.csv',index=False)

    return None