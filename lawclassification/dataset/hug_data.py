import pandas as pd
import numpy as np
from datasets import load_dataset

def scotus(df):

    



    return None


def get_hug_data():

    '''
    Download and save huggin face lexGlue.
    '''

    subset = ['unfair_tos','ledgar','scotus','eurlex','ecthr_b','ecthr_a']
    foldername = ['unfair_huggin','ledgar_huggin','scotus_huggin','eurlex_huggin','ecthr_b_huggin','ecthr_a_huggin']
    bugados = ['ledgar','scotus']

    cache_path = '/home/jaco/Projetos/landMarkClassification/data/tmp_huggin'

    #dataHug = {'train':[],'test':[],'validation':[]}

    dataHug = {'train':{},'test':{},'val':{}}

    for setb in subset:
        dataHug['train'][setb] = load_dataset("lex_glue", setb, split="train", cache_dir=cache_path+'/'+setb)
        dataHug['test'][setb] = load_dataset("lex_glue", setb, split="test", cache_dir=cache_path+'/'+setb)
        dataHug['val'][setb] = load_dataset("lex_glue", setb, split="validation", cache_dir=cache_path+'/'+setb)
    
    dfList = []
    tmpDfList = []

    for split in dataHug:
        for data in dataHug[split]:
            textArr = dataHug[split][data]['text']

            if data in bugados:
                labelArr = dataHug[split][data]['label']
            else:
                labelArr = dataHug[split][data]['labels']

            datanameArr = [data] * len(textArr)
            splitnameArr = [split] * len(textArr)

            tmpDf = pd.DataFrame(list(zip(textArr, labelArr, datanameArr, splitnameArr)),columns =['text','labels','dataname','split'])

            dfList.append(tmpDf)

    df = pd.concat(dfList,ignore_index=True)
    df['docId'] = df.groupby(['dataname'])['dataname'].rank('first').astype(np.int64) - 1

    return df

def main():

    df = get_hug_data()    

    scotus(df[df['dataname']=='scotus'])

    return None


if __name__ == '__main__':
    main()