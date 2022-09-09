import pandas as pd
import os
from utils.definitions import ROOT_DIR
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import math
from typing import List
#from multiprocessing import Process, Manager
from datetime import datetime
import multiprocessing
import itertools
from tqdm.auto import tqdm
import spacy
import gc

def _find_word_ngrams(vocabularyVec,idx,npAggText,return_dict,lastOrder):

    rowIndicesList = []
    for wordRow in vocabularyVec:
        indicesRow = [idx for idx,ngrams in enumerate(npAggText) if ' '+wordRow+' ' in ' '+ngrams+' ']
        rowIndicesList.append(indicesRow)

    return_dict[lastOrder]=rowIndicesList

def _split_listN(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def _calc_pmi(rdx:int,
              cdx:int,
              npAgg:np.array,
              rowIndicesList:List,
              totalWindows:np.int16) -> np.float16:

    rowSumTT = np.sum(npAgg[rowIndicesList[rdx]])
    colSumTT = np.sum(npAgg[rowIndicesList[cdx]])

    indicesRowandCol = list(np.intersect1d(rowIndicesList[rdx],rowIndicesList[cdx]))
    rowColSumTT = np.sum(npAgg[indicesRowandCol])

    pRow = rowSumTT / totalWindows
    pCol = colSumTT / totalWindows
    pRowCol = rowColSumTT / totalWindows
    
    if pRowCol == 0 or (pRow*pCol == 0):
        #avoid math errors
        pmi = 0
    else:
        pmi = math.log(pRowCol / (pRow*pCol),2)

    return pmi

NLP = spacy.load('en_core_web_lg')

#usar a funcao de fast tokenizers to hugginface
def _tokenizer(text:str) -> str:

    #text = text[0:3000] #~512 words
    newText = []

    #slow:
    for word in NLP(text):
        if (not word.is_stop or not word.is_punct) and (word.is_alpha and word.is_ascii and not word.is_digit):
            newText.append(word.lemma_.lower())

    return newText

def _load_csv(path: str,max_rows: int) -> pd.DataFrame:

    dfTrain = pd.read_csv(os.path.join(ROOT_DIR,'data',path,'interm','train','train.csv'))
    dfTrain['split'] = 'train'
    dfTest = pd.read_csv(os.path.join(ROOT_DIR,'data',path,'interm','test','test.csv'))
    dfTest['split'] = 'test'
    dfVal = pd.read_csv(os.path.join(ROOT_DIR,'data',path,'interm','val','val.csv'))
    dfVal['split'] = 'val'

    df = pd.concat([dfTrain,dfTest,dfVal],ignore_index=True)
    df = df.sample(frac=1)
    
    df = df.head(max_rows)
    df.reset_index(inplace=True,drop=True)

    return df

def create_graph(path: str,maxRows: int, windowSize:int) -> None:
    '''
    https://arxiv.org/pdf/1809.05679.pdf

    input is the dataname folder name from the data folder of the project

    Create a graph from a .csv file in the format:
    
    csv input:

    label | text
    1     | Something something ...
    .     | .
    .     | .
    n     | something something ...
    '''

    count_vector = CountVectorizer(tokenizer = _tokenizer,
                                #stop_words = 'english', 
                                #sublinear_tf = True, 
                                ngram_range=(windowSize,windowSize),
                                lowercase = True,
                                strip_accents = 'unicode',
                                encoding = 'utf-8',
                                decode_error = 'strict',
                                dtype = np.int16,
                                max_df = 1.0, #tirar 'outliers' palavras muito repetidas entre documentos
                                min_df = 1, #tirar 'outliers' palavras muito raras entre documentos#'cut-off'
                                max_features = None, #numero maximo de features
                                vocabulary = None)

    tfidf_vector = TfidfVectorizer(tokenizer = _tokenizer,
                                #stop_words = 'english', 
                                #sublinear_tf = True, 
                                ngram_range=(1,1),
                                lowercase = True,
                                strip_accents = 'unicode',
                                encoding = 'utf-8',
                                decode_error = 'strict',
                                dtype = np.float32,
                                max_df = 1.0, #tirar 'outliers' palavras muito repetidas entre documentos
                                min_df = 1, #tirar 'outliers' palavras muito raras entre documentos#'cut-off'
                                max_features = None, #numero maximo de features
                                vocabulary = None)


    dfTable = _load_csv(path,maxRows)

    countMatrix = count_vector.fit_transform(dfTable['text'])
    tfidf_matrix = tfidf_vector.fit_transform(dfTable['text'])
    labels = dfTable['labels'].to_frame()
    labels['src'] = labels.index
    labels['src'] = labels['src'].apply(lambda row: 'doc_'+str(row))

    splits = dfTable['split'].to_frame()
    splits['src'] = splits.index
    splits['src'] = splits['src'].apply(lambda row: 'doc_'+str(row))

    del dfTable
    gc.collect()

    vocabularyVec = tfidf_vector.get_feature_names_out().astype('U')
    vocabularyWindow = count_vector.get_feature_names_out().astype('U')

    dfCount = pd.DataFrame.sparse.from_spmatrix(countMatrix, columns=vocabularyWindow)

    dfMeltCount = []
    step = int(len(dfCount.columns) / 100)+1 #isso aqui até o limite da memoria
    for idx,cols in enumerate(range(0,len(dfCount.columns),step)):
        next = step if idx == 0 else step*(idx+1)
        next = len(dfCount.columns)+1 if next+step > len(dfCount.columns) else next
    
        tmpDf = pd.melt(dfCount.iloc[:,cols:next-1],value_name='__value__',var_name='__variable__')
        tmpDf['__value__'] = tmpDf['__value__'].astype(np.int16)
        tmpDf['__variable__'] = tmpDf['__variable__'].astype(str)
        tmpDf = tmpDf[tmpDf['__value__']!=0]
        #print(f"{cols}:{next-1}=>{sum(tmpDf['__value__'])}")
        dfMeltCount.append(tmpDf)

        if next == len(dfCount.columns)+1:
            break

    dfCount = pd.concat(dfMeltCount,ignore_index=True)
    dfCount = dfCount.groupby(by=['__variable__']).sum()
    dfCount.reset_index(inplace=True)
    
    dfDocWord = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix, columns=vocabularyVec)
    dfDocWord['docId'] = dfDocWord.index

    dfMeltCount = []
    for col in dfDocWord.columns[:-1]:
        tmpDf = pd.melt(dfDocWord[[col,'docId']],id_vars=['docId'],value_name='__value__',var_name='__variable__')
        tmpDf = tmpDf[tmpDf['__value__']!=0]
        tmpDf.set_index(['docId','__variable__'],drop=True,inplace=True)
        tmpDf.reset_index(inplace=True)
        dfMeltCount.append(tmpDf)

    dfDocWord = pd.concat(dfMeltCount)
    del dfMeltCount
    gc.collect()

    dfDocWord.reset_index(inplace=True,drop=True)
    dfDocWord['docId'] = dfDocWord['docId'].apply(lambda row: 'doc_'+str(row))
    dfDocWord.rename(columns={'docId':'src','__variable__':'tgt','__value__':'weight'},inplace=True)
    dfDocWord = dfDocWord.merge(labels, how='left', on='src')
    dfDocWord = dfDocWord.merge(splits, how='left', on='src')

    npAggText = dfCount['__variable__'].to_numpy(dtype=str)
    npAggCount = dfCount['__value__'].to_numpy(dtype=np.int16)

    del dfCount
    del countMatrix
    del count_vector
    del tfidf_matrix
    del tfidf_vector
    del tmpDf
    del labels
    del splits
    gc.collect()

    print("Current Time Multi-thread:", datetime.now().strftime("%H:%M:%S"))
    ####################################multi thread:############################################

    nThreads = 14
    vocabThreads = list(_split_listN(list(vocabularyVec), nThreads))

    dicOrder = {}
    prev = 0
    for idx,lista in enumerate(vocabThreads):

        dicOrder[idx] = len(lista)+prev
        prev = prev + len(lista)

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    procs = []
    for idx,vocabVec in enumerate(vocabThreads):
        proc = multiprocessing.Process(target=_find_word_ngrams, args=(vocabVec,idx,npAggText,return_dict,dicOrder[idx],))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
    
    rowIndicesList = [item for sublist in list(dict(sorted(return_dict.items())).values()) for item in sublist]

    #manager.join()
    ####################################multi thread:############################################
    print("end time Multithread =", datetime.now().strftime("%H:%M:%S"))

    del npAggText
    del vocabularyWindow
    gc.collect()

    wordWordList = []
    totalWindows = np.sum(npAggCount)
    #bottleneck:
    for idx in tqdm(itertools.product(range(len(vocabularyVec)), range(len(vocabularyVec))),total=len(vocabularyVec)**2):

        if idx[0] == idx[1]:
            wordWordList.append([vocabularyVec[idx[0]],vocabularyVec[idx[1]],1,999,'wordword'])
        elif idx[0] > idx[1]:
            pmi = _calc_pmi(idx[0],idx[1],npAggCount,rowIndicesList,totalWindows)
            if pmi > 0:
                wordWordList.append([vocabularyVec[idx[0]],vocabularyVec[idx[1]],pmi,999,'wordword'])

    dfWordWord = pd.DataFrame(wordWordList,columns=['src','tgt','weight','labels','split'])

    del wordWordList
    gc.collect

    dfGraph = pd.concat([dfWordWord,dfDocWord])
    np.testing.assert_array_equal(np.unique(dfWordWord['src']),np.unique(dfWordWord['tgt']))

    dfGraph.drop_duplicates(subset=['src','tgt'],inplace=True)

    dfGraph = dfGraph.sample(frac=1)

    dfGraph.to_csv(os.path.join(ROOT_DIR,'data',path,'interm','graph.csv'),index=False)
    
    return None