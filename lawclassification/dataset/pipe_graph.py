import pandas as pd
import os
from utils.definitions import ROOT_DIR
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import math
from typing import List
import re
from tqdm.auto import tqdm
import spacy
import gc

NLP = spacy.load('en_core_web_lg')

def _tokenizer(text:str) -> str:

    text = text[0:3000] #~512 words
    newText = []

    #slow:
    for word in NLP(text):
        if (not word.is_stop or not word.is_punct) and (word.is_alpha and word.is_ascii and not word.is_digit and word.ent_iob_ == 'O'):
            newText.append(word.lemma_.lower())

    return newText

def _load_csv(fileName: str,path: str,max_rows: int) -> pd.DataFrame:

    df = pd.read_csv(os.path.join(ROOT_DIR,'data',path,'interm',fileName,fileName+'.csv'))
    df = df.head(max_rows)

    return df

def _getExactMatch(arr:np.array,word:str) -> List:

    indices = []

    #indices2 = [idx for idx,ngrams in enumerate(arr) if re.search(r'\b' + word + r'\b', ngrams)]

    for idx,ngrams in enumerate(arr):
        if re.search(r'\b' + word + r'\b', ngrams):
            indices.append(idx)

    return indices

def _aggNpArray(arr:np.array) -> np.array:

    totalCountperWindow = arr[:,-2:].astype('int')
    aggSum = np.bincount(totalCountperWindow[:,-1],weights=totalCountperWindow[:,-2])
    uniqueId = np.unique(totalCountperWindow[:,-1])
    uniqueVocab = np.unique(arr[:,0])
    aggArr = np.concatenate((np.expand_dims(uniqueVocab,axis=0).T,
                            np.expand_dims(uniqueId,axis=0).T,
                            np.expand_dims(aggSum,axis=0).T),
                            axis=1)

    return aggArr

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

    count_vector = CountVectorizer(#tokenizer = _tokenizer,
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

    tfidf_vector = TfidfVectorizer(#tokenizer = _tokenizer,
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


    dfTrain = _load_csv('train',path,maxRows)

    countMatrix = count_vector.fit_transform(dfTrain['text'])
    tfidf_matrix = tfidf_vector.fit_transform(dfTrain['text'])
    labels = dfTrain['labels'].to_frame()
    labels['src'] = labels.index
    labels['src'] = labels['src'].apply(lambda row: 'doc_'+str(row))

    del dfTrain
    gc.collect()

    vocabularyVec = tfidf_vector.get_feature_names_out().astype('U')
    vocabularyWindow = count_vector.get_feature_names_out().astype('U')

    dfCount = pd.DataFrame.sparse.from_spmatrix(countMatrix, columns=vocabularyWindow)
    dfCount = pd.melt(dfCount,value_name='__value__',var_name='__variable__')
    dfCount = dfCount.groupby(by=['__variable__']).sum()
    dfCount.reset_index(inplace=True)
    
    dfDocWord = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix, columns=vocabularyVec)
    dfDocWord['docId'] = dfDocWord.index
    dfDocWord = pd.melt(dfDocWord,id_vars=['docId'],value_name='__value__',var_name='__variable__')
    dfDocWord.set_index(['docId','__variable__'],drop=True,inplace=True)
    dfDocWord = dfDocWord[dfDocWord['__value__']>0]
    dfDocWord.reset_index(inplace=True)
    dfDocWord['docId'] = dfDocWord['docId'].apply(lambda row: 'doc_'+str(row))
    dfDocWord.rename(columns={'docId':'src','__variable__':'tgt','__value__':'weight'},inplace=True)
    dfDocWord = dfDocWord.merge(labels, how='left', on='src')

    #structure to search for stuff from a sparse matrix
    #for gId,ngram in enumerate(vocabularyWindow):
    #    tmpGram = np.array(ngram, dtype='U')
    #    tmpGram = np.expand_dims(tmpGram,axis=0)

        #if gId > 100:
        #    break

        #for docId,doc in enumerate(count_matrix):

        #    itemIdx = np.where(doc.indices == gId)
        #    itemIdx = itemIdx[0][0] if len(itemIdx[0]) > 0 else None
        #    if itemIdx != None:
        #        count = doc.data[itemIdx]
        #        newGram = np.concatenate((tmpGram,np.array([int(docId),int(count),int(gId)])),axis=0)

                #finalGram.append(np.expand_dims(newGram,axis=0))

    npAggText = dfCount['__variable__'].to_numpy(dtype=str)
    #npAggText = npAggText[:100]
    npAggCount = dfCount['__value__'].to_numpy(dtype=np.int16)

    del dfCount
    gc.collect()

    totalWindows = np.sum(npAggCount)

    progressBar = tqdm(range(int((len(vocabularyVec)**2)/2)))

    vocLoopRow = vocabularyVec.copy()
    vocLoopCol = vocabularyVec.copy()

    print(len(vocLoopRow))
    print(len(npAggCount))
    wordWordList = []
    for wordRow in vocLoopRow:
        for wordCol in vocLoopCol:

            #indicesRow = [idx for idx,ngrams in enumerate(npAggText) if re.search(r'\b' + wordRow + r'\b', ngrams)]
            indicesRow = [1]#list(np.flatnonzero(np.core.defchararray.find(npAggText,wordRow)!=-1))
            #indicesRow = _getExactMatch(npAggText,wordRow)
            rowSumTT = np.sum(npAggCount[indicesRow])

            #indicesCol = [idx for idx,ngrams in enumerate(npAggText) if re.search(r'\b' + wordCol + r'\b', ngrams)]
            indicesCol = [1]#list(np.flatnonzero(np.core.defchararray.find(npAggText,wordCol)!=-1))
            #indicesCol = _getExactMatch(npAggText,wordCol)
            colSumTT = np.sum(npAggCount[indicesCol])

            indicesRowandCol = list(np.intersect1d(indicesRow,indicesCol))
            rowColSumTT = np.sum(npAggCount[indicesRowandCol])

            pRow = rowSumTT / totalWindows
            pCol = colSumTT / totalWindows
            pRowCol = rowColSumTT / totalWindows
            
            if pRowCol == 0 or (pRow*pCol == 0):
                pmi = 0
            else:
                pmi = math.log(pRowCol / (pRow*pCol),2)

            #try:
                #pmi = math.log(pRowCol / (pRowTpCol),2)
            #except:
                #pmi = 0
            
            if wordRow == wordCol:
                wordWordList.append([wordRow,wordCol,1])
            elif pmi > 0:
                wordWordList.append([wordRow,wordCol,pmi])

            progressBar.update(1)

        #getting only a->b , not a->b AND b->a
        vocLoopCol = vocLoopCol[vocLoopCol != wordRow]
    
    return None