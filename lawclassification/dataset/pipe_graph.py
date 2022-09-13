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

def _cross_words(vocabularyVec,npAggCount,rowIndicesList,totalWindows,idx,dicVocabTT,vocabTT,return_dict):

    wordWordList = []

    for wordTuple in itertools.product(vocabularyVec, vocabTT):

        rdx = dicVocabTT[wordTuple[0]]
        cdx = dicVocabTT[wordTuple[1]]

        if rdx == cdx:
            wordWordList.append([wordTuple[0],wordTuple[1],1,999,'wordword'])
        elif rdx > cdx:
            pmi = _calc_pmi(rdx,cdx,npAggCount,rowIndicesList,totalWindows)
            if pmi > 0:
                wordWordList.append([wordTuple[0],wordTuple[1],pmi,999,'wordword'])

    return_dict[idx]=wordWordList

def _find_word_ngrams(vocabularyVec,npAggText,return_dict,lastOrder):

    rowIndicesList = []
    for wordRow in vocabularyVec:
        indicesRow = [idx for idx,ngrams in enumerate(npAggText) if ' '+wordRow+' ' in ' '+ngrams+' ']
        rowIndicesList.append(indicesRow)

    return_dict[lastOrder]=rowIndicesList

def _remove_non_pairs(vocabularyVec,rowIndicesList,dicVocabTT):

    flag_remove = False

    for n in rowIndicesList:
        for v in rowIndicesList:
            if np.intersect1d(n,v)!=None:
                flag_remove = True
                break

    return None

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
    
    #if pRowCol == 0 or (pRow*pCol == 0):
    if pRowCol == 0:
        #avoid math errors
        pmi = 0
    else:
        #pmi = math.log(pRowCol / (pRow*pCol),2)
        pmi = math.log(pRowCol / (pRow*pCol) )

    return pmi

NLP = spacy.load('en_core_web_lg')

#usar a funcao de fast tokenizers do hugginface
def _tokenizer(text:str) -> str:

    #be careful to not remove the special padding with the tokenzier 'pad__gcn'

    #text = text[0:500] #~512 words
    newText = []

    #slow:
    for word in NLP(text):
        if ((not word.is_stop or not word.is_punct) and (word.is_alpha and word.is_ascii and not word.is_digit)) or (word.text == 'pad__gcn'):
            newText.append(word.lemma_.lower())

    return newText

def _load_csv(path: str,max_rows: int, windowSize: int) -> pd.DataFrame:

    dfTrain = pd.read_csv(os.path.join(ROOT_DIR,'data',path,'interm','train','train.csv'))
    dfTrain['split'] = 'train'
    dfTest = pd.read_csv(os.path.join(ROOT_DIR,'data',path,'interm','test','test.csv'))
    dfTest['split'] = 'test'
    dfVal = pd.read_csv(os.path.join(ROOT_DIR,'data',path,'interm','val','val.csv'))
    dfVal['split'] = 'val'

    df = pd.concat([dfTrain,dfTest,dfVal],ignore_index=True)

    def _fix_padding(row):

        if len(row.split()) < windowSize:

            tokenizerSafeMargin = 5 #different tokenizers might count different, so we put a safe margin

            padding = ['pad__gcn']*((windowSize - len(row.split())) + tokenizerSafeMargin)
            padding = ' '.join(padding)
            row = row + ' ' + padding

        return row

    df['text'] = df['text'].apply(_fix_padding)
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


    dfTable = _load_csv(path,maxRows,windowSize)

    countMatrix = count_vector.fit_transform(dfTable['text'])
    tfidfMatrix = tfidf_vector.fit_transform(dfTable['text'])
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

    sum_words = countMatrix.sum(axis = 0) 
    words_freq = [[word, sum_words[0, i]] for word, i in count_vector.vocabulary_.items()]

    dfCount = pd.DataFrame(words_freq,columns=['__variable__','__value__'])
    npAggText = dfCount['__variable__'].to_numpy(dtype=str)
    npAggCount = dfCount['__value__'].to_numpy(dtype=int)
    
    idToVocab = {label:idx for idx, label in tfidf_vector.vocabulary_.items()}

    words_freq = [['doc_'+str(i), idToVocab[j], tfidfMatrix[i,j]] for i, j in zip(*tfidfMatrix.nonzero())]
    dfDocWord = pd.DataFrame(words_freq,columns=['src','tgt','weight'])
    dfDocWord = dfDocWord.merge(labels, how='left', on='src')
    dfDocWord = dfDocWord.merge(splits, how='left', on='src')

    del dfCount
    del countMatrix
    del count_vector
    del tfidf_vector
    del labels
    del splits
    gc.collect()

    print("Current Time Multi-thread:", datetime.now().strftime("%H:%M:%S"))
    #################################### multi thread: ###############################################

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
        proc = multiprocessing.Process(target=_find_word_ngrams, args=(vocabVec,npAggText,return_dict,dicOrder[idx],))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
    
    rowIndicesList = [item for sublist in list(dict(sorted(return_dict.items())).values()) for item in sublist]

    manager.shutdown()
    ###################################################################################################
    print("end time Multithread =", datetime.now().strftime("%H:%M:%S"))

    del npAggText
    del vocabularyWindow
    gc.collect()

    #################################### multi thread: #############################################
    print("Current Time Multi-thread2:", datetime.now().strftime("%H:%M:%S"))
    totalWindows = np.sum(npAggCount)

    dicVocabTT = {word:idx for idx,word in enumerate(vocabularyVec)}

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    procs = []
    for idx,vocabVec in enumerate(vocabThreads):
        proc = multiprocessing.Process(target=_cross_words, args=(vocabVec,
                                                                  npAggCount,
                                                                  rowIndicesList,
                                                                  totalWindows,
                                                                  idx,
                                                                  dicVocabTT,
                                                                  vocabularyVec,
                                                                  return_dict,
                                                                  ))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

    wordWordList = [item for sublist in list(dict(sorted(return_dict.items())).values()) for item in sublist]

    manager.shutdown()

    print("Current Time Multi-thread2:", datetime.now().strftime("%H:%M:%S"))
    ###############################################################################################

    dfWordWord = pd.DataFrame(wordWordList,columns=['src','tgt','weight','labels','split'])

    del wordWordList
    gc.collect

    dfGraph = pd.concat([dfWordWord,dfDocWord],ignore_index=True)
    np.testing.assert_array_equal(np.unique(dfWordWord['src']),np.unique(dfWordWord['tgt']))

    dfGraph.drop_duplicates(subset=['src','tgt'],inplace=True)

    dfGraph = dfGraph.sample(frac=1)

    dfGraph = dfGraph[(dfGraph['src']!='pad__gcn') & (dfGraph['tgt']!='pad__gcn')]

    dfGraph.to_csv(os.path.join(ROOT_DIR,'data',path,'interm','graph.csv'),index=False)
    
    return None