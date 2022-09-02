import pandas as pd
import os
from utils.definitions import ROOT_DIR
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import math
from typing import List
import itertools
from tqdm.auto import tqdm
import spacy
import gc

def _calc_pmi(rdx:int,
              cdx:int,
              npAgg:np.array,
              rowIndicesList:List,
              totalWindows:np.int16) -> np.float16:

    if cdx == rdx:
        return 1

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

    if pmi > 0:
        return pmi
    else:
        return None

NLP = spacy.load('en_core_web_lg')

#usar a funcao de fast tokenizers to hugginface
def _tokenizer(text:str) -> str:

    text = text[0:3000] #~512 words
    newText = []

    #slow:
    for word in NLP(text):
        if (not word.is_stop or not word.is_punct) and (word.is_alpha and word.is_ascii and not word.is_digit and word.ent_iob_ == 'O'):
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

    #trainMask

    del dfTable
    gc.collect()

    vocabularyVec = tfidf_vector.get_feature_names_out().astype('U')
    vocabularyWindow = count_vector.get_feature_names_out().astype('U')

    dfCount = pd.DataFrame.sparse.from_spmatrix(countMatrix, columns=vocabularyWindow)

    dfMeltCount = []
    step = int(len(dfCount.columns) / 100)+1 #isso aqui até o limite da memoria
    for idx,cols in enumerate(range(0,len(dfCount.columns),step)):
        next = step if idx == 0 else step*(idx+1)
        next = len(dfCount.columns) if next+step-1 > len(dfCount.columns) else next
    
        tmpDf = pd.melt(dfCount.iloc[:,[cols,next-1]],value_name='__value__',var_name='__variable__')
        tmpDf['__value__'] = tmpDf['__value__'].astype(np.int16)
        tmpDf['__variable__'] = tmpDf['__variable__'].astype(str)
        tmpDf = tmpDf[tmpDf['__value__']!=0]
        dfMeltCount.append(tmpDf)

        #tem q ver se row * coluna = row do novo df

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
    gc.collect()

    progressBar = tqdm(range(len(vocabularyVec)))
    rowIndicesList = []
    for wordRow in vocabularyVec:
        #pode tentar colocar ' ' na frente e atras de todos e fazer só 1 in 
        #indicesRow = [idx for idx,ngrams in enumerate(npAggText) if re.search(r'\b' + wordRow + r'\b', ngrams)]
        #indicesRow = [idx for idx,ngrams in enumerate(npAggText) if re.match(r'\b' + wordRow + r'\b', ngrams)]
        #indicesRow = [idx for idx,ngrams in enumerate(npAggText) if ngrams.startswith(wordRow+' ') or ngrams.endswith(' '+wordRow) or ' '+wordRow+' ' in ngrams]
        indicesRow = [idx for idx,ngrams in enumerate(npAggText) if ' '+wordRow+' ' in ' '+ngrams+' ']
        rowIndicesList.append(indicesRow)
        progressBar.update(1)

    vocabIdxPermutations = []
    for r in itertools.product(range(len(vocabularyVec)), range(len(vocabularyVec))):
        if r[0] >= r[1]:
            vocabIdxPermutations.append([r[0],r[1]])

    progressBar = tqdm(range(len(vocabIdxPermutations)))

    wordWordList = []
    totalWindows = np.sum(npAggCount)
    #bottleneck:
    for idx in vocabIdxPermutations:

        pmi = _calc_pmi(idx[0],idx[1],npAggCount,rowIndicesList,totalWindows)
        
        if pmi != None:
            wordWordList.append([vocabularyVec[idx[0]],vocabularyVec[idx[1]],pmi,999,'wordword'])

        progressBar.update(1)

    dfWordWord = pd.DataFrame(wordWordList,columns=['src','tgt','weight','labels','split'])

    dfGraph = pd.concat([dfWordWord,dfDocWord])
    np.testing.assert_array_equal(np.unique(dfWordWord['src']),np.unique(dfWordWord['tgt']))

    dfGraph.drop_duplicates(subset=['src','tgt'],inplace=True)

    dfGraph = dfGraph.sample(frac=1)

    dfGraph.to_csv(os.path.join(ROOT_DIR,'data',path,'interm','graph.csv'),index=False)
    
    return None