import pandas as pd
from utils.definitions import ROOT_DIR
import os
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter
import multiprocessing
import torch_geometric.data as data
from datetime import datetime
import gc
import torch
import torch_sparse as S
from tqdm.auto import tqdm
import pickle
from psutil import Process

def _split_listN(a, n):
    '''split an array in n equal parts
    '''
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def _load_csv(path: str, maxRows: int, windowSize: int) -> pd.DataFrame:
    '''load split csv file from other the deep/xgboost pipe and get ready for the graph pipe
    '''

    NLP = spacy.load('en_core_web_lg')

    def _fix_padding(row):
        '''pad documents that have setences that are smaller than windowsSize
        '''

        if len(row.split()) < windowSize:

            #todo, fazer o multiplo, nao perder frase
            tokenizerSafeMargin = 5 #different tokenizers might count different, so we put a safe margin

            padding = ['pad__gcn']*((windowSize - len(row.split())) + tokenizerSafeMargin)
            padding = ' '.join(padding)
            row = row + ' ' + padding

        return row

    def _light_tokenizer(row):
        '''light tokenizer to not affect the learning models, but try to make the graphs smallers'''

        newText = []
        #row = row[0:4000] #restringir o tamanho do texto

        #usar o tokenizador do legal bert, deixar em paralelo:
        #criar coluna nova no pre_csv já com o texto tokenizado.

        for word in NLP(row):
            if ((not word.is_stop or not word.is_punct) and (word.is_alpha and word.is_ascii and not word.is_digit)) or (word.text == 'pad__gcn'):
            #if ((not word.is_punct) and (word.is_alpha and word.is_ascii and not word.is_digit)) or (word.text == 'pad__gcn'):
                newText.append(word.text.lower())

        return ' '.join(newText)

    dfTrain = pd.read_csv(os.path.join(ROOT_DIR,'data',path,'interm','train','train.csv'))
    dfTrain['split'] = 'train'
    dfTest = pd.read_csv(os.path.join(ROOT_DIR,'data',path,'interm','test','test.csv'))
    dfTest['split'] = 'test'
    dfVal = pd.read_csv(os.path.join(ROOT_DIR,'data',path,'interm','val','val.csv'))
    dfVal['split'] = 'val'

    df = pd.concat([dfTrain,dfTest,dfVal],ignore_index=True)

    #df['text'] = df['text'].apply(_light_tokenizer)
    df['text'] = df['text'].apply(_fix_padding)
    
    df = df.head(maxRows)
    df.reset_index(inplace=True,drop=True)

    df.to_csv(os.path.join(ROOT_DIR,'data',path,'interm','pre_graph.csv'),index=False)

    return df

def _get_worddocGraph(df:pd.DataFrame,nThreads:int):

    '''building vocab and windows (ngrams) using sklearn functions, so it's more readible
    '''

    #pode existir tokens (nodes) que so existe na aresta doc -> word, nao tem na word, word (fazer um teste)

    tfidfVector = TfidfVectorizer(ngram_range=(1,1),
                                  lowercase = True,
                                  #strip_accents = 'unicode',
                                  #encoding = 'utf-8',
                                  #decode_error = 'strict',
                                  #max_df = 1.0, #tirar 'outliers' palavras muito repetidas entre documentos
                                  #min_df = 1, #tirar 'outliers' palavras muito raras entre documentos#'cut-off'
                                  dtype = np.float32,
                                  max_features = None, #numero maximo de features
                                  vocabulary = None)

    tfidfMatrix = tfidfVector.fit_transform(df['text'])

    vocabularyVec = tfidfVector.get_feature_names_out().astype('U')

    vocabToId = {label:idx for idx, label in enumerate(vocabularyVec)}
    idToVocab = {idx:label for idx, label in enumerate(vocabularyVec)}
    docToLabel = {idx:label for idx, label in enumerate(df['labels'])}
    docToSplit = {idx:label for idx, label in enumerate(df['split'])}

    wordDocDf = pd.DataFrame([[i, j, tfidfMatrix[i,j], docToLabel[i], docToSplit[i]] for i, j in zip(*tfidfMatrix.nonzero())],
                             columns=['src','tgt','weight','label','splits'])# <-- lento

    dfTypeDic = {'src':np.int32,'tgt':np.int32,'label':np.int32,'weight':np.float32,'splits':np.str_}
    wordDocDf = wordDocDf.astype(dfTypeDic)

    wordDocDf['src'] = 'doc_'+wordDocDf['src'].astype(np.str_)
    wordDocDf['tgt'] = wordDocDf['tgt'].apply(lambda row: idToVocab[row])
    wordDocDf = wordDocDf[wordDocDf['tgt']!='pad__gcn']

    #add symmetric adj matrix
    _wordDocDf = wordDocDf.copy(deep=False)
    _wordDocDf.rename(columns={'src':'src_'},inplace=True)
    _wordDocDf.rename(columns={'tgt':'src'},inplace=True)
    _wordDocDf.rename(columns={'src_':'tgt'},inplace=True)

    wordDocDf = pd.concat([wordDocDf,_wordDocDf],ignore_index=True).drop_duplicates()

    #add selfloop
    dfSelfLoop = wordDocDf[(wordDocDf['src'].str.contains("doc_"))][['src','label','splits']].drop_duplicates()
    dfSelfLoop['weight'] = 1.0 #try 0 here
    dfSelfLoop['tgt'] = dfSelfLoop['src']

    dfSelfLoop = dfSelfLoop[['src','tgt','weight','label','splits']]

    wordDocDf = pd.concat([wordDocDf,dfSelfLoop],ignore_index=True)

    docsArr = df['text'].to_list()
    vocDocsArr = []

    for docs in tqdm(docsArr):
        docToken = docs.split()
        #docToken = [vocabToId[word] for word in docToken if word in vocabularyVec] #<--slow
        #docToken = np.array([vocabToId[word.lower()] for word in docToken if word.lower() in vocabularyVec],dtype=np.int32) #<--slow
        #docToken = np.array([vocabToId[word.lower()] for word in docToken if vocabToId.get(word.lower()) != None],dtype=np.int32) #<--slow
        docToken = [vocabToId[word.lower()] for word in docToken if vocabToId.get(word.lower()) != None] #<--slow
        #docToken = np.delete(docToken, np.where(docToken == -1))
        #docToken = np.array([vocabToId[word] for word in docToken],dtype=np.int32)
        vocDocsArr.append(docToken)

    vocDocsArr = _split_listN(vocDocsArr,nThreads)

    print('num of chars :', sum(len(s) for s in df['text']))
    print('Nvocab: ',len(vocabularyVec))
    print('numDocs: ',len(df['text']))
    
    return vocDocsArr, wordDocDf, [vocabToId,idToVocab]

def _dic_sum(dicOri,dicSum):

    '''add two dicts together, if there is a new key, "append" the new key in the original dic'''

    for words in dicSum:

        if dicOri.get(words) != None:
            dicOri[words] = 1 + dicSum[words]
        else:
            dicOri[words] = dicSum[words]

    return dicOri

def _co_corrence_build(docsArr, windowSize, processNum, returnDict):

    '''build dics of word co-ocorrence'''

    dic = {}
    windowsCount = 0

    for doc in docsArr:

        for idx in range(len(doc) - windowSize + 1):

            curWindow = doc[idx: idx + windowSize]

            windowsCount += 1
            sentence = set(curWindow) #unique, PMI being defined as occurence once in the window

            for words in sentence:

                dicCount = dict(Counter(sentence))
            
                if dic.get(words) != None:
                    dic[words] = _dic_sum(dic[words], dicCount)
                else:
                    dic[words] = dicCount

    returnDict[processNum] = (dic,windowsCount)

def _parallel_load(docsArrThread, windowSize, vocabMaps):

    '''split docArr in n threads'''


    #mudar esse negocio do manager
    manager = multiprocessing.Manager()
    returnDict = manager.dict()

    procs = []
    for idx,docs in enumerate(docsArrThread):
        proc = multiprocessing.Process(target=_co_corrence_build, args=(docs,
                                                                        windowSize,
                                                                        idx,
                                                                        returnDict,
                                                                        ))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()


    idToVocab=vocabMaps[1]
    windowQty = 0

    edgeList=[]
    for threads in returnDict.items():
        dic = threads[1][0]
        windowQty = windowQty + threads[1][1]
        for sdx, subDic in dic.items():
            for rdx, qty in subDic.items():
                edgeList.append([sdx,rdx,qty])

    manager.shutdown()

    print('Nwindows: ',windowQty)

    edgeDf = pd.DataFrame(edgeList,columns=['src','tgt','qty'])

    del edgeList
    gc.collect()

    edgeDf = edgeDf.groupby(['src','tgt']).sum().reset_index()

    singleOccrDf = edgeDf[(edgeDf['src']==edgeDf['tgt'])][['src','qty']]
    singleOccrDf = singleOccrDf.groupby(['src']).sum().reset_index()
    singleOccrDf.rename(columns={'qty':'qtyTTsrc'},inplace=True)

    edgeDf = edgeDf.merge(singleOccrDf,how='inner',on='src')
    singleOccrDf.rename(columns={'src':'tgt'},inplace=True)
    singleOccrDf.rename(columns={'qtyTTsrc':'qtyTTtgt'},inplace=True)
    edgeDf = edgeDf.merge(singleOccrDf,how='inner',on='tgt')

    edgeDf['weight'] = np.log(edgeDf['qty']/windowQty / (edgeDf['qtyTTsrc']/windowQty*edgeDf['qtyTTtgt']/windowQty)) ##pq os weights aumentaram? e estao sem uma boa distribuição?
    edgeDf.loc[edgeDf['src']==edgeDf['tgt'], 'weight'] = 1.0
    edgeDf = edgeDf[edgeDf['weight']>0]
    edgeDf = edgeDf[(edgeDf['src']!='pad__gcn')]
    edgeDf = edgeDf[(edgeDf['tgt']!='pad__gcn')]

    edgeDf = edgeDf[['src','tgt','weight']]

    #add adj symmetric:
    _edgeDf = edgeDf.copy(deep=False)
    _edgeDf.rename(columns={'src':'src_'},inplace=True)
    _edgeDf.rename(columns={'tgt':'src'},inplace=True)
    _edgeDf.rename(columns={'src_':'tgt'},inplace=True)

    edgeDf = pd.concat([edgeDf,_edgeDf],ignore_index=True).drop_duplicates()

    edgeDf['label'] = 999
    edgeDf['splits'] = 'wordword'
    edgeDf['src'] = edgeDf['src'].apply(lambda row: idToVocab[row])
    edgeDf['tgt'] = edgeDf['tgt'].apply(lambda row: idToVocab[row])

    return edgeDf

def _create_edge_df(path: str, maxRows: int, windowSize: int, nThreads:int) -> pd.DataFrame:

    if not os.path.exists(os.path.join(ROOT_DIR,'data',path,'interm','pre_graph.csv')):
        df = _load_csv(path,maxRows,windowSize)
    else:
        df = pd.read_csv(os.path.join(ROOT_DIR,'data',path,'interm','pre_graph.csv'))

    docsArr, wordDocDf, vocabMaps = _get_worddocGraph(df,nThreads)

    print("Begin parallel =", datetime.now().strftime("%H:%M:%S"))
    edgeDf = _parallel_load(docsArr,windowSize, vocabMaps)
    print("end parallel =", datetime.now().strftime("%H:%M:%S"))

    graphDf = pd.concat([edgeDf,wordDocDf],ignore_index=True)

    return graphDf

def _create_pygraph_data(graphDf:pd.DataFrame) -> None:

    graphDf = graphDf.sample(frac=1)
    graphDf.dropna(inplace=True) #pode ter algum termo como #N/A como token?

    _graphDf = graphDf.copy(deep=True) #precisa mesmo?
    _graphDf = _graphDf[~((_graphDf['tgt'].str.contains("doc_")))][['src','label','splits']].drop_duplicates() #bad

    allNodesSrc = np.array(graphDf['src'],dtype=np.str_)
    allNodesTgt = np.array(graphDf['tgt'],dtype=np.str_)

    np.testing.assert_array_equal(np.array(graphDf['src'].sort_values()),np.array(graphDf['tgt'].sort_values()))

    wgtEdges = np.array(graphDf['weight'])

    labels = np.array(graphDf['label'])

    allUniqueNodes = np.unique(allNodesSrc) #AllNodesSrc e Tgt uniques are equal. undirected graph
    allUniqueLabels = np.unique(labels)

    label2idNodes = {label:id for id,label in enumerate(allUniqueNodes.tolist())}
    allUniqueNodesId = np.array([label2idNodes[idx] for idx in allUniqueNodes.tolist()])

    label2id = {label:id for id,label in enumerate(allUniqueLabels.tolist())}

    _graphDf['nodeId'] = _graphDf['src'].apply(lambda row: label2idNodes[row])
    _graphDf['label'] = _graphDf['label'].apply(lambda row: label2id[row])
    _graphDf = _graphDf[['nodeId','label','splits']].drop_duplicates()
    _graphDf['train_mask'] = _graphDf['splits'].apply(lambda row: True if row=='train' else False)
    _graphDf['test_mask'] = _graphDf['splits'].apply(lambda row: True if row=='test' else False)
    _graphDf['val_mask'] = _graphDf['splits'].apply(lambda row: True if row=='val' else False)

    numClasses = len(allUniqueLabels) - 1

    trainMask = _graphDf['train_mask'].to_numpy(dtype=bool)
    valMask = _graphDf['val_mask'].to_numpy(dtype=bool)
    testMask = _graphDf['test_mask'].to_numpy(dtype=bool)
    labelToNodes = _graphDf['label'].to_numpy(dtype=int)
    allUniqueNodesId = _graphDf['nodeId'].to_numpy(dtype=int)

    arr1inds = allUniqueNodesId.argsort()[::-1][:len(allUniqueNodesId)]
    trainMask = trainMask[arr1inds[::-1]]
    valMask = valMask[arr1inds[::-1]]
    testMask = testMask[arr1inds[::-1]]
    labelToNodes = labelToNodes[arr1inds[::-1]]
    allUniqueNodesId = allUniqueNodesId[arr1inds[::-1]]

    del graphDf
    del _graphDf
    gc.collect()

    srcNodesId = np.array([label2idNodes[idx] for idx in allNodesSrc.tolist()])
    tgtNodesId = np.array([label2idNodes[idx] for idx in allNodesTgt.tolist()])
    edgeIndex = np.concatenate((np.expand_dims(srcNodesId,0),np.expand_dims(tgtNodesId,0)),axis=0)

    edgeAttr = np.expand_dims(wgtEdges,0).T

    oneHotMtx = S.SparseTensor.eye(M=len(allUniqueNodesId),dtype=torch.float32)
    #oneHotMtx = oneHotMtx.to_dense()
    oneHotMtx = oneHotMtx.to_torch_sparse_coo_tensor()

    edgeIndex = torch.tensor(edgeIndex,dtype=torch.long)
    wgtEdges = torch.tensor(wgtEdges,dtype=torch.float32)
    labels = torch.tensor(labelToNodes,dtype=torch.long)
    trainMask = torch.tensor(trainMask,dtype=torch.bool)
    valMask = torch.tensor(valMask,dtype=torch.bool)
    testMask = torch.tensor(testMask,dtype=torch.bool)

    return data.Data(x = oneHotMtx,
                     edge_index = edgeIndex,
                     edge_weight = wgtEdges,
                     #edge_attr = edgeAttr,
                     y = labels,
                     num_classes = numClasses,
                     test_mask = testMask,
                     train_mask = trainMask,
                     val_mask = valMask)

def fast_pipe_graph(path: str, maxRows: int, windowSize: int, nThreads:int) -> None:

    print('Dataset: ',path)
    print("Creating graph dataset... this might take a while", datetime.now().strftime("%H:%M:%S"))

    graphDf = _create_edge_df(path, maxRows, windowSize, nThreads)

    pyData = _create_pygraph_data(graphDf)

    with open(os.path.join(ROOT_DIR,'data',path,'pygraph.pickle'), 'wb') as f:
        pickle.dump(pyData, f)

    return None