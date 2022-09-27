import pandas as pd
from utils.definitions import ROOT_DIR
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter
import multiprocessing
import torch_geometric.data as data
from torch_geometric import utils as U
from datetime import datetime
import gc
import torch
import torch_sparse as S
import pickle
from utils.helper_funs import hug_tokenizer
import pickle

def _split_listN(a, n):
    '''split an array in n equal parts
    '''
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def _tokenizer(df:pd.DataFrame, vocab_size:int):

    bertTokenizer , trainer = hug_tokenizer(vocab_size)

    bertTokenizer.train_from_iterator(df['text'], trainer=trainer)

    vocabToId = bertTokenizer.get_vocab()
    vocabVec = list(bertTokenizer.get_vocab().keys())
    idToVocab = {idx:label for label,idx in vocabToId.items()}

    encodedText = df['text'].apply(lambda row: bertTokenizer.encode(row,add_special_tokens=False).ids)

    vocabMaps = [vocabVec,vocabToId,idToVocab]

    return encodedText, vocabMaps

def _load_csv(path: str, maxRows: int) -> pd.DataFrame:
    '''load split csv file from other the deep/xgboost pipe and get ready for the graph pipe
    '''

    dfTrain = pd.read_csv(os.path.join(ROOT_DIR,'data',path,'interm','train','train.csv'))
    dfTrain['split'] = 'train'
    dfTest = pd.read_csv(os.path.join(ROOT_DIR,'data',path,'interm','test','test.csv'))
    dfTest['split'] = 'test'
    dfVal = pd.read_csv(os.path.join(ROOT_DIR,'data',path,'interm','val','val.csv'))
    dfVal['split'] = 'val'

    df = pd.concat([dfTrain,dfTest,dfVal],ignore_index=True)

    numChar = sum(len(s) for s in df['text'])

    #eurlex_lexbench , scotus_lexbench e o tj
    if path == 'tj':
        df['text'] = df['text'].apply(lambda row: row[:int(len(row)/3)])
    elif numChar > 300_000_000: #limite da memoria
        df['text'] = df['text'].apply(lambda row: row[:int(len(row)/2)])

    df['text_token'],vocabMaps = _tokenizer(df,vocab_size=300000)
    
    df = df.head(maxRows)
    df.reset_index(inplace=True,drop=True)

    multi2label = []

    try:
        #check for multilabel data
        #df['labels'].apply(eval)
        numClasses = len(eval(df['labels'][0]))
        labelListList = df['labels'].to_list()
        #uniqueLabel = list(k for k,_ in itertools.groupby(labelListList))
        uniqueLabel = list(set(labelListList))
        multiIdToLabel = {idx:label for idx,label in enumerate(uniqueLabel)}
        #pdJoin = pd.DataFrame.from_dict(multiIdToLabel,orient='tight')
        multiLabelToId = {label:idx for idx,label in enumerate(uniqueLabel)}
        df['labels'] = df['labels'].apply(lambda row: multiLabelToId[row])
        multi2label = [multiIdToLabel,multiLabelToId,numClasses]
        with open(os.path.join(ROOT_DIR,'data',path,'interm','multi2label.pickle'), 'wb') as f:
            pickle.dump(multi2label, f)
    except:
        pass

    df.to_csv(os.path.join(ROOT_DIR,'data',path,'interm','pre_graph.csv'),index=False)

    with open(os.path.join(ROOT_DIR,'data',path,'interm','vocabMaps.pickle'), 'wb') as f:
        pickle.dump(vocabMaps, f)

    return df, vocabMaps, multi2label

def _get_worddocGraph(df:pd.DataFrame,nThreads:int, vocabMaps):

    '''building vocab and windows (ngrams) using sklearn functions, so it's more readible
    '''

    vocabularyVec = vocabMaps[0]
    vocabToId = vocabMaps[1]
    idToVocab = vocabMaps[2]

    def dummytfidf(doc):
        return doc

    #pode existir tokens (nodes) que so existe na aresta doc -> word, nao tem na word, word (fazer um teste)
    tfidfVector = TfidfVectorizer(#ngram_range=(1,1),
                                  dtype = np.float32,
                                  preprocessor=dummytfidf,
                                  tokenizer=dummytfidf,
                                  token_pattern=None)

    tfidfMatrix = tfidfVector.fit_transform(df['text_token'])

    tfidfColToVocab = {pair[1]:pair[0] for pair in tfidfVector.vocabulary_.items()}

    docToLabel = {idx:label for idx, label in enumerate(df['labels'])}
    docToSplit = {idx:label for idx, label in enumerate(df['split'])}

    wordDocDf = pd.DataFrame([[i, j, tfidfMatrix[i,j], docToLabel[i], docToSplit[i]] for i, j in zip(*tfidfMatrix.nonzero())],
                             columns=['src','tgt','weight','label','splits'])# <-- lento

    dfTypeDic = {'src':np.int32,'tgt':np.int32,'label':np.int32,'weight':np.float32,'splits':np.str_}
    wordDocDf = wordDocDf.astype(dfTypeDic)

    wordDocDf['src'] = 'doc_'+wordDocDf['src'].astype(np.str_)
    wordDocDf['tgt'] = wordDocDf['tgt'].apply(lambda row: idToVocab[tfidfColToVocab[row]])

    #add symmetric adj matrix
    _wordDocDf = wordDocDf.copy(deep=False)
    _wordDocDf.rename(columns={'src':'src_'},inplace=True)
    _wordDocDf.rename(columns={'tgt':'src'},inplace=True)
    _wordDocDf.rename(columns={'src_':'tgt'},inplace=True)

    wordDocDf = pd.concat([wordDocDf,_wordDocDf],ignore_index=True)

    #add selfloop
    dfSelfLoop = wordDocDf[(wordDocDf['src'].str.contains("doc_"))][['src','label','splits']].drop_duplicates()
    dfSelfLoop['weight'] = 1.0 #try 0 here
    dfSelfLoop['tgt'] = dfSelfLoop['src']

    dfSelfLoop = dfSelfLoop[['src','tgt','weight','label','splits']]

    wordDocDf = pd.concat([wordDocDf,dfSelfLoop],ignore_index=True)

    docsArr = df['text_token'].to_list()

    vocDocsArr = _split_listN(docsArr,nThreads)

    print('num of chars :', sum(len(s) for s in df['text']))
    print('Nvocab: ',len(vocabularyVec))
    print('numDocs: ',len(df['text']))
    
    return vocDocsArr, wordDocDf

def _dic_sum(dicOri,dicSum):

    '''add two dicts together, if there is a new key, "append" the new key in the original dic'''

    for words in dicSum:

        if dicOri.get(words) != None:
            #dicOri[words] = 1 + dicSum[words]
            dicOri[words] = dicOri[words] + dicSum[words]
        else:
            dicOri[words] = dicSum[words]

    return dicOri

def _co_corrence_build(docsArr, windowSize, processNum, returnDict):

    '''build dics of word co-ocorrence'''

    dic = {}
    windowsCount = 0

    for doc in docsArr:

        if len(doc) < windowSize:
            wRange = range(0,1)
        else:
            wRange = range(len(doc) - windowSize + 1)

        for idx in wRange:

            curWindow = doc[idx: idx + windowSize]

            windowsCount += 1
            #sentence = set(curWindow)
            sentence = curWindow

            for words in sentence:

                dicCount = dict(Counter(sentence))
            
                if dic.get(words) != None:
                    dic[words] = _dic_sum(dic[words], dicCount)
                else:
                    dic[words] = dicCount

    returnDict[processNum] = (dic,windowsCount)

def _parallel_load(docsArrThread, windowSize, vocabMaps):

    '''split docArr in n threads'''

    #windows size os cara tao colocando = tamanho da sentença.

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

    windowQty = 0

    edgeList=[]
    for threads in returnDict.items():
        dic = threads[1][0]
        windowQty = windowQty + threads[1][1]
        for sdx, subDic in dic.items():
            for rdx, qty in subDic.items():
                edgeList.append([sdx,rdx,qty])
                

    manager.shutdown()

    edgeDf = pd.DataFrame(edgeList,columns=['src','tgt','qty'])

    del edgeList
    gc.collect()

    print('Nwindows: ',windowQty)

    edgeDf = edgeDf.groupby(['src','tgt']).sum().reset_index()

    singleOccrDf = edgeDf[(edgeDf['src']==edgeDf['tgt'])][['src','qty']]
    singleOccrDf = singleOccrDf.groupby(['src']).sum().reset_index()
    singleOccrDf.rename(columns={'qty':'qtyTTsrc'},inplace=True)

    edgeDf = edgeDf.merge(singleOccrDf,how='inner',on='src')
    singleOccrDf.rename(columns={'src':'tgt'},inplace=True)
    singleOccrDf.rename(columns={'qtyTTsrc':'qtyTTtgt'},inplace=True)
    edgeDf = edgeDf.merge(singleOccrDf,how='inner',on='tgt')
    
    normPmiFac = -1/np.log2(edgeDf['qty']/windowQty)

    edgeDf['weight'] = normPmiFac * np.log2((edgeDf['qty']/windowQty) / ((edgeDf['qtyTTsrc']/windowQty)*(edgeDf['qtyTTtgt']/windowQty)))
    edgeDf = edgeDf[edgeDf['weight']>0.2]
    #edgeDf = edgeDf[edgeDf['weight']>0]
    edgeDf.loc[edgeDf['src']==edgeDf['tgt'], 'weight'] = 1.0
    
     #isso tira a simetria da adj?
    
    edgeDf = edgeDf[['src','tgt','weight']]

    #add adj symmetric: já é simetrico.
    #_edgeDf = edgeDf.copy(deep=True)
    #_edgeDf.rename(columns={'src':'src_'},inplace=True)
    #_edgeDf.rename(columns={'tgt':'src'},inplace=True)
    #_edgeDf.rename(columns={'src_':'tgt'},inplace=True)

    #edgeDf = pd.concat([edgeDf,_edgeDf],ignore_index=True).drop_duplicates()

    idToVocab=vocabMaps[2]
    edgeDf['label'] = 999999
    edgeDf['splits'] = 'wordword'
    edgeDf['src'] = edgeDf['src'].apply(lambda row: idToVocab[row])
    edgeDf['tgt'] = edgeDf['tgt'].apply(lambda row: idToVocab[row])

    return edgeDf

def _create_edge_df(path: str, maxRows: int, windowSize: int, nThreads:int) -> pd.DataFrame:

    multi2label = []

    if not os.path.exists(os.path.join(ROOT_DIR,'data',path,'interm','pre_graph.csv')):

        df, vocabMaps, multi2label = _load_csv(path,maxRows)

    else:

        df = pd.read_csv(os.path.join(ROOT_DIR,'data',path,'interm','pre_graph.csv'))
        df['text_token']=df['text_token'].apply(eval)
        f = open(os.path.join(ROOT_DIR,'data',path,'interm','vocabMaps.pickle'),'rb')
        vocabMaps = pickle.load(f)
        f.close()

        if os.path.exists(os.path.join(ROOT_DIR,'data',path,'interm','multi2label.pickle')):

            f = open(os.path.join(ROOT_DIR,'data',path,'interm','multi2label.pickle'),'rb')
            multi2label = pickle.load(f)
            f.close()


    docsArr, wordDocDf = _get_worddocGraph(df,nThreads,vocabMaps)

    edgeDf = _parallel_load(docsArr,windowSize, vocabMaps)

    graphDf = pd.concat([edgeDf,wordDocDf],ignore_index=True)

    #em algumas bases sim, em outras nao: (?)
    #graphDf['weight'] = preprocessing.RobustScaler().fit_transform(graphDf[['weight']])

    #mini = min(preprocessing.RobustScaler().fit_transform(graphDf[['weight']]))
    #if mini >= 0:
    #    mini = 0
    #graphDf['weight'] = preprocessing.RobustScaler().fit_transform(graphDf[['weight']]) - mini

    #import matplotlib.pyplot as plt
    #plt.hist(np.array(graphDf['weight']), 100, density=True)
    #plt.savefig("matplotlib.png")
    #plt.close()

    #todo:
    #tirando palavras do TFIDF (doc-word), pois não existem no word->word, por causa de um possivel pmi = 0 ou pmi < 0.2 (olhar.)

    return graphDf, multi2label

def _create_pygraph_data(graphDf:pd.DataFrame,multi2label) -> None:

    #graphDf = graphDf.sample(frac=1)
    graphDf.dropna(inplace=True) #pode ter algum termo como #N/A como token?

    #removing selfloops and symmetry
    #_graphDf = graphDf[(graphDf['src']!=graphDf['tgt'])]
    _graphDf = graphDf[~((graphDf['tgt'].str.contains("doc_")))][['src','label','splits']].drop_duplicates() #<--isso aqui só vai funcionar se tiver simetrico

    allNodesSrc = np.array(graphDf['src'],dtype=np.str_)
    allNodesTgt = np.array(graphDf['tgt'],dtype=np.str_)

    np.testing.assert_array_equal(np.array(graphDf['src'].sort_values()),np.array(graphDf['tgt'].sort_values()))

    wgtEdges = np.array(graphDf['weight'])

    labels = np.array(graphDf['label'])

    allUniqueNodes = np.unique(allNodesSrc) #AllNodesSrc e Tgt uniques are equal. undirected graph

    label2idNodes = {label:id for id,label in enumerate(allUniqueNodes.tolist())}
    allUniqueNodesId = np.array([label2idNodes[idx] for idx in allUniqueNodes.tolist()])

    _graphDf['docIndex'] = _graphDf['src'].apply(lambda row: int(row.replace("doc_","")) if "doc_" in row else 999999)
    _graphDf['nodeId'] = _graphDf['src'].apply(lambda row: label2idNodes[row])
    _graphDf = _graphDf[['nodeId','label','splits','docIndex']].drop_duplicates()
    _graphDf['train_mask'] = _graphDf['splits'].apply(lambda row: True if row=='train' else False)
    _graphDf['test_mask'] = _graphDf['splits'].apply(lambda row: True if row=='test' else False)
    _graphDf['val_mask'] = _graphDf['splits'].apply(lambda row: True if row=='val' else False)

    indexMask = _graphDf['docIndex'].to_numpy(dtype=int)
    trainMask = _graphDf['train_mask'].to_numpy(dtype=bool)
    valMask = _graphDf['val_mask'].to_numpy(dtype=bool)
    testMask = _graphDf['test_mask'].to_numpy(dtype=bool)
    labelToNodes = _graphDf['label'].to_numpy(dtype=int)
    allUniqueNodesId = _graphDf['nodeId'].to_numpy(dtype=int)

    arr1inds = allUniqueNodesId.argsort()[::-1][:len(allUniqueNodesId)]
    labelToNodes = labelToNodes[arr1inds[::-1]]
    trainMask = trainMask[arr1inds[::-1]]
    valMask = valMask[arr1inds[::-1]]
    testMask = testMask[arr1inds[::-1]]
    indexMask = indexMask[arr1inds[::-1]]
    
    allUniqueNodesId = allUniqueNodesId[arr1inds[::-1]]

    if len(multi2label)>0:
        numClasses = multi2label[2]
        homoLabels = torch.tensor(labelToNodes,dtype=torch.int32)
        labelToNodes = np.array([eval(multi2label[0].get(n,str([0.0]*multi2label[2]))) for n in labelToNodes])
        labels = torch.tensor(labelToNodes,dtype=torch.float32)
    else:
        numClasses = len(np.unique(labels)) - 1
        labels = torch.tensor(labelToNodes,dtype=torch.long)
        homoLabels = torch.tensor(labelToNodes,dtype=torch.int32)

    del graphDf
    del _graphDf
    gc.collect()

    srcNodesId = np.array([label2idNodes[idx] for idx in allNodesSrc.tolist()])
    tgtNodesId = np.array([label2idNodes[idx] for idx in allNodesTgt.tolist()])
    edgeIndex = np.concatenate((np.expand_dims(srcNodesId,0),np.expand_dims(tgtNodesId,0)),axis=0)

    #edgeAttr = np.expand_dims(wgtEdges,0).T

    oneHotMtx = S.SparseTensor.eye(M=len(allUniqueNodesId),dtype=torch.float32)
    #oneHotMtx = oneHotMtx.to_dense()
    oneHotMtx = oneHotMtx.to_torch_sparse_coo_tensor()

    indexMask = torch.tensor(indexMask,dtype=torch.int32)
    edgeIndex = torch.tensor(edgeIndex,dtype=torch.long)
    wgtEdges = torch.tensor(wgtEdges,dtype=torch.float32)
    trainMask = torch.tensor(trainMask,dtype=torch.bool)
    valMask = torch.tensor(valMask,dtype=torch.bool)
    testMask = torch.tensor(testMask,dtype=torch.bool)

    homophily = round(U.homophily(edgeIndex,homoLabels),4)

    return data.Data(x = oneHotMtx,
                     edge_index = edgeIndex,
                     edge_weight = wgtEdges,
                     indexMask = indexMask,
                     homoLabels = homoLabels,
                     #edge_attr = edgeAttr,
                     homophily = homophily,
                     y = labels,
                     num_classes = numClasses,
                     test_mask = testMask,
                     train_mask = trainMask,
                     val_mask = valMask)

def fast_pipe_graph(path: str, maxRows: int, windowSize: int, nThreads:int) -> None:

    print('Dataset: ',path)
    print("Creating graph dataset... this might take a while", datetime.now().strftime("%H:%M:%S"))

    graphDf, multi2label = _create_edge_df(path, maxRows, windowSize, nThreads)

    pyData = _create_pygraph_data(graphDf, multi2label)

    with open(os.path.join(ROOT_DIR,'data',path,'pygraph.pickle'), 'wb') as f:
        pickle.dump(pyData, f)

    return None