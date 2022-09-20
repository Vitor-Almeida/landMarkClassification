import pandas as pd
from utils.definitions import ROOT_DIR
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter
import multiprocessing
import torch_geometric.data as data
from datetime import datetime
import gc
import torch
import torch_sparse as S
import pickle
from tokenizers import Tokenizer
from tokenizers import pre_tokenizers
from tokenizers import normalizers
from tokenizers.models import WordPiece
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer

def _split_listN(a, n):
    '''split an array in n equal parts
    '''
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def _tokenizer(df:pd.DataFrame, vocab_size:int):

    bertTokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    bertTokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])
    bertTokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Punctuation('removed'),pre_tokenizers.Whitespace()])

    bertTokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )

    trainer = WordPieceTrainer(
        vocab_size = vocab_size, 
        #special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        special_tokens=[],
        min_frequency = 0, 
        show_progress = True, 
        initial_alphabet  = [],
        #continuing_subword_prefix = '##'
        continuing_subword_prefix = ''
    )

    bertTokenizer.train_from_iterator(df['text'], trainer=trainer)

    vocabToId = bertTokenizer.get_vocab()
    vocabVec = list(bertTokenizer.get_vocab().keys())
    idToVocab = {idx:label for label,idx in vocabToId.items()}

    encodedText = df['text'].apply(lambda row: bertTokenizer.encode(row,add_special_tokens=False).ids)

    return encodedText, [vocabVec,vocabToId,idToVocab]


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

    df['text_token'],vocabMaps = _tokenizer(df,vocab_size=30522)
    
    df = df.head(maxRows)
    df.reset_index(inplace=True,drop=True)

    df.to_csv(os.path.join(ROOT_DIR,'data',path,'interm','pre_graph.csv'),index=False)

    return df, vocabMaps

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

    #vocabularyVec = tfidfVector.get_feature_names_out().astype('U')

    tfidfColToVocab = {pair[1]:pair[0] for pair in tfidfVector.vocabulary_.items()}

    #vocabToId = {label:idx for idx, label in enumerate(vocabularyVec)}
    #idToVocab = {idx:label for idx, label in enumerate(vocabularyVec)}
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
            dicOri[words] = 1 + dicSum[words]
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


    idToVocab=vocabMaps[2]
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
    
    edgeDf = edgeDf[['src','tgt','weight']]

    #add adj symmetric: já é simetrico.
    #_edgeDf = edgeDf.copy(deep=True)
    #_edgeDf.rename(columns={'src':'src_'},inplace=True)
    #_edgeDf.rename(columns={'tgt':'src'},inplace=True)
    #_edgeDf.rename(columns={'src_':'tgt'},inplace=True)

    #edgeDf = pd.concat([edgeDf,_edgeDf],ignore_index=True).drop_duplicates()

    edgeDf['label'] = 999
    edgeDf['splits'] = 'wordword'
    edgeDf['src'] = edgeDf['src'].apply(lambda row: idToVocab[row])
    edgeDf['tgt'] = edgeDf['tgt'].apply(lambda row: idToVocab[row])

    return edgeDf

def _create_edge_df(path: str, maxRows: int, windowSize: int, nThreads:int) -> pd.DataFrame:

    #if not os.path.exists(os.path.join(ROOT_DIR,'data',path,'interm','pre_graph.csv')):
    df, vocabMaps = _load_csv(path,maxRows)
    #else:
    #    df = pd.read_csv(os.path.join(ROOT_DIR,'data',path,'interm','pre_graph.csv'))
        #FAZER
        #vocabVec = 

    docsArr, wordDocDf = _get_worddocGraph(df,nThreads,vocabMaps)

    edgeDf = _parallel_load(docsArr,windowSize, vocabMaps)

    graphDf = pd.concat([edgeDf,wordDocDf],ignore_index=True)

    #todo:
    #tirando palavras do TFIDF (doc-word), pois não existem no word->word, por causa de um possivel pmi = 0 ou pmi < 0 (olhar.)

    return graphDf

def _create_pygraph_data(graphDf:pd.DataFrame) -> None:

    graphDf = graphDf.sample(frac=1)
    graphDf.dropna(inplace=True) #pode ter algum termo como #N/A como token?

    #removing selfloops and symmetry
    _graphDf = graphDf[(graphDf['src']!=graphDf['tgt'])]
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

    #edgeAttr = np.expand_dims(wgtEdges,0).T

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