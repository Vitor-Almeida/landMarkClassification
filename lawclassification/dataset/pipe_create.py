import pandas as pd
from transformers import pipeline
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
import os
import json
import gc
import numpy as np
from sklearn.model_selection import train_test_split
from utils.definitions import ROOT_DIR
import re

def _deparaCodigos(df):

    deparaDf = pd.read_csv(os.path.join(ROOT_DIR,'data','big_tj','raw','deparaCodigos.csv'),encoding="Latin")
    df = df.reset_index(drop=True)
    df = df.reset_index()
    df.rename(columns={'index':'old_index'},inplace=True)

    dfAssunto = df[['old_index','assuntoPrincipal']].copy()

    dfAssunto['assuntoPrincipal'] = dfAssunto['assuntoPrincipal'].str.replace('+','',regex=False)

    dfAssunto[['raiz','folha_1','folha_2']] = dfAssunto['assuntoPrincipal'].str.split(":", expand = True)

    dfAssunto.drop(columns=['assuntoPrincipal'],inplace=True)
    dfAssunto = dfAssunto.astype(int)

    dfAssunto = dfAssunto.merge(deparaDf,left_on='raiz',right_on='cod')
    dfAssunto.rename(columns={'Assunto':'raiz_texto'},inplace=True)
    dfAssunto.drop(columns=['cod'],inplace=True)

    dfAssunto = dfAssunto.merge(deparaDf,left_on='folha_1',right_on='cod')
    dfAssunto.rename(columns={'Assunto':'folha_1_texto'},inplace=True)
    dfAssunto.drop(columns=['cod'],inplace=True)

    dfAssunto = dfAssunto.merge(deparaDf,left_on='folha_2',right_on='cod')
    dfAssunto.rename(columns={'Assunto':'folha_2_texto'},inplace=True)
    dfAssunto.drop(columns=['cod'],inplace=True)

    dfAssunto['assuntoPrincipal_texto'] = dfAssunto['raiz_texto'] + '_/_' + dfAssunto['folha_1_texto'] + '_/_' + dfAssunto['folha_2_texto']

    dfAssunto = dfAssunto[['old_index','assuntoPrincipal_texto']]

    df = df.merge(dfAssunto, how='left' , on='old_index')

    df.drop(columns=['assuntoPrincipal','old_index'],inplace=True)
    df.rename(columns={'assuntoPrincipal_texto':'assuntoPrincipal'},inplace=True)

    df = df[['text','assuntoPrincipal','split']]

    return df

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

    labelList = np.unique(dataset['stars'].to_numpy()).tolist()
    id2label = {idx:label for idx, label in enumerate(labelList)}
    label2id = {label:idx for idx, label in enumerate(labelList)}
    dataset['labels'] = dataset['stars'].apply(lambda row: label2id[row])

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
    
    XTest.to_csv(os.path.join(ROOT_DIR,'data','yelp','interm','test','test.csv'),index=False)
    XTrain.to_csv(os.path.join(ROOT_DIR,'data','yelp','interm','train','train.csv'),index=False)
    Xval.to_csv(os.path.join(ROOT_DIR,'data','yelp','interm','val','val.csv'),index=False)

    with open(os.path.join(ROOT_DIR,'data','yelp','interm','id2label.json'),'w') as f:
        json.dump(id2label,f)
        f.close()
    with open(os.path.join(ROOT_DIR,'data','yelp','interm','label2id.json'),'w') as f:
        json.dump(label2id,f)
        f.close()

    return None

def dmoz_cats(test_split):

    """
    File source: https://data.mendeley.com/datasets/9mpgz8z257/1
    Load raw data in a pytorch dataset class
    All data will follow the (label,text) format
    """

    #https://data.mendeley.com/datasets/9mpgz8z257/1


    def build_url_df():

        path = os.path.join(ROOT_DIR,'data','dmoz','raw','urls')
        fileList = []
        for file in os.listdir(path):
            if file.endswith(".txt"):
                fileList.append(file)

        dfList = []

        for file in fileList:

            df = pd.read_csv(os.path.join(path,file),sep='\t',names=['url'])
            df['topic_id'] = int(file.replace('.txt',''))
            dfList.append(df)

        df = pd.concat(dfList)

        df = df.astype({"url": str, "topic_id": np.int32})

        df.reset_index(inplace=True)
        df.rename(columns={'index':'file_id'},inplace=True)
        df.set_index(['file_id','topic_id'],inplace=True,drop=True)

        return df

    def build_topic_name_df():

        path = os.path.join(ROOT_DIR,'data','dmoz','raw','topic_names')

        df = pd.read_csv(os.path.join(path,'topic_names.txt'),sep='\t',names=['topic_id','topic_names'])

        df = df.astype({"topic_id": np.int32, "topic_names": str})

        df.drop_duplicates(inplace=True)
        df.set_index(['topic_id'],inplace=True,drop=True)

        return df

    def build_topic_desc():

        path = os.path.join(ROOT_DIR,'data','dmoz','raw','topic_descriptions','texts')
        fileList = []
        for file in os.listdir(path):
            if file.endswith(".txt"):
                fileList.append(file)

        strList = []

        for file in fileList:

            with open(os.path.join(path,file), encoding='utf8') as f:
                topic_id = int(file.replace('.txt',''))
                contents = f.read() #checar algum problema de encoding, se encoding != utf8 apagar?
                strList.append([contents,topic_id])
                f.close()

        df = pd.DataFrame(strList,columns=['topic_desc','topic_id'])

        df = df.astype({"topic_desc": str, "topic_id": np.int32})

        df.drop_duplicates(inplace=True)
        df.set_index(['topic_id'],inplace=True,drop=True)

        return df

    def build_text_df():

        folderPath = os.path.join(ROOT_DIR,'data','dmoz','raw','texts')

        folderFileList = []
        for folder in os.listdir(folderPath):

            path = os.path.join(ROOT_DIR,'data','dmoz','raw','texts',folder)
            fileList = []
            for file in os.listdir(path):
                if file.endswith(".txt"):
                    fileList.append([file,int(folder)])
            folderFileList.append(fileList)

        txtList = []

        for folderArr in folderFileList:

            strList = []
            for fileArr in folderArr:

                folder = fileArr[1]
                file = fileArr[0]
                path = os.path.join(ROOT_DIR,'data','dmoz','raw','texts',str(folder))
                
                if (folder == 335 or folder == 94) and file.find('a') != -1: #folder bugado!!

                    fSizeList = list(filter(lambda file: file.find('a') == -1 , os.listdir(path)))
                    fSizeList = list(map(lambda file: int(file.replace('.txt','')) , fSizeList))
                    fSizeList.sort()
                    fSizeList = fSizeList[-1]

                    with open(os.path.join(path,file)) as f:

                        if (folder == 335) and file.find('a') != -1:
                            file_id = int(file.replace('a.txt','')) + fSizeList + 2
                        else:
                            file_id = int(file.replace('a.txt','')) + fSizeList + 1
                        contents = f.read() #checar algum problema de encoding, se encoding != utf8 apagar?
                        strList.append([contents,file_id,folder])
                        f.close()

                else:

                    with open(os.path.join(path,file)) as f:
                        file_id = int(file.replace('.txt',''))
                        contents = f.read() #checar algum problema de encoding, se encoding != utf8 apagar?
                        strList.append([contents,file_id,folder])
                        f.close()

            txtList.append(strList)

        #flatten the result:
        txtList = [item for sublist in txtList for item in sublist]

        df = pd.DataFrame(txtList,columns=['text_contents','file_id','topic_id'])

        df = df.astype({"text_contents": str, "file_id": np.int32, "topic_id": np.int32})

        df.set_index(['file_id','topic_id'],inplace=True,drop=True)

        return df

    def merge_df():

        urlDf = build_url_df()
        topicDf = build_topic_name_df()
        descDf = build_topic_desc()
        txtDf = build_text_df()

        df = urlDf.join(txtDf)
        dfTopic = topicDf.join(descDf)
        df = df.join(dfTopic)

        del urlDf
        del topicDf
        del descDf
        del txtDf
        del dfTopic
        gc.collect()

        return df.sample(frac=1)

    def split_data_ml(dataset,test_split):

        X_train, X_testval, y_train, y_testval = train_test_split(dataset['text'],dataset['labels'], test_size=test_split)#, stratify= dataset['labels'])

        del dataset
        gc.collect()

        XTestval = pd.DataFrame()
        XTestval['labels'] = y_testval.to_frame()
        XTestval['text'] = X_testval.to_frame()

        X_test, X_val, y_test, y_val = train_test_split(XTestval['text'],XTestval['labels'], test_size=test_split)#, stratify= XTestval['labels'] )

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

        return XTest,XTrain,Xval

    df = merge_df()
    df.reset_index(inplace=True)
    
    df = df[['topic_names','text_contents']]
    topQtyLabels = 30
    df2 = df.groupby(['topic_names']).count().sort_values(by=['text_contents'],ascending=False).head(topQtyLabels)
    topList = list(df2.index)

    del df2
    gc.collect()

    df = df[df['topic_names'].isin(topList)]

    labelList = np.unique(df['topic_names'].to_numpy()).tolist()

    id2label = {idx:label for idx, label in enumerate(labelList)}
    label2id = {label:idx for idx, label in enumerate(labelList)}

    df['labels'] = df['topic_names'].apply(lambda row: label2id[row])

    df.drop(columns=['topic_names'],inplace=True)
    df.rename(columns={'text_contents':'text'},inplace=True)

    df=df.dropna()
    df.drop_duplicates(subset=['text'],inplace=True) #?

    dataset_510_1500 = df[df["text"].apply(lambda x: len(x) > 3500 and len(x) < 10000)] 
    dataset_1500 = df[df["text"].apply(lambda x: len(x) > 10000)] 
    dataset_4090 = df[df["text"].apply(lambda x: len(x) > 30000)] 

    del df
    gc.collect()

    XTest_510_1500,XTrain_510_1500,Xval_510_1500 = split_data_ml(dataset_510_1500,test_split)
    XTest_1500,XTrain_1500,Xval_1500 = split_data_ml(dataset_1500,test_split)
    XTest_4090,XTrain_4090,Xval_4090 = split_data_ml(dataset_4090,test_split)
    
    XTest_510_1500.to_csv(os.path.join(ROOT_DIR,'data','dmoz_510_1500','interm','test','test.csv'),index=False)
    XTrain_510_1500.to_csv(os.path.join(ROOT_DIR,'data','dmoz_510_1500','interm','train','train.csv'),index=False)
    Xval_510_1500.to_csv(os.path.join(ROOT_DIR,'data','dmoz_510_1500','interm','val','val.csv'),index=False)

    XTest_1500.to_csv(os.path.join(ROOT_DIR,'data','dmoz_1500','interm','test','test.csv'),index=False)
    XTrain_1500.to_csv(os.path.join(ROOT_DIR,'data','dmoz_1500','interm','train','train.csv'),index=False)
    Xval_1500.to_csv(os.path.join(ROOT_DIR,'data','dmoz_1500','interm','val','val.csv'),index=False)

    XTest_4090.to_csv(os.path.join(ROOT_DIR,'data','dmoz_4090','interm','test','test.csv'),index=False)
    XTrain_4090.to_csv(os.path.join(ROOT_DIR,'data','dmoz_4090','interm','train','train.csv'),index=False)
    Xval_4090.to_csv(os.path.join(ROOT_DIR,'data','dmoz_4090','interm','val','val.csv'),index=False)

    with open(os.path.join(ROOT_DIR,'data','dmoz_510_1500','interm','id2label.json'),'w') as f:
        json.dump(id2label,f)
        f.close()
    with open(os.path.join(ROOT_DIR,'data','dmoz_510_1500','interm','label2id.json'),'w') as f:
        json.dump(label2id,f)
        f.close()
    with open(os.path.join(ROOT_DIR,'data','dmoz_1500','interm','id2label.json'),'w') as f:
        json.dump(id2label,f)
        f.close()
    with open(os.path.join(ROOT_DIR,'data','dmoz_1500','interm','label2id.json'),'w') as f:
        json.dump(label2id,f)
        f.close()
    with open(os.path.join(ROOT_DIR,'data','dmoz_4090','interm','id2label.json'),'w') as f:
        json.dump(id2label,f)
        f.close()
    with open(os.path.join(ROOT_DIR,'data','dmoz_4090','interm','label2id.json'),'w') as f:
        json.dump(label2id,f)
        f.close()

    return None

def echr_build(problem_column,max_classes):

    #https://archive.org/details/ECHR-ACL2019

    #a dúvida aqui é se concatenaria todas as colunas 'problemas'

    def get_file_list(folderName):

        path = os.path.join(ROOT_DIR,'data','echr','raw')

        path = os.path.join(path,folderName)
        fileList = []
        for file in os.listdir(path):
            if file.endswith(".json"):
                fileList.append(file)

        return fileList

    path = os.path.join(ROOT_DIR,'data','echr','raw')

    folderNames = ['EN_dev','EN_test','EN_train']#,'EN_dev_Anon','EN_test_Anon','EN_train_Anon']

    dfList = []
    strList = []

    for name in folderNames:
        strList = []
        for file in get_file_list(name):

            with open(os.path.join(path,name,file)) as f:
                contents = json.load(f)
                contents['FOLDER'] = name
                strList.append(contents)
                f.close()

        dfList.append(strList)
        
    dfList = [item for sublist in dfList for item in sublist]

    df = pd.DataFrame(dfList)
    df['TEXT'] = df['TEXT'].apply(lambda row: ''.join(row))

    df = df.sample(frac=1) #shuffle

    problemList = df[problem_column].to_list()
    problemList = [item for sublist in problemList for item in sublist]

    unique, counts = np.unique(np.array(problemList), return_counts=True)

    arr1inds = counts.argsort()
    #sorted_arr1 = counts[arr1inds[::-1]]
    sorted_arr2 = unique[arr1inds[::-1]]

    problemList = sorted_arr2[:max_classes]

    def replace_extra_articles(row):

        new_row = []

        for n in row:
            if n in problemList:
                new_row.append(n)

        return new_row

    def remove_extra_articles(row):

        for n in row:
            if n not in problemList:
                return 1

    #df['help'] = df[problem_column].apply(lambda row: remove_extra_articles(row))
    #df = df[df['help']!=1]

    df[problem_column] = df[problem_column].apply(lambda row: replace_extra_articles(row))

    max_label_num = len(problemList)

    id2label = {idx:label for idx, label in enumerate(problemList)}
    label2id = {label:idx for idx, label in enumerate(problemList)}

    df = df[[problem_column,'TEXT','FOLDER']]

    #### multi label encoding:
    def encoding_multi_labels(arr,label2id,max_label_num):

        id2label_arr = [0.0] * max_label_num

        for n in arr:
            id2label_arr[label2id[n]]=1.0

        return id2label_arr

    df['labels'] = df[problem_column].apply(lambda row: encoding_multi_labels(row,label2id,max_label_num))

    ####

    df.drop(columns=[problem_column],inplace=True)
    df.rename(columns={'TEXT':'text'},inplace=True)

    df = df[['labels','text','FOLDER']]

    df_test = df[df['FOLDER']=='EN_test'].copy()
    df_train = df[df['FOLDER']=='EN_train'].copy()
    df_val = df[df['FOLDER']=='EN_dev'].copy()

    del df
    gc.collect()

    df_test.drop(columns=['FOLDER'],inplace=True)
    df_train.drop(columns=['FOLDER'],inplace=True)
    df_val.drop(columns=['FOLDER'],inplace=True)

    df_test.to_csv(os.path.join(ROOT_DIR,'data','echr','interm','test','test.csv'),index=False)
    df_train.to_csv(os.path.join(ROOT_DIR,'data','echr','interm','train','train.csv'),index=False)
    df_val.to_csv(os.path.join(ROOT_DIR,'data','echr','interm','val','val.csv'),index=False)
    
    with open(os.path.join(ROOT_DIR,'data','echr','interm','id2label.json'),'w') as f:
        json.dump(id2label,f)
        f.close()
    with open(os.path.join(ROOT_DIR,'data','echr','interm','label2id.json'),'w') as f:
        json.dump(label2id,f)
        f.close()
     
    return None

def eurlex_lexbench(max_classes):

    #https://zenodo.org/record/5532997/files/eurlex.tar.gz

    path = os.path.join(ROOT_DIR,'data','eurlex_lexbench','raw','eurlex.jsonl')

    df = pd.read_json(path,lines=True)

    df = df.sample(frac=1) #shuffle

    #teste vazio:
    #df = df[df[problem_column].apply(lambda row: len(row))>0]

    problemList = df['labels'].to_list()
    problemList = [item for sublist in problemList for item in sublist]

    ### tirar as linhas que tem alguma class extra?? ou simplesmente editar e tirar ela de dentro da lista???

    unique, counts = np.unique(np.array(problemList), return_counts=True)

    arr1inds = counts.argsort()
    #sorted_arr1 = counts[arr1inds[::-1]]
    sorted_arr2 = unique[arr1inds[::-1]]

    problemList = sorted_arr2[:max_classes]

    #fazer duas versoes:

    def replace_extra_articles(row):

        new_row = []

        for n in row:
            if n in problemList:
                new_row.append(n)

        return new_row

    def remove_extra_articles(row):

        for n in row:
            if n not in problemList:
                return 1

    #df['help'] = df['labels'].apply(lambda row: remove_extra_articles(row))
    #df = df[df['help']!=1]

    df['labels'] = df['labels'].apply(lambda row: replace_extra_articles(row))

    max_label_num = len(problemList)

    id2label = {idx:label for idx, label in enumerate(problemList)}
    label2id = {label:idx for idx, label in enumerate(problemList)}

    #### multi label encoding:
    def encoding_multi_labels(arr,label2id,max_label_num):

        id2label_arr = [0.0] * max_label_num

        for n in arr:
            id2label_arr[label2id[n]]=1.0

        return id2label_arr

    df['labels'] = df['labels'].apply(lambda row: encoding_multi_labels(row,label2id,max_label_num))

    df = df[['labels','text','data_type']]

    df_test = df[df['data_type']=='test'].copy()
    df_train = df[df['data_type']=='train'].copy()
    df_val = df[df['data_type']=='dev'].copy()

    del df
    gc.collect()

    df_test.drop(columns=['data_type'],inplace=True)
    df_train.drop(columns=['data_type'],inplace=True)
    df_val.drop(columns=['data_type'],inplace=True)

    df_test.to_csv(os.path.join(ROOT_DIR,'data','eurlex_lexbench','interm','test','test.csv'),index=False)
    df_train.to_csv(os.path.join(ROOT_DIR,'data','eurlex_lexbench','interm','train','train.csv'),index=False)
    df_val.to_csv(os.path.join(ROOT_DIR,'data','eurlex_lexbench','interm','val','val.csv'),index=False)
    
    with open(os.path.join(ROOT_DIR,'data','eurlex_lexbench','interm','id2label.json'),'w') as f:
        json.dump(id2label,f)
        f.close()
    with open(os.path.join(ROOT_DIR,'data','eurlex_lexbench','interm','label2id.json'),'w') as f:
        json.dump(label2id,f)
        f.close()

    return None

def colab_torch_tweets():

    #https://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/SemEval2018-Task1-all-data.zip

    path_val = os.path.join(ROOT_DIR,'data','SemEval2018-Task1-all-data','raw','2018-E-c-En-dev.txt')
    path_test = os.path.join(ROOT_DIR,'data','SemEval2018-Task1-all-data','raw','2018-E-c-En-test-gold.txt')
    path_train = os.path.join(ROOT_DIR,'data','SemEval2018-Task1-all-data','raw','2018-E-c-En-train.txt')


    def concat_list(row,to_concat):

        newRow = []

        for col in to_concat:
            if row[col] == 1:
                newRow.append(1.0)
            else:
                newRow.append(0.0)

        return newRow

    def prepare_data(path):
        
        df = pd.read_csv(path,sep='\t')

        to_concat = list(set(df.columns) - set(['ID','Tweet'])) #fica na ordem?

        df['labels'] = df.apply(lambda row: concat_list(row,to_concat),axis=1)

        df = df[['labels','Tweet']]

        df.rename(columns={'Tweet':'text'},inplace=True)

        id2label = {idx:label for idx, label in enumerate(to_concat)}
        label2id = {label:idx for idx, label in enumerate(to_concat)}

        return df,id2label,label2id

    df_val,_,_ = prepare_data(path_val)
    df_test,_,_ = prepare_data(path_test)
    df_train,id2label,label2id = prepare_data(path_train)

    df_test.to_csv(os.path.join(ROOT_DIR,'data','SemEval2018-Task1-all-data','interm','test','test.csv'),index=False)
    df_train.to_csv(os.path.join(ROOT_DIR,'data','SemEval2018-Task1-all-data','interm','train','train.csv'),index=False)
    df_val.to_csv(os.path.join(ROOT_DIR,'data','SemEval2018-Task1-all-data','interm','val','val.csv'),index=False)
    
    with open(os.path.join(ROOT_DIR,'data','SemEval2018-Task1-all-data','interm','id2label.json'),'w') as f:
        json.dump(id2label,f)
        f.close()
    with open(os.path.join(ROOT_DIR,'data','SemEval2018-Task1-all-data','interm','label2id.json'),'w') as f:
        json.dump(label2id,f)
        f.close()

    return None

def unfair_tos_lexbench():

    #https://zenodo.org/record/5532997/files/eurlex.tar.gz

    path = os.path.join(ROOT_DIR,'data','unfair_lexbench','raw','unfair_tos.jsonl')

    df = pd.read_json(path,lines=True)

    df = df.sample(frac=1) #shuffle

    #teste vazio:
    #df = df[df[problem_column].apply(lambda row: len(row))>0]

    problemList = df['labels'].to_list()
    problemList = [item for sublist in problemList for item in sublist]

    ### tirar as linhas que tem alguma class extra?? ou simplesmente editar e tirar ela de dentro da lista???

    unique, counts = np.unique(np.array(problemList), return_counts=True)

    arr1inds = counts.argsort()
    #sorted_arr1 = counts[arr1inds[::-1]]
    sorted_arr2 = unique[arr1inds[::-1]]

    problemList = sorted_arr2[:]

    #fazer duas versoes:
    def remove_extra_articles(row):

        for n in row:
            if n not in problemList:
            #if n in problemList, pop
                return 1

    df['help'] = df['labels'].apply(lambda row: remove_extra_articles(row))

    df = df[df['help']!=1]

    max_label_num = len(problemList)

    id2label = {idx:label for idx, label in enumerate(problemList)}
    label2id = {label:idx for idx, label in enumerate(problemList)}

    #### multi label encoding:
    def encoding_multi_labels(arr,label2id,max_label_num):

        id2label_arr = [0.0] * max_label_num

        for n in arr:
            id2label_arr[label2id[n]]=1.0

        return id2label_arr

    df['labels'] = df['labels'].apply(lambda row: encoding_multi_labels(row,label2id,max_label_num))

    df = df[['labels','text','data_type']]

    df_test = df[df['data_type']=='test'].copy()
    df_train = df[df['data_type']=='train'].copy()
    df_val = df[df['data_type']=='val'].copy()

    del df
    gc.collect()

    df_test.drop(columns=['data_type'],inplace=True)
    df_train.drop(columns=['data_type'],inplace=True)
    df_val.drop(columns=['data_type'],inplace=True)

    df_test.to_csv(os.path.join(ROOT_DIR,'data','unfair_lexbench','interm','test','test.csv'),index=False)
    df_train.to_csv(os.path.join(ROOT_DIR,'data','unfair_lexbench','interm','train','train.csv'),index=False)
    df_val.to_csv(os.path.join(ROOT_DIR,'data','unfair_lexbench','interm','val','val.csv'),index=False)
    
    with open(os.path.join(ROOT_DIR,'data','unfair_lexbench','interm','id2label.json'),'w') as f:
        json.dump(id2label,f)
        f.close()
    with open(os.path.join(ROOT_DIR,'data','unfair_lexbench','interm','label2id.json'),'w') as f:
        json.dump(label2id,f)
        f.close()

    return None

def ecthr_a_lexbench():

    #https://zenodo.org/record/5532997/files/ecthr.tar.gz

    path = os.path.join(ROOT_DIR,'data','ecthr_a_lexbench','raw','ecthr.jsonl')

    df = pd.read_json(path,lines=True)

    df = df.sample(frac=1) #shuffle

    problem_column = 'violated_articles'

    #certeza é isso? o artigo nao deixa isso claro.
    df['text'] = df['facts'].apply(lambda row: ''.join(row))
    
    #teste vazio:
    #df = df[df[problem_column].apply(lambda row: len(row))>0]

    problemList = df[problem_column].to_list()
    problemList = [item for sublist in problemList for item in sublist]

    unique, counts = np.unique(np.array(problemList), return_counts=True)

    arr1inds = counts.argsort()
    #sorted_arr1 = counts[arr1inds[::-1]]
    sorted_arr2 = unique[arr1inds[::-1]]

    problemList = sorted_arr2[:]

    def replace_extra_articles(row):

        new_row = []

        for n in row:
            if n in problemList:
                new_row.append(n)

        return new_row

    def remove_extra_articles(row):

        for n in row:
            if n not in problemList:
                return 1

    #df['help'] = df['labels'].apply(lambda row: remove_extra_articles(row))
    #df = df[df['help']!=1]

    df['labels'] = df[problem_column].apply(lambda row: replace_extra_articles(row))

    max_label_num = len(problemList)

    id2label = {idx:label for idx, label in enumerate(problemList)}
    label2id = {label:idx for idx, label in enumerate(problemList)}

    #### multi label encoding:
    def encoding_multi_labels(arr,label2id,max_label_num):

        id2label_arr = [0.0] * max_label_num

        for n in arr:
            id2label_arr[label2id[n]]=1.0

        return id2label_arr

    df['labels'] = df['labels'].apply(lambda row: encoding_multi_labels(row,label2id,max_label_num))

    df = df[['labels','text','data_type']]

    df_test = df[df['data_type']=='test'].copy()
    df_train = df[df['data_type']=='train'].copy()
    df_val = df[df['data_type']=='dev'].copy()

    del df
    gc.collect()

    df_test.drop(columns=['data_type'],inplace=True)
    df_train.drop(columns=['data_type'],inplace=True)
    df_val.drop(columns=['data_type'],inplace=True)

    df_test.to_csv(os.path.join(ROOT_DIR,'data','ecthr_a_lexbench','interm','test','test.csv'),index=False)
    df_train.to_csv(os.path.join(ROOT_DIR,'data','ecthr_a_lexbench','interm','train','train.csv'),index=False)
    df_val.to_csv(os.path.join(ROOT_DIR,'data','ecthr_a_lexbench','interm','val','val.csv'),index=False)
    
    with open(os.path.join(ROOT_DIR,'data','ecthr_a_lexbench','interm','id2label.json'),'w') as f:
        json.dump(id2label,f)
        f.close()
    with open(os.path.join(ROOT_DIR,'data','ecthr_a_lexbench','interm','label2id.json'),'w') as f:
        json.dump(label2id,f)
        f.close()

    return None

def ecthr_b_lexbench():

    #https://zenodo.org/record/5532997/files/ecthr.tar.gz

    path = os.path.join(ROOT_DIR,'data','ecthr_b_lexbench','raw','ecthr.jsonl')

    df = pd.read_json(path,lines=True)

    df = df.sample(frac=1) #shuffle

    problem_column = 'allegedly_violated_articles'

    #certeza é isso? o artigo nao deixa isso claro.
    df['text'] = df['facts'].apply(lambda row: ''.join(row))
    
    #teste vazio:
    #df = df[df[problem_column].apply(lambda row: len(row))>0]

    problemList = df[problem_column].to_list()
    problemList = [item for sublist in problemList for item in sublist]

    unique, counts = np.unique(np.array(problemList), return_counts=True)

    arr1inds = counts.argsort()
    #sorted_arr1 = counts[arr1inds[::-1]]
    sorted_arr2 = unique[arr1inds[::-1]]

    problemList = sorted_arr2[:]

    def replace_extra_articles(row):

        new_row = []

        for n in row:
            if n in problemList:
                new_row.append(n)

        return new_row

    def remove_extra_articles(row):

        for n in row:
            if n not in problemList:
                return 1

    #df['help'] = df['labels'].apply(lambda row: remove_extra_articles(row))
    #df = df[df['help']!=1]

    df['labels'] = df[problem_column].apply(lambda row: replace_extra_articles(row))

    max_label_num = len(problemList)

    id2label = {idx:label for idx, label in enumerate(problemList)}
    label2id = {label:idx for idx, label in enumerate(problemList)}

    #### multi label encoding:
    def encoding_multi_labels(arr,label2id,max_label_num):

        id2label_arr = [0.0] * max_label_num

        for n in arr:
            id2label_arr[label2id[n]]=1.0

        return id2label_arr

    df['labels'] = df['labels'].apply(lambda row: encoding_multi_labels(row,label2id,max_label_num))

    df = df[['labels','text','data_type']]

    df_test = df[df['data_type']=='test'].copy()
    df_train = df[df['data_type']=='train'].copy()
    df_val = df[df['data_type']=='dev'].copy()

    del df
    gc.collect()

    df_test.drop(columns=['data_type'],inplace=True)
    df_train.drop(columns=['data_type'],inplace=True)
    df_val.drop(columns=['data_type'],inplace=True)

    df_test.to_csv(os.path.join(ROOT_DIR,'data','ecthr_b_lexbench','interm','test','test.csv'),index=False)
    df_train.to_csv(os.path.join(ROOT_DIR,'data','ecthr_b_lexbench','interm','train','train.csv'),index=False)
    df_val.to_csv(os.path.join(ROOT_DIR,'data','ecthr_b_lexbench','interm','val','val.csv'),index=False)
    
    with open(os.path.join(ROOT_DIR,'data','ecthr_b_lexbench','interm','id2label.json'),'w') as f:
        json.dump(id2label,f)
        f.close()
    with open(os.path.join(ROOT_DIR,'data','ecthr_b_lexbench','interm','label2id.json'),'w') as f:
        json.dump(label2id,f)
        f.close()

    return None

def scotus_lexbench():

    #https://zenodo.org/record/5532997/files/scotus.tar.gz

    path = os.path.join(ROOT_DIR,'data','scotus_lexbench','raw','scotus.jsonl')

    df = pd.read_json(path,lines=True)

    df = df.sample(frac=1) #shuffle

    problem_column = 'issueArea'
    df = df.astype({problem_column: np.int32})

    problemList = df[problem_column].to_list()

    unique, counts = np.unique(np.array(problemList), return_counts=True)

    arr1inds = counts.argsort()
    #sorted_arr1 = counts[arr1inds[::-1]]
    sorted_arr2 = unique[arr1inds[::-1]]

    problemList = sorted_arr2[:]

    max_label_num = len(problemList)

    id2label = {idx:label for idx, label in enumerate(problemList.tolist())}
    label2id = {label:idx for idx, label in enumerate(problemList.tolist())}
    df['labels'] = df[problem_column].apply(lambda row: label2id[row])

    df = df[['labels','text','data_type']]

    df_test = df[df['data_type']=='test'].copy()
    df_train = df[df['data_type']=='train'].copy()
    df_val = df[df['data_type']=='dev'].copy()

    del df
    gc.collect()

    df_test.drop(columns=['data_type'],inplace=True)
    df_train.drop(columns=['data_type'],inplace=True)
    df_val.drop(columns=['data_type'],inplace=True)

    df_test.to_csv(os.path.join(ROOT_DIR,'data','scotus_lexbench','interm','test','test.csv'),index=False)
    df_train.to_csv(os.path.join(ROOT_DIR,'data','scotus_lexbench','interm','train','train.csv'),index=False)
    df_val.to_csv(os.path.join(ROOT_DIR,'data','scotus_lexbench','interm','val','val.csv'),index=False)
    
    with open(os.path.join(ROOT_DIR,'data','scotus_lexbench','interm','id2label.json'),'w') as f:
        json.dump(id2label,f)
        f.close()
    with open(os.path.join(ROOT_DIR,'data','scotus_lexbench','interm','label2id.json'),'w') as f:
        json.dump(label2id,f)
        f.close()

    return None

def ledgar_lexbench():

    #https://zenodo.org/record/5532997/files/ledgar.tar.gz

    path = os.path.join(ROOT_DIR,'data','ledgar_lexbench','raw','ledgar.jsonl')

    df = pd.read_json(path,lines=True)

    df = df.sample(frac=1) #shuffle

    problem_column = 'clause_type'

    problemList = df[problem_column].to_list()

    unique, counts = np.unique(np.array(problemList), return_counts=True)

    arr1inds = counts.argsort()
    #sorted_arr1 = counts[arr1inds[::-1]]
    sorted_arr2 = unique[arr1inds[::-1]]

    problemList = sorted_arr2[:]

    max_label_num = len(problemList)

    id2label = {idx:label for idx, label in enumerate(problemList.tolist())}
    label2id = {label:idx for idx, label in enumerate(problemList.tolist())}
    df['labels'] = df[problem_column].apply(lambda row: label2id[row])

    df = df[['labels','text','data_type']]

    df_test = df[df['data_type']=='test'].copy()
    df_train = df[df['data_type']=='train'].copy()
    df_val = df[df['data_type']=='dev'].copy()

    del df
    gc.collect()

    df_test.drop(columns=['data_type'],inplace=True)
    df_train.drop(columns=['data_type'],inplace=True)
    df_val.drop(columns=['data_type'],inplace=True)

    df_test.to_csv(os.path.join(ROOT_DIR,'data','ledgar_lexbench','interm','test','test.csv'),index=False)
    df_train.to_csv(os.path.join(ROOT_DIR,'data','ledgar_lexbench','interm','train','train.csv'),index=False)
    df_val.to_csv(os.path.join(ROOT_DIR,'data','ledgar_lexbench','interm','val','val.csv'),index=False)
    
    with open(os.path.join(ROOT_DIR,'data','ledgar_lexbench','interm','id2label.json'),'w') as f:
        json.dump(id2label,f)
        f.close()
    with open(os.path.join(ROOT_DIR,'data','ledgar_lexbench','interm','label2id.json'),'w') as f:
        json.dump(label2id,f)
        f.close()

    return None

def fix_scrape_landmarks(max_classes,test_split):

    path = os.path.join(ROOT_DIR,'data','cornell_landmarks','raw','scrapedData.csv')

    df = pd.read_csv(path)
    
    df = df[df["text"].apply(lambda x: len(x) > 1000)] ### checar depois pq aparece isso, pq tem processos que vem do scraper que nao puxa quase nada?

    df = df.sample(frac=1)

    def limpeza_basica(texto):

        texto = texto[169:-124] ### tirando uns html que nao sairam

        texto = re.sub('\. [0-9]+ ', ' ', texto)
        texto = re.sub('\* \* \*', ' ', texto)
        texto = re.sub('\.\' [0-9]+ ', ' ', texto)
        texto = re.sub('\.+ \.+ \.+', ' ', texto)
        texto = re.sub('\[.+?omitted\]', '', texto)
        texto = re.sub('"\." [0-9]+ ', ' ', texto)
        texto = re.sub('\' [0-9]+ [a-z]', ' ', texto,flags=re.IGNORECASE) #deixar todos com essa parada de ignore case?

        texto = re.sub('"\." [0-9]+ ', ' ', texto)

        texto = re.sub('\.[0-9]+ ', '.', texto) #pode mexer muito, muito especifico
        texto = re.sub('\. [0-9]+ ', ' ', texto) #pode mexer muito, muito especifico

        texto = re.sub(' +',' ',texto)

        return texto

    df['text'] = df['text'].apply(lambda row : limpeza_basica(row))

    problem_column = 'subject'

    problemList = df[problem_column].to_list()

    unique, counts = np.unique(np.array(problemList), return_counts=True)

    arr1inds = counts.argsort()
    #sorted_arr1 = counts[arr1inds[::-1]]
    sorted_arr2 = unique[arr1inds[::-1]]

    problemList = sorted_arr2[:max_classes]

    df = df[df['subject'].isin(problemList)]

    max_label_num = len(problemList)

    id2label = {idx:label for idx, label in enumerate(problemList.tolist())}
    label2id = {label:idx for idx, label in enumerate(problemList.tolist())}
    df['labels'] = df[problem_column].apply(lambda row: label2id[row])

    df = df[['labels','text']]

    X_train, X_testval, y_train, y_testval = train_test_split(df['text'],df['labels'], test_size=test_split, stratify= df['labels'])

    del df
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
    
    XTest.to_csv(os.path.join(ROOT_DIR,'data','cornell_landmarks','interm','test','test.csv'),index=False)
    XTrain.to_csv(os.path.join(ROOT_DIR,'data','cornell_landmarks','interm','train','train.csv'),index=False)
    Xval.to_csv(os.path.join(ROOT_DIR,'data','cornell_landmarks','interm','val','val.csv'),index=False)

    with open(os.path.join(ROOT_DIR,'data','cornell_landmarks','interm','id2label.json'),'w') as f:
        json.dump(id2label,f)
        f.close()
    with open(os.path.join(ROOT_DIR,'data','cornell_landmarks','interm','label2id.json'),'w') as f:
        json.dump(label2id,f)
        f.close()

    with open(os.path.join(ROOT_DIR,'data','cornell_landmarks_summary','interm','id2label.json'),'w') as f:
        json.dump(id2label,f)
        f.close()
    with open(os.path.join(ROOT_DIR,'data','cornell_landmarks_summary','interm','label2id.json'),'w') as f:
        json.dump(label2id,f)
        f.close()

    return None

def summary_facebook():

    def get_syllabus(row,char_size):

        text = row

        end_text = 'delivered the opinion of the Court'
        start_text = 'Syllabus'

        start_pos = text.find(start_text)
        end_pos = text.find(end_text)

        if (start_pos > 0 and end_pos > 0) and (end_pos > start_pos):
            text = text[start_pos+len(start_text):end_pos]

            if len(text) < int(char_size/2):
                text = text[start_pos+len(start_text):start_pos+char_size]

        elif start_pos < 0 and end_pos < 0:
            text = text[0:char_size] # o tokenizer trunca, nao tem problema aq
        elif start_pos > 0:
            text = text[start_pos+len(start_text):start_pos+char_size] # o tokenizer trunca, nao tem problema aq
        elif end_pos > 0:
            text = text[end_pos+len(end_text):end_pos+char_size] # o tokenizer trunca, nao tem problema aq
        else:
            text = text[0:char_size]

        if len(text) < int(char_size/2):
            text = text[0:char_size]

        return text.strip()

    modelPath = os.path.join(ROOT_DIR,'lawclassification','models','external','facebook.bart-large-cnn')

    summarizerFacebook = pipeline("summarization", model=modelPath, device=0)
    tokenizerFaceBook = summarizerFacebook.tokenizer

    testPath = os.path.join(ROOT_DIR,'data','cornell_landmarks','interm','test','test.csv')
    trainPath = os.path.join(ROOT_DIR,'data','cornell_landmarks','interm','train','train.csv')
    valPath = os.path.join(ROOT_DIR,'data','cornell_landmarks','interm','val','val.csv')

    def summary_df(testPath):

        df = pd.read_csv(testPath)
        print(len(df))
        #df = df.head(4)

        df['text'] = df['text'].astype(str)
        df['labels'] = df['labels'].astype(int)

        #df['text'] = df['text'].apply(lambda row : get_syllabus(row,4700))

        df = df[['labels','text']]

        dfList = tokenizerFaceBook(df['text'].tolist(),padding='max_length', max_length = int((1024)*0.98), truncation=True, return_tensors="pt")
        dfList = [tokenizerFaceBook.decode(idx, skip_special_tokens=True) for idx in dfList.input_ids]
        dfList = summarizerFacebook(dfList, max_length=512, min_length=256, do_sample=False)
        dfList = [idx['summary_text'] for idx in dfList]

        df['text'] = pd.Series(dfList).to_frame()

        return df

    summary_df(testPath).to_csv(os.path.join(ROOT_DIR,'data','cornell_landmarks_summary','interm','test','test.csv'),index=False)
    summary_df(trainPath).to_csv(os.path.join(ROOT_DIR,'data','cornell_landmarks_summary','interm','train','train.csv'),index=False)
    summary_df(valPath).to_csv(os.path.join(ROOT_DIR,'data','cornell_landmarks_summary','interm','val','val.csv'),index=False)

def customer_complain_check_boost(max_row,test_split):

    path = os.path.join(ROOT_DIR,'data','customer_complain','raw','rows.csv')

    df = pd.read_csv(path)
    df = df.sample(frac=1)
    df = df[['Product', 'Consumer complaint narrative']].copy()
    df = df[pd.notnull(df['Consumer complaint narrative'])]

    df = df.sample(n=max_row, random_state=1)

    #print(len(df))

    df.replace({'Product': 
                {'Credit reporting, credit repair services, or other personal consumer reports': 
                'Credit reporting, repair, or other', 
                'Credit reporting': 'Credit reporting, repair, or other',
                'Credit card': 'Credit card or prepaid card',
                'Prepaid card': 'Credit card or prepaid card',
                'Payday loan': 'Payday loan, title loan, or personal loan',
                'Money transfer': 'Money transfer, virtual currency, or money service',
                'Virtual currency': 'Money transfer, virtual currency, or money service'}}, 
                inplace= True)

    problem_column = 'Product'

    problemList = df[problem_column].to_list()

    unique, counts = np.unique(np.array(problemList), return_counts=True)

    arr1inds = counts.argsort()
    sorted_arr2 = unique[arr1inds[::-1]]

    problemList = sorted_arr2[:]

    max_label_num = len(problemList)

    id2label = {idx:label for idx, label in enumerate(problemList.tolist())}
    label2id = {label:idx for idx, label in enumerate(problemList.tolist())}
    df['labels'] = df[problem_column].apply(lambda row: label2id[row])

    df.rename(columns={'Consumer complaint narrative':'text'},inplace=True)
    df = df[['labels','text']]

    X_train, X_testval, y_train, y_testval = train_test_split(df['text'],df['labels'], test_size=test_split)#, stratify= df['labels'])

    del df
    gc.collect()

    XTestval = pd.DataFrame()
    XTestval['labels'] = y_testval.to_frame()
    XTestval['text'] = X_testval.to_frame()

    X_test, X_val, y_test, y_val = train_test_split(XTestval['text'],XTestval['labels'], test_size=test_split)#, stratify= XTestval['labels'] )

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
    
    XTest.to_csv(os.path.join(ROOT_DIR,'data','customer_complain','interm','test','test.csv'),index=False)
    XTrain.to_csv(os.path.join(ROOT_DIR,'data','customer_complain','interm','train','train.csv'),index=False)
    Xval.to_csv(os.path.join(ROOT_DIR,'data','customer_complain','interm','val','val.csv'),index=False)

    with open(os.path.join(ROOT_DIR,'data','customer_complain','interm','id2label.json'),'w') as f:
        json.dump(id2label,f)
        f.close()
    with open(os.path.join(ROOT_DIR,'data','customer_complain','interm','label2id.json'),'w') as f:
        json.dump(label2id,f)
        f.close()

    return None

def ohsumed_create(test_split:float,max_classes:int,max_row:int) -> None:

    #wget http://disi.unitn.it/moschitti/corpora/ohsumed-all-docs.tar.gz

    path = os.path.join(ROOT_DIR,'data','ohsumed','raw','ohsumed-first-20000-docs')

    pathSplit = ['training','test']

    fileContentList = []
    for splitsPath in pathSplit:
        for folder in os.listdir(os.path.join(path,splitsPath)):

            folderPath = os.path.join(ROOT_DIR,'data','ohsumed','raw','ohsumed-first-20000-docs',splitsPath,folder)
            for file in os.listdir(folderPath):

                with open(os.path.join(folderPath,file), encoding='utf8') as f:
                    labelId = folder
                    contents = f.read()
                    fileContentList.append([labelId,contents,splitsPath])
                    f.close()

    df = pd.DataFrame(fileContentList,columns=['labels','text','split'])
    
    df = df.sample(frac=1) # shuffle data
    df = df.head(max_row)

    df['countHelper'] = 1
    duplicDf = df.groupby(by=['text'])['countHelper'].transform(np.sum)
    duplicDf = duplicDf.to_frame()
    df = df.join(duplicDf,lsuffix='_df',rsuffix='_dup')
    df = df[df['countHelper_dup']==1]
    df = df[['labels','text']]

    problemList = np.unique(df['labels'].to_numpy()).tolist()
    unique, counts = np.unique(np.array(problemList), return_counts=True)
    arr1inds = counts.argsort()
    sorted_arr2 = unique[arr1inds[::-1]]
    problemList = sorted_arr2[:max_classes]
    df = df[df['labels'].isin(problemList)]

    id2label = {idx:label for idx, label in enumerate(problemList.tolist())}
    label2id = {label:idx for idx, label in enumerate(problemList.tolist())}
    df['labels'] = df['labels'].apply(lambda row: label2id[row])

    X_train, X_testval, y_train, y_testval = train_test_split(df['text'],df['labels'], test_size=test_split, stratify= df['labels'])

    del df
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
    
    XTest.to_csv(os.path.join(ROOT_DIR,'data','ohsumed','interm','test','test.csv'),index=False)
    XTrain.to_csv(os.path.join(ROOT_DIR,'data','ohsumed','interm','train','train.csv'),index=False)
    Xval.to_csv(os.path.join(ROOT_DIR,'data','ohsumed','interm','val','val.csv'),index=False)

    with open(os.path.join(ROOT_DIR,'data','ohsumed','interm','id2label.json'),'w') as f:
        json.dump(id2label,f)
        f.close()
    with open(os.path.join(ROOT_DIR,'data','ohsumed','interm','label2id.json'),'w') as f:
        json.dump(label2id,f)
        f.close()

def tj(test_split:float,selected_col:str) -> None:

    path = os.path.join(ROOT_DIR,'data','tj','raw','dataset_tjce_v1_trata_texto.csv.zip')

    df = pd.read_csv(path)

    #df = df[df['tamanho_texto']>500]

    df.rename(columns={'texto_tratado':'text'},inplace=True)
    cols = ['text'] + [selected_col]
    df.index = df['documento_id']

    df = df[cols].drop_duplicates()

    df = df.sample(frac=1)

    labelList = np.unique(df[selected_col]).tolist()
    id2label = {idx:label for idx, label in enumerate(labelList)}
    label2id = {label:idx for idx, label in enumerate(labelList)}
    df['labels'] = df[selected_col].apply(lambda row: label2id[row])

    df.drop(columns=[selected_col],inplace=True)

    X_train, X_testval, y_train, y_testval = train_test_split(df['text'],df['labels'], test_size=test_split)#, stratify= df['labels'])

    del df
    gc.collect()

    XTestval = pd.DataFrame()
    XTestval['labels'] = y_testval.to_frame()
    XTestval['text'] = X_testval.to_frame()

    X_test, X_val, y_test, y_val = train_test_split(XTestval['text'],XTestval['labels'], test_size=test_split)#, stratify= XTestval['labels'] )

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
    
    XTest.to_csv(os.path.join(ROOT_DIR,'data','tj','interm','test','test.csv'),index=False)
    XTrain.to_csv(os.path.join(ROOT_DIR,'data','tj','interm','train','train.csv'),index=False)
    Xval.to_csv(os.path.join(ROOT_DIR,'data','tj','interm','val','val.csv'),index=False)

    with open(os.path.join(ROOT_DIR,'data','tj','interm','id2label.json'),'w') as f:
        json.dump(id2label,f)
        f.close()
    with open(os.path.join(ROOT_DIR,'data','tj','interm','label2id.json'),'w') as f:
        json.dump(label2id,f)
        f.close()

    return None

def small_tj(test_split:float) -> None:

    path = os.path.join(ROOT_DIR,'data','small_tj','raw','TrainingSet_s34787.csv.xz')
    df = pd.read_csv(path)

    path_test = os.path.join(ROOT_DIR,'data','small_tj','raw','TestSet_s34787.csv.xz')
    df_test = pd.read_csv(path_test)

    pathCTexto = os.path.join(ROOT_DIR,'data','tj','raw','dataset_tjce_v1_trata_texto.csv.zip')

    df_c_texto = pd.read_csv(pathCTexto)
    df_c_texto = df_c_texto[['documento_id','processo_id','texto_tratado']]
    df = df.merge(df_c_texto,on=['processo_id','documento_id'])
    df_test = df_test.merge(df_c_texto,on=['processo_id','documento_id'])

    df['new_id'] = df['processo_id'].astype(str) + '0' + df['documento_id'].astype(str)
    df['new_id'] = df['new_id'].astype(int)
    df_test['new_id'] = df_test['processo_id'].astype(str) + '0' + df_test['documento_id'].astype(str)
    df_test['new_id'] = df_test['new_id'].astype(int)

    df.rename(columns={'texto_tratado':'text'},inplace=True)
    df_test.rename(columns={'texto_tratado':'text'},inplace=True)
    df.set_index(['new_id'],drop=True,inplace=True)
    df_test.set_index(['new_id'],drop=True,inplace=True)
    
    cols = ['text','assuntoPrincipal']

    df = df[cols].drop_duplicates()
    df_test = df_test[cols].drop_duplicates()

    df = pd.concat([df,df_test],ignore_index=True)

    df = df.sample(frac=1)

    multiLabel = np.unique(df['assuntoPrincipal']).tolist()
    multiLabel = [sub.replace('+', '') for sub in multiLabel]
    newLabels = []

    for label in multiLabel:
        splitLabel = label.split(":")
        newLabels.append(splitLabel)

    newLabels = [item for sublist in newLabels for item in sublist]
    newLabels = np.unique(np.array(newLabels)).tolist()
    numberOfClasses = len(newLabels)
    newLabels.sort()

    multiIdLabelDic = {idx:label for idx, label in enumerate(newLabels)}
    multiLabelIdDic = {label:idx for idx, label in enumerate(newLabels)}

    id2label = {idx:int(label) for idx, label in enumerate(newLabels)}
    label2id = {int(label):idx for idx, label in enumerate(newLabels)}

    labelsToExpand = df['assuntoPrincipal'].tolist()
    labelsToExpand = [sub.replace('+', '') for sub in labelsToExpand]

    posArr = []

    for label in labelsToExpand:
        multilabelFinal = [0.0] * numberOfClasses
        splitLabel = label.split(":")

        for splits in splitLabel:
            multilabelFinal[multiLabelIdDic[splits]] = 1.0

        posArr.append(multilabelFinal.copy())

    df['labels'] = posArr

    df.drop(columns=['assuntoPrincipal'],inplace=True)

    X_train, X_testval, y_train, y_testval = train_test_split(df['text'],df['labels'], test_size=test_split)#, stratify= df['labels'])

    del df
    gc.collect()

    XTestval = pd.DataFrame()
    XTestval['labels'] = y_testval.to_frame()
    XTestval['text'] = X_testval.to_frame()

    X_test, X_val, y_test, y_val = train_test_split(XTestval['text'],XTestval['labels'], test_size=test_split)#, stratify= XTestval['labels'] )

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
    
    XTest.to_csv(os.path.join(ROOT_DIR,'data','small_tj','interm','test','test.csv'),index=False)
    XTrain.to_csv(os.path.join(ROOT_DIR,'data','small_tj','interm','train','train.csv'),index=False)
    Xval.to_csv(os.path.join(ROOT_DIR,'data','small_tj','interm','val','val.csv'),index=False)

    with open(os.path.join(ROOT_DIR,'data','small_tj','interm','id2label.json'),'w') as f:
        json.dump(id2label,f)
        f.close()
    with open(os.path.join(ROOT_DIR,'data','small_tj','interm','label2id.json'),'w') as f:
        json.dump(label2id,f)
        f.close()

    return None

def big_tj(test_split:float) -> None:

    path = os.path.join(ROOT_DIR,'data','big_tj','raw','TrainingSet_s34787.csv.xz')
    df = pd.read_csv(path)

    path_test = os.path.join(ROOT_DIR,'data','big_tj','raw','TestSet_s34787.csv.xz')
    df_test = pd.read_csv(path_test)

    pathCTexto_tj = os.path.join(ROOT_DIR,'data','big_tj','raw','dataset1_tj.csv')
    pathCTexto_cnj = os.path.join(ROOT_DIR,'data','big_tj','raw','dataset3_cnj.csv')

    df_c_texto_tj = pd.read_csv(pathCTexto_tj)
    df_c_texto_tj = df_c_texto_tj[['documento_id','processo_id','texto_tratado']]

    #df_c_texto_cnj[df_c_texto_cnj['grupo_assunto'].str.contains("DIREITO DO TRABALHO")].sample(n=50)

    df_c_texto_cnj = pd.read_csv(pathCTexto_cnj)

    #df_c_texto_cnj = df_c_texto_cnj[df_c_texto_cnj['texto_tratado'].str.len() <= 32000]
    #exportar50 = df_c_texto_cnj[df_c_texto_cnj['grupo_assunto'].str.contains("DIREITO DO TRABALHO")].sample(n=50)
    #exportar50.to_csv("amostrar50.csv",sep=",",index=False,encoding='latin',errors="replace")

    df_c_texto_cnj = df_c_texto_cnj[['documento_id','processo_id','texto_tratado']]

    df_c_texto = pd.concat([df_c_texto_tj,df_c_texto_cnj],ignore_index=True)
    df_c_texto.drop_duplicates(inplace=True)

    df = df.merge(df_c_texto,on=['processo_id','documento_id'])
    df_test = df_test.merge(df_c_texto,on=['processo_id','documento_id'])

    df['new_id'] = df['processo_id'].astype(str) + '0' + df['documento_id'].astype(str)
    df['new_id'] = df['new_id'].astype(int)
    df_test['new_id'] = df_test['processo_id'].astype(str) + '0' + df_test['documento_id'].astype(str)
    df_test['new_id'] = df_test['new_id'].astype(int)

    df.rename(columns={'texto_tratado':'text'},inplace=True)
    df_test.rename(columns={'texto_tratado':'text'},inplace=True)
    df.set_index(['new_id'],drop=True,inplace=True)
    df_test.set_index(['new_id'],drop=True,inplace=True)
    
    cols = ['text','assuntoPrincipal']

    df = df[cols].drop_duplicates()
    df_test = df_test[cols].drop_duplicates()

    df['split'] = 'train'
    df_test['split'] = 'test'

    df = pd.concat([df,df_test],ignore_index=True)

    df = df.sample(frac=1)

    multiLabel = np.unique(df['assuntoPrincipal']).tolist()
    multiLabel = [sub.replace('+', '') for sub in multiLabel]
    newLabels = []

    for label in multiLabel:
        splitLabel = label.split(":")
        newLabels.append(splitLabel)

    newLabels = [item for sublist in newLabels for item in sublist]
    newLabels = np.unique(np.array(newLabels)).tolist()
    numberOfClasses = len(newLabels)
    newLabels.sort()

    multiIdLabelDic = {idx:label for idx, label in enumerate(newLabels)}
    multiLabelIdDic = {label:idx for idx, label in enumerate(newLabels)}

    id2label = {idx:int(label) for idx, label in enumerate(newLabels)}
    label2id = {int(label):idx for idx, label in enumerate(newLabels)}

    labelsToExpand = df['assuntoPrincipal'].tolist()
    labelsToExpand = [sub.replace('+', '') for sub in labelsToExpand]

    posArr = []

    for label in labelsToExpand:
        multilabelFinal = [0.0] * numberOfClasses
        splitLabel = label.split(":")

        for splits in splitLabel:
            multilabelFinal[multiLabelIdDic[splits]] = 1.0

        posArr.append(multilabelFinal.copy())

    df['labels'] = posArr

    df.drop(columns=['assuntoPrincipal'],inplace=True)

    X_train, X_testval, y_train, y_testval = train_test_split(df[df['split']=='train']['text'],df[df['split']=='train']['labels'], test_size=0.00000000001)#, stratify= df['labels'])

    #X_train, X_testval, y_train, y_testval = train_test_split(df['text'],df['labels'], test_size=test_split)#, stratify= df['labels'])

    #del df
    #gc.collect()

    #XTestval = pd.DataFrame()
    #XTestval['labels'] = y_testval.to_frame()
    #XTestval['text'] = X_testval.to_frame()

    X_test, X_val, y_test, y_val = train_test_split(df[df['split']=='test']['text'],df[df['split']=='test']['labels'], test_size=test_split)#, stratify= XTestval['labels'] )

    #X_test, X_val, y_test, y_val = train_test_split(XTestval['text'],XTestval['labels'], test_size=test_split)#, stratify= XTestval['labels'] )

    #del XTestval
    #gc.collect()

    XTrain = pd.DataFrame()
    XTrain['labels'] = y_train.to_frame()
    XTrain['text'] = X_train.to_frame()

    XTest = pd.DataFrame()
    XTest['labels'] = y_test.to_frame()
    XTest['text'] = X_test.to_frame()

    Xval = pd.DataFrame()
    Xval['labels'] = y_val.to_frame()
    Xval['text'] = X_val.to_frame()
    
    XTest.to_csv(os.path.join(ROOT_DIR,'data','big_tj','interm','test','test.csv'),index=False)
    XTrain.to_csv(os.path.join(ROOT_DIR,'data','big_tj','interm','train','train.csv'),index=False)
    Xval.to_csv(os.path.join(ROOT_DIR,'data','big_tj','interm','val','val.csv'),index=False)

    with open(os.path.join(ROOT_DIR,'data','big_tj','interm','id2label.json'),'w') as f:
        json.dump(id2label,f)
        f.close()
    with open(os.path.join(ROOT_DIR,'data','big_tj','interm','label2id.json'),'w') as f:
        json.dump(label2id,f)
        f.close()

    return None

def big_tj_single(test_split:float) -> None:

    path = os.path.join(ROOT_DIR,'data','big_tj','raw','TrainingSet_s34787.csv.xz')
    df = pd.read_csv(path)

    path_test = os.path.join(ROOT_DIR,'data','big_tj','raw','TestSet_s34787.csv.xz')
    df_test = pd.read_csv(path_test)

    pathCTexto_tj = os.path.join(ROOT_DIR,'data','big_tj','raw','dataset1_tj.csv')
    pathCTexto_cnj = os.path.join(ROOT_DIR,'data','big_tj','raw','dataset3_cnj.csv')

    df_c_texto_tj = pd.read_csv(pathCTexto_tj)
    df_c_texto_tj = df_c_texto_tj[['documento_id','processo_id','texto_tratado']]

    #df_c_texto_cnj[df_c_texto_cnj['grupo_assunto'].str.contains("DIREITO DO TRABALHO")].sample(n=50)

    df_c_texto_cnj = pd.read_csv(pathCTexto_cnj)

    #df_c_texto_cnj = df_c_texto_cnj[df_c_texto_cnj['texto_tratado'].str.len() <= 32000]
    #exportar50 = df_c_texto_cnj[df_c_texto_cnj['grupo_assunto'].str.contains("DIREITO DO TRABALHO")].sample(n=50)
    #exportar50.to_csv("amostrar50.csv",sep=",",index=False,encoding='latin',errors="replace")

    df_c_texto_cnj = df_c_texto_cnj[['documento_id','processo_id','texto_tratado']]

    df_c_texto = pd.concat([df_c_texto_tj,df_c_texto_cnj],ignore_index=True)
    df_c_texto.drop_duplicates(inplace=True)

    df = df.merge(df_c_texto,on=['processo_id','documento_id'])
    df_test = df_test.merge(df_c_texto,on=['processo_id','documento_id'])

    df['new_id'] = df['processo_id'].astype(str) + '0' + df['documento_id'].astype(str)
    df['new_id'] = df['new_id'].astype(int)
    df_test['new_id'] = df_test['processo_id'].astype(str) + '0' + df_test['documento_id'].astype(str)
    df_test['new_id'] = df_test['new_id'].astype(int)

    df.rename(columns={'texto_tratado':'text'},inplace=True)
    df_test.rename(columns={'texto_tratado':'text'},inplace=True)
    df.set_index(['new_id'],drop=True,inplace=True)
    df_test.set_index(['new_id'],drop=True,inplace=True)
    
    cols = ['text','assuntoPrincipal']

    df = df[cols].drop_duplicates()
    df_test = df_test[cols].drop_duplicates()

    df['split'] = 'train'
    df_test['split'] = 'test'

    df = pd.concat([df,df_test],ignore_index=True)

    df = df.sample(frac=1)

    df = _deparaCodigos(df)

    unique, counts = np.unique(np.array(df['assuntoPrincipal']), return_counts=True)
    arr1inds = counts.argsort()
    sorted_arr2 = unique[arr1inds[::-1]]

    max_classes = len(counts) - len(counts[counts <= 25]) # tirando exemplos com menos de 25 casos

    problemList = sorted_arr2[:max_classes]

    df = df[df['assuntoPrincipal'].isin(problemList)]

    id2label = {idx:label for idx, label in enumerate(problemList.tolist())}
    label2id = {label:idx for idx, label in enumerate(problemList.tolist())}

    df['labels'] = df['assuntoPrincipal'].apply(lambda row: label2id[row])

    df.drop(columns=['assuntoPrincipal'],inplace=True)

    #X_train, X_testval, y_train, y_testval = train_test_split(df[df['split']=='train']['text'],df[df['split']=='train']['labels'], test_size=0.00000000001)#, stratify= df['labels'])

    X_train, X_testval, y_train, y_testval = train_test_split(df['text'],df['labels'], test_size=test_split, stratify= df['labels'])

    del df
    gc.collect()

    XTestval = pd.DataFrame()
    XTestval['labels'] = y_testval.to_frame()
    XTestval['text'] = X_testval.to_frame()

    #X_test, X_val, y_test, y_val = train_test_split(df[df['split']=='test']['text'],df[df['split']=='test']['labels'], test_size=test_split)#, stratify= XTestval['labels'] )

    X_test, X_val, y_test, y_val = train_test_split(XTestval['text'],XTestval['labels'], test_size=test_split)#, stratify= XTestval['labels'] )

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
    
    XTest.to_csv(os.path.join(ROOT_DIR,'data','big_tj_single','interm','test','test.csv'),index=False)
    XTrain.to_csv(os.path.join(ROOT_DIR,'data','big_tj_single','interm','train','train.csv'),index=False)
    Xval.to_csv(os.path.join(ROOT_DIR,'data','big_tj_single','interm','val','val.csv'),index=False)

    with open(os.path.join(ROOT_DIR,'data','big_tj_single','interm','id2label.json'),'w') as f:
        json.dump(id2label,f)
        f.close()
    with open(os.path.join(ROOT_DIR,'data','big_tj_single','interm','label2id.json'),'w') as f:
        json.dump(label2id,f)
        f.close()

    return None

def big_tj_single_mixed(test_split:float) -> None:

    path = os.path.join(ROOT_DIR,'data','big_tj','raw','TrainingSet_s34787.csv.xz')
    df = pd.read_csv(path)

    path_test = os.path.join(ROOT_DIR,'data','big_tj','raw','TestSet_s34787.csv.xz')
    df_test = pd.read_csv(path_test)

    pathCTexto_tj = os.path.join(ROOT_DIR,'data','big_tj','raw','dataset1_tj.csv')
    pathCTexto_cnj = os.path.join(ROOT_DIR,'data','big_tj','raw','dataset3_cnj.csv')

    df_c_texto_tj = pd.read_csv(pathCTexto_tj)
    df_c_texto_tj = df_c_texto_tj[['documento_id','processo_id','texto_tratado']]

    df_c_texto_cnj = pd.read_csv(pathCTexto_cnj)

    df_c_texto_cnj = df_c_texto_cnj[['documento_id','processo_id','texto_tratado']]

    df_c_texto = pd.concat([df_c_texto_tj,df_c_texto_cnj],ignore_index=True)
    df_c_texto.drop_duplicates(inplace=True)

    df = df.merge(df_c_texto,on=['processo_id','documento_id'])
    df_test = df_test.merge(df_c_texto,on=['processo_id','documento_id'])

    df['new_id'] = df['processo_id'].astype(str) + '0' + df['documento_id'].astype(str)
    df['new_id'] = df['new_id'].astype(int)
    df_test['new_id'] = df_test['processo_id'].astype(str) + '0' + df_test['documento_id'].astype(str)
    df_test['new_id'] = df_test['new_id'].astype(int)

    df.rename(columns={'texto_tratado':'text'},inplace=True)
    df_test.rename(columns={'texto_tratado':'text'},inplace=True)
    df.set_index(['new_id'],drop=True,inplace=True)
    df_test.set_index(['new_id'],drop=True,inplace=True)
    
    cols = ['text','assuntoPrincipal']

    df = df[cols].drop_duplicates()
    df_test = df_test[cols].drop_duplicates()

    df['split'] = 'train'
    df_test['split'] = 'test'

    df = pd.concat([df,df_test],ignore_index=True)

    df = df.sample(frac=1)

    df = df[df['text'].str.len() >= 10*2048]

    #len(df[df['text'].str.len() >= 10*1024])
    #43766
    #len(df[df['text'].str.len() >= 10*2048])
    #23436
    #len(df[df['text'].str.len() >= 10*4096])
    #7137

    unique, counts = np.unique(np.array(df['assuntoPrincipal']), return_counts=True)
    arr1inds = counts.argsort()
    sorted_arr2 = unique[arr1inds[::-1]]

    problemList = sorted_arr2[:]

    id2label = {idx:label for idx, label in enumerate(problemList.tolist())}
    label2id = {label:idx for idx, label in enumerate(problemList.tolist())}

    df['labels'] = df['assuntoPrincipal'].apply(lambda row: label2id[row])

    df.drop(columns=['assuntoPrincipal'],inplace=True)

    X_train, X_testval, y_train, y_testval = train_test_split(df['text'],df['labels'], test_size=test_split)#, stratify= df['labels'])

    del df
    gc.collect()

    XTestval = pd.DataFrame()
    XTestval['labels'] = y_testval.to_frame()
    XTestval['text'] = X_testval.to_frame()

    X_test, X_val, y_test, y_val = train_test_split(XTestval['text'],XTestval['labels'], test_size=test_split)#, stratify= XTestval['labels'] )

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
    
    XTest.to_csv(os.path.join(ROOT_DIR,'data','big_tj_single_mixed','interm','test','test.csv'),index=False)
    XTrain.to_csv(os.path.join(ROOT_DIR,'data','big_tj_single_mixed','interm','train','train.csv'),index=False)
    Xval.to_csv(os.path.join(ROOT_DIR,'data','big_tj_single_mixed','interm','val','val.csv'),index=False)

    with open(os.path.join(ROOT_DIR,'data','big_tj_single_mixed','interm','id2label.json'),'w') as f:
        json.dump(id2label,f)
        f.close()
    with open(os.path.join(ROOT_DIR,'data','big_tj_single_mixed','interm','label2id.json'),'w') as f:
        json.dump(label2id,f)
        f.close()

    return None

def small_tj_single(test_split:float) -> None:

    path = os.path.join(ROOT_DIR,'data','small_tj','raw','TrainingSet_s34787.csv.xz')
    df = pd.read_csv(path)

    path_test = os.path.join(ROOT_DIR,'data','small_tj','raw','TestSet_s34787.csv.xz')
    df_test = pd.read_csv(path_test)

    pathCTexto_tj = os.path.join(ROOT_DIR,'data','tj','raw','dataset_tjce_v1_trata_texto.csv.zip')
    #pathCTexto_cnj = os.path.join(ROOT_DIR,'data','big_tj','raw','dataset3_cnj.csv')

    df_c_texto_tj = pd.read_csv(pathCTexto_tj)
    df_c_texto_tj = df_c_texto_tj[['documento_id','processo_id','texto_tratado']]

    #df_c_texto_cnj[df_c_texto_cnj['grupo_assunto'].str.contains("DIREITO DO TRABALHO")].sample(n=50)

    #df_c_texto_cnj = pd.read_csv(pathCTexto_cnj)

    #df_c_texto_cnj = df_c_texto_cnj[df_c_texto_cnj['texto_tratado'].str.len() <= 32000]
    #exportar50 = df_c_texto_cnj[df_c_texto_cnj['grupo_assunto'].str.contains("DIREITO DO TRABALHO")].sample(n=50)
    #exportar50.to_csv("amostrar50.csv",sep=",",index=False,encoding='latin',errors="replace")

    #df_c_texto_cnj = df_c_texto_cnj[['documento_id','processo_id','texto_tratado']]

    df_c_texto = df_c_texto_tj
    df_c_texto.drop_duplicates(inplace=True)

    df = df.merge(df_c_texto,on=['processo_id','documento_id'])
    df_test = df_test.merge(df_c_texto,on=['processo_id','documento_id'])

    df['new_id'] = df['processo_id'].astype(str) + '0' + df['documento_id'].astype(str)
    df['new_id'] = df['new_id'].astype(int)
    df_test['new_id'] = df_test['processo_id'].astype(str) + '0' + df_test['documento_id'].astype(str)
    df_test['new_id'] = df_test['new_id'].astype(int)

    df.rename(columns={'texto_tratado':'text'},inplace=True)
    df_test.rename(columns={'texto_tratado':'text'},inplace=True)
    df.set_index(['new_id'],drop=True,inplace=True)
    df_test.set_index(['new_id'],drop=True,inplace=True)
    
    cols = ['text','assuntoPrincipal']

    df = df[cols].drop_duplicates()
    df_test = df_test[cols].drop_duplicates()

    df['split'] = 'train'
    df_test['split'] = 'test'

    df = pd.concat([df,df_test],ignore_index=True)

    df = df.sample(frac=1)

    df = _deparaCodigos(df)

    unique, counts = np.unique(np.array(df['assuntoPrincipal']), return_counts=True)
    arr1inds = counts.argsort()
    sorted_arr2 = unique[arr1inds[::-1]]

    max_classes = len(counts) - len(counts[counts <= 10]) # tirando exemplos com menos de 10 casos

    problemList = sorted_arr2[:max_classes]

    df = df[df['assuntoPrincipal'].isin(problemList)]

    id2label = {idx:label for idx, label in enumerate(problemList.tolist())}
    label2id = {label:idx for idx, label in enumerate(problemList.tolist())}

    df['labels'] = df['assuntoPrincipal'].apply(lambda row: label2id[row])

    df.drop(columns=['assuntoPrincipal'],inplace=True)

    X_train, X_testval, y_train, y_testval = train_test_split(df['text'],df['labels'], test_size=0.2, stratify= df['labels'])

    #X_train, X_testval, y_train, y_testval = train_test_split(df['text'],df['labels'], test_size=test_split)#, stratify= df['labels'])

    del df
    gc.collect()

    XTestval = pd.DataFrame()
    XTestval['labels'] = y_testval.to_frame()
    XTestval['text'] = X_testval.to_frame()

    X_test, X_val, y_test, y_val = train_test_split(XTestval['text'],XTestval['labels'], test_size=test_split)#, stratify= XTestval['labels'] )

    #X_test, X_val, y_test, y_val = train_test_split(XTestval['text'],XTestval['labels'], test_size=test_split)#, stratify= XTestval['labels'] )

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
    
    XTest.to_csv(os.path.join(ROOT_DIR,'data','small_tj_single','interm','test','test.csv'),index=False)
    XTrain.to_csv(os.path.join(ROOT_DIR,'data','small_tj_single','interm','train','train.csv'),index=False)
    Xval.to_csv(os.path.join(ROOT_DIR,'data','small_tj_single','interm','val','val.csv'),index=False)

    with open(os.path.join(ROOT_DIR,'data','small_tj_single','interm','id2label.json'),'w') as f:
        json.dump(id2label,f)
        f.close()
    with open(os.path.join(ROOT_DIR,'data','small_tj_single','interm','label2id.json'),'w') as f:
        json.dump(label2id,f)
        f.close()

    return None

def small_tj_single_cross(test_split:float) -> None:

    path = os.path.join(ROOT_DIR,'data','small_tj','raw','TrainingSet_s34787.csv.xz')
    df = pd.read_csv(path)

    path_test = os.path.join(ROOT_DIR,'data','small_tj','raw','TestSet_s34787.csv.xz')
    df_test = pd.read_csv(path_test)

    ####################PEGAR LEGENDAS#########################
    #aaa = pd.read_csv(os.path.join(ROOT_DIR,'data','big_tj','raw','dataset1_tj.csv'))
    #bbb = pd.read_csv(os.path.join(ROOT_DIR,'data','big_tj','raw','dataset3_cnj.csv'))
    #ccc = pd.read_csv(os.path.join(ROOT_DIR,'data','big_tj','raw','dataset_340k.csv'))
    #aaa = aaa[['grupo_Hid','grupo','assunto_Hid','assunto','assuntoPrincipal','assunto_Hid','assuntos_id','grupo_assunto','assunto_folha']].drop_duplicates()
    #bbb = bbb[['grupo_Hid','grupo','assunto_Hid','assunto','assuntoPrincipal','id_assunto','assuntos_id','grupo_assunto','assunto_folha']].drop_duplicates()
    #ccc = ccc[['grupo_Hid','grupo','assuntos_id','assunto_folha']].drop_duplicates()
    #aaa.to_csv('aa.csv')
    #bbb.to_csv('bb.csv')
    #ccc.to_csv('cc.csv')
    ###########################################################

    pathCTexto_tj = os.path.join(ROOT_DIR,'data','tj','raw','dataset_tjce_v1_trata_texto.csv.zip')
    #pathCTexto_cnj = os.path.join(ROOT_DIR,'data','big_tj','raw','dataset3_cnj.csv')

    df_c_texto_tj = pd.read_csv(pathCTexto_tj)
    df_c_texto_tj = df_c_texto_tj[['documento_id','processo_id','texto_tratado']]

    #df_c_texto_cnj[df_c_texto_cnj['grupo_assunto'].str.contains("DIREITO DO TRABALHO")].sample(n=50)

    #df_c_texto_cnj = pd.read_csv(pathCTexto_cnj)

    #df_c_texto_cnj = df_c_texto_cnj[df_c_texto_cnj['texto_tratado'].str.len() <= 32000]
    #exportar50 = df_c_texto_cnj[df_c_texto_cnj['grupo_assunto'].str.contains("DIREITO DO TRABALHO")].sample(n=50)
    #exportar50.to_csv("amostrar50.csv",sep=",",index=False,encoding='latin',errors="replace")

    #df_c_texto_cnj = df_c_texto_cnj[['documento_id','processo_id','texto_tratado']]

    df_c_texto = df_c_texto_tj
    df_c_texto.drop_duplicates(inplace=True)

    df = df.merge(df_c_texto,on=['processo_id','documento_id'])
    df_test = df_test.merge(df_c_texto,on=['processo_id','documento_id'])

    df['new_id'] = df['processo_id'].astype(str) + '0' + df['documento_id'].astype(str)
    df['new_id'] = df['new_id'].astype(int)
    df_test['new_id'] = df_test['processo_id'].astype(str) + '0' + df_test['documento_id'].astype(str)
    df_test['new_id'] = df_test['new_id'].astype(int)

    df.rename(columns={'texto_tratado':'text'},inplace=True)
    df_test.rename(columns={'texto_tratado':'text'},inplace=True)
    df.set_index(['new_id'],drop=True,inplace=True)
    df_test.set_index(['new_id'],drop=True,inplace=True)
    
    cols = ['text','assuntoPrincipal']

    df = df[cols].drop_duplicates()
    df_test = df_test[cols].drop_duplicates()

    df['split'] = 'train'
    df_test['split'] = 'test'

    df = pd.concat([df,df_test],ignore_index=True)

    df = df.sample(frac=1)

    #colocar para todos os q tem TJ no nome.
    df = _deparaCodigos(df)

    unique, counts = np.unique(np.array(df['assuntoPrincipal']), return_counts=True)
    arr1inds = counts.argsort()
    sorted_arr2 = unique[arr1inds[::-1]]

    max_classes = len(counts) - len(counts[counts <= 10]) # tirando exemplos com menos de 10 casos

    problemList = sorted_arr2[:max_classes]

    df = df[df['assuntoPrincipal'].isin(problemList)]

    id2label = {idx:label for idx, label in enumerate(problemList.tolist())}
    label2id = {label:idx for idx, label in enumerate(problemList.tolist())}

    df['labels'] = df['assuntoPrincipal'].apply(lambda row: label2id[row])

    df.drop(columns=['assuntoPrincipal'],inplace=True)

    skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

    crossValDfs_Train = []
    crossValDfs_Test = []

    X_train, X_val, y_train, y_val = train_test_split(df['text'],df['labels'], test_size=0.05, stratify= df['labels'])

    X_sem_val = pd.DataFrame()
    X_sem_val['labels'] = y_train.to_frame()
    X_sem_val['text'] = X_train.to_frame()
    X_sem_val['doc_id'] = X_sem_val.index

    X_sem_val = X_sem_val.reset_index(drop=True)

    for i, (train_index, test_index) in enumerate(skf.split(X_sem_val['text'], X_sem_val['labels'])):
        tmpDfTrain = X_sem_val[X_sem_val.index.isin(train_index)].copy()
        tmpDfTrain['split'] = 'Train'
        tmpDfTrain['cross'] = i
        tmpDfTest = X_sem_val[X_sem_val.index.isin(test_index)].copy()
        tmpDfTest['split'] = 'Test'
        tmpDfTest['cross'] = i
        crossValDfs_Train.append(tmpDfTrain.copy())
        crossValDfs_Test.append(tmpDfTest.copy())
        tmpDfTrain = None
        tmpDfTest = None

    allCrossTrain = pd.concat(crossValDfs_Train,ignore_index=True)
    allCrossTest = pd.concat(crossValDfs_Test,ignore_index=True)

    X_sem_val.set_index(X_sem_val['doc_id'],inplace=True)
    X_sem_val.drop(columns=['doc_id'],inplace=True)


    ############################# FALTA SO A EXPORTACAO DOS ARQUIVOS #########################################

    XTrain = pd.DataFrame()
    XTrain['labels'] = y_train.to_frame()
    XTrain['text'] = X_train.to_frame()

    XTest = pd.DataFrame()
    XTest['labels'] = y_test.to_frame()
    XTest['text'] = X_test.to_frame()

    Xval = pd.DataFrame()
    Xval['labels'] = y_val.to_frame()
    Xval['text'] = X_val.to_frame()
    
    XTest.to_csv(os.path.join(ROOT_DIR,'data','small_tj_single','interm','test','test.csv'),index=False)
    XTrain.to_csv(os.path.join(ROOT_DIR,'data','small_tj_single','interm','train','train.csv'),index=False)
    Xval.to_csv(os.path.join(ROOT_DIR,'data','small_tj_single','interm','val','val.csv'),index=False)

    with open(os.path.join(ROOT_DIR,'data','small_tj_single','interm','id2label.json'),'w') as f:
        json.dump(id2label,f)
        f.close()
    with open(os.path.join(ROOT_DIR,'data','small_tj_single','interm','label2id.json'),'w') as f:
        json.dump(label2id,f)
        f.close()

    return None