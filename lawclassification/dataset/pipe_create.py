import pandas as pd
import os
import json
import gc
import numpy as np
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
    XTrain.to_csv(os.path.join(ROOT_DIR,'data','yelp','interm','test','train.csv'),index=False)
    Xval.to_csv(os.path.join(ROOT_DIR,'data','yelp','interm','test','val.csv'),index=False)

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

def echr_build(problem_column):

    #https://archive.org/details/ECHR-ACL2019

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
    problemList = np.unique(np.array(problemList)).tolist()
    max_label_num = len(problemList)

    id2label = {idx:label for idx, label in enumerate(problemList)}
    label2id = {label:idx for idx, label in enumerate(problemList)}

    df = df[[problem_column,'TEXT','FOLDER']]

    #### multi label encoding:
    def encoding_multi_labels(arr,label2id,max_label_num):

        id2label_arr = [0] * max_label_num

        for n in arr:
            id2label_arr[label2id[n]]=1

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