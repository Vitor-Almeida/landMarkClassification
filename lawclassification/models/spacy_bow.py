import spacy
import pandas as pd
import xgboost as xgb
import random
import gc
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


def main():

    df = pd.read_csv("/home/jaco/Projetos/landMarkClassification/data/spacy_data/amazon_alexa.tsv", sep="\t")
    #df = df.head(1000)

    problemColumn = 'feedback'

    problemList = df['feedback'].to_list()
    max_classes = 5
    df = df[['feedback','verified_reviews']]

    dfLength = len(df)
    testFrac = 0.2
    testValSize = int(testFrac*dfLength)

    trainSize = dfLength - testValSize
    valSize = int(testValSize*testFrac)
    testSize = testValSize - valSize

    allIndices = [idx for idx in range(0,dfLength)]
    random.shuffle(allIndices)
    trainIndices = random.sample(range(0,dfLength),trainSize)
    testIndices = [idx for idx in allIndices if idx not in trainIndices]
    valIndices = [idx for idx in allIndices if idx not in trainIndices and not idx in testIndices]

    unique, counts = np.unique(np.array(problemList), return_counts=True)
    arr1inds = counts.argsort()
    sorted_arr2 = unique[arr1inds[::-1]]
    problemList = sorted_arr2[:max_classes]
    df = df[df[problemColumn].isin(problemList)]

    id2label = {idx:label for idx, label in enumerate(problemList.tolist())}
    label2id = {label:idx for idx, label in enumerate(problemList.tolist())}
    df['labels'] = df[problemColumn].apply(lambda row: label2id[row])

    df.rename(columns={'verified_reviews':'text'},inplace=True)
    df = df[['labels','text']]

    # Create our list of stopwords
    nlp = spacy.load('en_core_web_lg')

    # Creating our tokenizer function
    def spacy_tokenizer(sentence):

        newSentence = []

        for word in nlp(sentence):
            if (not word.is_stop or not word.is_punct or not word.is_oov) and (word.is_alpha and word.is_ascii):
                newSentence.append(word.lemma_.lower())

        return newSentence

    #bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
    tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer,ngram_range=(1,1))

    DMatrix = xgb.DMatrix(tfidf_vector.fit_transform(df['text']),
                          label=df['labels'],
                          feature_names=tfidf_vector.get_feature_names_out())

    DMatrixTrain = DMatrix.slice(trainIndices)
    DMatrixTest = DMatrix.slice(testIndices)
    DMatrixVal = DMatrix.slice(valIndices)

    #DMatrixTrain.save_binary('train.buffer')
    #DMatrixTest.save_binary('test.buffer')
    #DMatrixVal.save_binary('val.buffer')

    del DMatrix
    del df
    gc.collect()
    
    evallist = [(DMatrixTest, 'test'), (DMatrixTrain, 'train')]

    #classificadores:
    #multi:softmax => set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes)
    #mlogloss => logloss multiclass
    #binary:logistic => logistic regression for binary classification, output probability
    #logloss

    boosterParams = {'max_depth': 2, #more, more overfit
                    'min_split_loss' : 0, #larger, less overfitting
                    'learning_rate': 0.3, #larger, less overfitting
                    'objective': 'binary:logistic', 
                    'reg_lambda':1, #larger, less overfitting, L2 regu
                    'alpha':1, #larger, less overfitting, L1 regu
                    'num_class':1, #colocar aqui quando for multi-label
                    'max_bin' : 256, #Increasing this number improves the optimality of splits at the cost of higher computation time.
                    'tree_method': 'gpu_hist',
                    'predictor': 'gpu_predictor',
                    'eval_metric': 'logloss'}

    evals_result = {}

    bst = xgb.train(params=boosterParams,
                    dtrain=DMatrixTrain,
                    num_boost_round=1000,
                    evals=evallist,
                    early_stopping_rounds=10,
                    evals_result=evals_result)

    preds = bst.predict(DMatrixTest)
    labels = DMatrixTest.get_label()
    print('error=%f' %
        (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) /
        float(len(preds))))

    print('cu')

    return None

if __name__ == '__main__':
    main()
    #bunda()