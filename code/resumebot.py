
import nltk
import numpy as np
#import tflearn
#import tensorflow as tf
import random
import nltk.corpus
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from flask import Flask, request, Response, render_template, jsonify
import json
import string
import pickle
from os import listdir
import os
import logging
from logging.handlers import RotatingFileHandler
from sklearn import svm

app = Flask(__name__)
vectorizer = CountVectorizer(ngram_range=(1,2),token_pattern=r'\b\w+\b',min_df=0)
tfTransformer = TfidfTransformer(use_idf = True)
svmClassifier = svm.SVC(kernel='linear', C = 1.0,probability=True)


stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(string.punctuation)
stopwords.remove('not')
lemmatizer = WordNetLemmatizer()
classes = []
documents = []
modelPickleFileName = 'resumeBot.pkl'
output = []


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/getResponse',methods=['GET'])
def getResponse():
    message = ''
    if 'userMessage' in request.args:
        message = request.args['userMessage']
    print(message)
    test_X = vectorizer.transform([message])
    test_Y = tfTransformer.transform(test_X)
    predicted_svm = svmClassifier.predict(test_Y)

    #ft = vectorizer.get_feature_names()
    #result = list(map(lambda row:dict(zip(ft,row)),test_Y.toarray()))
    #result_array = model.predict([list(result[0].values())]).tolist()[0]
    #returnClass = classes[result_array.index(max(result_array))]

    return jsonify({"result":predicted_svm[0]})


def loadIntents():
    with open('intents.json') as json_data:
        intents = json.load(json_data)
    return intents


def cleanText(summary):
    summary = summary.replace('*','').replace('-',' ').replace('/',' ').replace("'",' ')
    tokens_summary = [str.lower().strip(string.punctuation) for str in summary.split() if str not in stopwords]
    lemma_summary = [lemmatizer.lemmatize(token) for token in tokens_summary if len(token) > 0]
    return(' '.join(word for word in lemma_summary))

def loadData():
    classes = []
    # loop through each sentence in our intents patterns
    for intent in loadIntents()['intents']:
        for pattern in intent['patterns']:
            documents.append((pattern, intent['tag']))
            # add to our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    classes = sorted(list(set(classes)))

    output_empty = [0] * len(classes)
    for doc in documents:
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        output.append(output_row)



if __name__ == "__main__":
    handler = RotatingFileHandler('foo.log', maxBytes=10000, backupCount=1)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)

    try:
        loadData()
        X = vectorizer.fit_transform([pattern for pattern,tag in documents])
        Y = tfTransformer.fit_transform(X)
        if(os.path.exists(modelPickleFileName) != True):
            pickle_file = open(modelPickleFileName,'wb')
            svmClassifier.fit(Y, list(tag for pattern,tag in documents))
            #tf.reset_default_graph()
            #net = tflearn.input_data(shape=[None, len(vectorizer.get_feature_names())])
            #net = tflearn.fully_connected(net, 8)
            #net = tflearn.fully_connected(net, 8)
            #net = tflearn.fully_connected(net, len(classes), activation='softmax')
            #net = tflearn.regression(net)

            # Define model and setup tensorboard
            #model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
            # Start training (apply gradient descent algorithm)
            #print(Y.toarray().shape)
            #print(output)
            #model.fit(Y.toarray(), output, n_epoch=1000, batch_size=8, show_metric=True)
            #model.save('model.tflearn')

            pickle.dump(svmClassifier,pickle_file)
            pickle_file.close()
        else:
            unpickle_file = open(modelPickleFileName,'rb')
            svmClassifier = pickle.load(unpickle_file)
            unpickle_file.close()
        app.run(host='0.0.0.0',port=5001)
    finally:
        print("Exit")
