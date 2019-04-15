
import nltk
import numpy as np
import tflearn
import tensorflow as tf
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
#svmClassifier = svm.SVC(kernel='linear', C = 1.0,probability=True)


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
    actualMessage = ''
    previousContext = ''
    if 'userMessage' in request.args:
        actualMessage = request.args['userMessage']
    if 'previousContext' in request.args:
        previousContext = request.args['previousContext']
    print(actualMessage)

    message = cleanText(actualMessage)
    test_Y = vectorizer.transform([message])
    #test_Y = tfTransformer.transform(test_X)

    ft = vectorizer.get_feature_names()
    result = list(map(lambda row:dict(zip(ft,row)),test_Y.toarray()))
    #print(result)
    #print(list(result[0].values()))
    result_array = model.predict([list(result[0].values())]).tolist()[0]
    for item,score in zip(classes,result_array):
        print(item+': '+str(score))
    print(classes)
    if max(result_array) > 0.4:
        intentTag = classes[result_array.index(max(result_array))]
    else:
        intentTag = 'unknown'

    #predicted_svm = svmClassifier.predict(test_Y)
    #intentTag = predicted_svm[0]

    for intent in intents:
        if intent['tag'] == intentTag:
            responseMsg = intent['responses'][0]
            #print("Response:  "+responseMsg)
            tag = intent['tag']
            print("Tag:  "+tag)
            context = intent['context']
            print("Context:  "+context)
            dataType = intent['type']
            print("Data Type:  "+dataType)

    #if tag == 'unknown' or tag == 'master' or tag == 'bachelor':
    pos_tokens = nltk.pos_tag(nltk.word_tokenize(actualMessage.lower()))
    print(pos_tokens)
    print(previousContext)
    if previousContext == 'master':
        for word,token in pos_tokens:
            if((token == 'WRB' and word=='where') or (word=='location')):
                responseMsg = 'Florida, Tampa, United Status'
            elif(token=='WDT'):
                responseMsg = 'University of South Florida'
            elif((token == 'WRB' and word=='when') or (word == 'year')) :
                responseMsg = 'May 2020'
            elif(word == 'gpa'):
                responseMsg = '3.91 out of 4.0'
        tag = previousContext
    if previousContext == 'bachelor':
        for word,token in pos_tokens:
            if((token == 'WRB' and word=='where') or (word=='location')):
                responseMsg = 'Trichy, India'
            elif(token=='WDT'):
                responseMsg = 'Anna University'
            elif((token == 'WRB' and word=='when') or (word == 'year')):
                responseMsg = 'May, 2014'
            elif(word == 'gpa'):
                responseMsg = '7.54 out of 10.0'
        tag = previousContext

    print("Response:  "+responseMsg)

    #test_X = vectorizer.transform([message])
    #test_Y = tfTransformer.transform(test_X)
    #predicted_svm = svmClassifier.predict(test_Y)

    #ft = vectorizer.get_feature_names()
    #result = list(map(lambda row:dict(zip(ft,row)),test_Y.toarray()))
    #result_array = model.predict([list(result[0].values())]).tolist()[0]
    #returnClass = classes[result_array.index(max(result_array))]
    #return (responseMsg,tag)
    #return jsonify({"result":predicted_svm[0]})
    result = []
    result.append(responseMsg)
    result.append(tag)
    return jsonify({"result":result})


def loadIntents():
    with open('profile.json') as json_data:
        intents = json.load(json_data)
    return intents


def cleanText(summary):
    summary = summary.replace('*','').replace('-',' ').replace('/',' ').replace("'",' ')
    #tokens_summary = [str.lower().strip(string.punctuation) for str in summary.split() if str not in stopwords]
    tokens_summary = [str.lower().strip(string.punctuation) for str in summary.split()]
    lemma_summary = [lemmatizer.lemmatize(token) for token in tokens_summary if len(token) > 0]
    return(' '.join(word for word in lemma_summary))

def loadData():
    classes = []
    # loop through each sentence in our intents patterns
    intents = loadIntents()['intents']
    for intent in intents:
        for pattern in intent['patterns']:
            documents.append((pattern, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    classes = sorted(list(set(classes)))

    output_empty = [0] * len(classes)
    for doc in documents:
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        output.append(output_row)
    return (classes,intents)


if __name__ == "__main__":
    handler = RotatingFileHandler('foo.log', maxBytes=10000, backupCount=1)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)

    try:
        classes, intents = loadData()
        Y = vectorizer.fit_transform([cleanText(pattern) for pattern,tag in documents])
        #Y = tfTransformer.fit_transform(X)

        # reset underlying graph data
        tf.reset_default_graph()
        # Build neural network
        net = tflearn.input_data(shape=[None,len(Y.toarray().tolist()[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
        net = tflearn.regression(net)

        # Define model and setup tensorboard
        model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
        # Start training (apply gradient descent algorithm)
        model.fit(Y.toarray().tolist(), output, n_epoch=1000, batch_size=8)
        #model.save('model.tflearn')

        #if(os.path.exists(modelPickleFileName) != True):
            #pickle_file = open(modelPickleFileName,'wb')
            #svmClassifier.fit(Y, list(tag for pattern,tag in documents))
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

            #pickle.dump(svmClassifier,pickle_file)
            #pickle_file.close()
        #else:
            #unpickle_file = open(modelPickleFileName,'rb')
            #svmClassifier = pickle.load(unpickle_file)
            #unpickle_file.close()
        app.run(host='0.0.0.0',port=5001)
    finally:
        print("Exit")
