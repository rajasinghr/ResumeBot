
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
import sqlite3
import datetime
import time

app = Flask(__name__)
vectorizer = CountVectorizer(ngram_range=(1,2),token_pattern=r'\b\w+\b',min_df=0)
tfTransformer = TfidfTransformer(use_idf = True)
svmClassifier = svm.SVC(kernel='linear', C = 1.0,probability=True)


stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(string.punctuation)
ignoreWords = ['not','i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'when', 'where', 'why', 'how', 'don', "don't", 'should', "should've", "?"]
for word in ignoreWords:
    stopwords.remove(word)
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
    sessionId = ''
    if 'userMessage' in request.args:
        actualMessage = request.args['userMessage']
    if 'previousContext' in request.args:
        previousContext = request.args['previousContext']
    if 'sessionId' in request.args:
        sessionId = request.args['sessionId']
        print(sessionId)
    print(actualMessage)

    message = cleanText(actualMessage)
    test_X = vectorizer.transform([message])
    test_Y = tfTransformer.transform(test_X)
    vectorizer._validate_vocabulary()
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
    #print(svmClassifier.predict_proba(test_Y))
    #print(svmClassifier.classes_)

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

    if intentTag == 'unknown':
        context = previousContext
    if (intentTag == 'unknown' and previousContext == 'master') or (context == 'master'):
        for word,token in pos_tokens:
            if((token == 'WRB' and word=='where') or (word=='location')):
                responseMsg = intents[5]['specifics'][0]['location']
            elif(token=='WDT'):
                responseMsg = intents[5]['specifics'][0]['university']
            elif((token == 'WRB' and word=='when') or (word == 'year')) :
                responseMsg = intents[5]['specifics'][0]['year']
            elif(word == 'gpa'):
                responseMsg = intents[5]['specifics'][0]['gpa']
            elif(word == 'major'):
                responseMsg = intents[5]['specifics'][0]['major']

    if (intentTag == 'unknown' and previousContext == 'bachelor') or (context == 'bachelor'):
        for word,token in pos_tokens:
            if((token == 'WRB' and word=='where') or (word=='location')):
                responseMsg = intents[4]['specifics'][0]['location']
            elif(token=='WDT'):
                responseMsg = intents[4]['specifics'][0]['university']
            elif((token == 'WRB' and word=='when') or (word == 'year')) :
                responseMsg = intents[4]['specifics'][0]['year']
            elif(word == 'gpa'):
                responseMsg = intents[4]['specifics'][0]['gpa']
            elif(word == 'major'):
                responseMsg = intents[4]['specifics'][0]['major']

    if (intentTag == 'unknown' and previousContext == 'skills') or (context == 'skills'):
        if "machine learning" in message:
            responseMsg = intents[7]['specifics'][0]['machine learning'][0]
        elif "big data" in message:
            responseMsg = intents[7]['specifics'][0]['big data'][0]
        elif "reporting" in message:
            responseMsg = intents[7]['specifics'][0]['reporting'][0]

    if (intentTag == 'unknown' and previousContext == 'past_experience') or (context == 'past_experience'):
        for word,token in pos_tokens:
            if((token == 'WRB' and word=='where') or (word=='location')):
                responseMsg = intents[8]['specifics'][0]['location'][0]
            elif(token=='WDT'):
                responseMsg = intents[8]['specifics'][0]['university'][0]
            elif(((token == 'WRB' and word=='when') and len([True for word,tag in pos_tokens  if 'start' in word])>0) or (len([True for word,tag in pos_tokens  if 'start' in word])>0 and len([True for word,tag in pos_tokens  if 'date' in word])>0)) :
                responseMsg = intents[8]['specifics'][0]['start_date'][0]
            elif(((token == 'WRB' and word=='when') and len([True for word,tag in pos_tokens  if 'end' in word])>0) or (len([True for word,tag in pos_tokens  if 'end' in word])>0 and len([True for word,tag in pos_tokens  if 'date' in word])>0)) :
                responseMsg = intents[8]['specifics'][0]['end_date'][0]
            elif((word == 'duration') or ((token == 'WRB' and word=='how') and  len([True for word,tag in pos_tokens  if 'long' in word])>0) ):
                responseMsg = intents[8]['specifics'][0]['duration'][0]
            elif(word == 'project' or word=='do'):
                responseMsg = intents[8]['specifics'][0]['project'][0]
            elif(token=='WDT' and word == 'company'):
                responseMsg = intents[8]['specifics'][0]['company'][0]

        if (intentTag == 'unknown' and previousContext == 'current_experience') or (context == 'current_experience'):
            for word,token in pos_tokens:
                if((token == 'WRB' and word=='where') or (word=='location')):
                    responseMsg = intents[9]['specifics'][0]['location'][0]
                elif(token=='WDT'):
                    responseMsg = intents[9]['specifics'][0]['university'][0]
                elif(((token == 'WRB' and word=='when') and len([True for word,tag in pos_tokens  if 'start' in word])>0) or (len([True for word,tag in pos_tokens  if 'start' in word])>0 and len([True for word,tag in pos_tokens  if 'date' in word])>0)) :
                    responseMsg = intents[9]['specifics'][0]['start_date'][0]
                elif(((token == 'WRB' and word=='when') and len([True for word,tag in pos_tokens  if 'end' in word])>0) or (len([True for word,tag in pos_tokens  if 'end' in word])>0 and len([True for word,tag in pos_tokens  if 'date' in word])>0)) :
                    responseMsg = intents[9]['specifics'][0]['end_date'][0]
                elif((word == 'duration') or ((token == 'WRB' and word=='how') and  len([True for word,tag in pos_tokens  if 'long' in word])>0) ):
                    responseMsg = intents[9]['specifics'][0]['duration'][0]
                elif(word == 'project' or word=='do'):
                    responseMsg = intents[9]['specifics'][0]['project'][0]
                elif(token=='WDT' and word == 'company'):
                    responseMsg = intents[9]['specifics'][0]['company'][0]



    sessionId = dbInsert(sessionId,message,responseMsg,context,previousContext)
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
    result.append(context)
    result.append(sessionId)
    return jsonify({"result":result})

def dbDelete(query):
    conn = sqlite3.connect('resume.db')
    c = conn.cursor()
    c.execute(query) #"DELETE FROM conversations"
    conn.commit()
    conn.close()

def dbCreate():
    conn = sqlite3.connect('resume.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS conversations (sessionId integer PRIMARY KEY, userMessage text, botResponse text, responseContext text, previousResponseContext text, insertedDate timestamp)''')
    conn.commit()
    conn.close()

def dbInsert(sessionId,userMessage,botResponse,respContext,prevContext):
    conn = sqlite3.connect('resume.db')
    c = conn.cursor()
    if sessionId == None or sessionId == '':
        c.execute('''SELECT MAX(sessionId) FROM conversations''')
        sessionId = c.fetchone()[0]+1
    elif sessionId == 0:
        sessionId = 1
    insertValues = (sessionId,userMessage,botResponse,respContext,prevContext,datetime.datetime.now())
    print(insertValues)
    print(c.execute('''INSERT INTO conversations VALUES (?,?,?,?,?,?)''',insertValues))
    conn.commit()
    conn.close()
    return sessionId

def dbSelect(sessionId):
    session = (sessionId,)
    conn = sqlite3.connect('resume.db')
    c = conn.cursor()
    c.execute('''SELECT * FROM conversations where sessionId = ?''',session)
    all_rows = c.fetchall()
    conn.commit()
    conn.close()
    return all_rows

def loadIntents():
    with open('profile.json') as json_data:
        intents = json.load(json_data)
    return intents


def cleanText(summary):
    summary = summary.replace('*','').replace('-',' ').replace('/',' ').replace("'",' ')
    summary_new = [word for word,tag in nltk.pos_tag(nltk.word_tokenize(summary.lower())) if tag not in ['PRP','IN','PRP$','TO','UH']]
    tokens_summary = [str.lower().strip(string.punctuation) for str in summary_new if str not in stopwords]
    #tokens_summary = [str.lower().strip(string.punctuation) for str in summary.split()]
    lemma_summary = [lemmatizer.lemmatize(token) for token in tokens_summary if len(token) > 0]
    print(' '.join(word for word in lemma_summary))
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
        print("main method")
        classes, intents = loadData()
        X = vectorizer.fit_transform([cleanText(pattern) for pattern,tag in documents])
        Y = tfTransformer.fit_transform(X)
        #svmClassifier.fit(Y, list(tag for pattern,tag in documents))

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
