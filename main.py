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
import sqlite3
import datetime
import time
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import random
from gingerit.gingerit import GingerIt
from sklearn.naive_bayes import MultinomialNB


class ResumeBot:
    def __init__(self):
        self.vectorizer = CountVectorizer(token_pattern=r'\b\w+\b',min_df=0)
        self.tfTransformer = TfidfTransformer(use_idf = True)
        self.svmClassifier = svm.SVC(kernel='linear', C = 1.0,probability=True)
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.stopwords.extend(string.punctuation)
        ignoreWords = ['not','what', 'which', 'who', 'whom', 'do', 'does', 'did', 'when', 'where', 'why', 'how']
        #ignoreWords = []
        for word in ignoreWords:
            self.stopwords.remove(word)
        self.lemmatizer = WordNetLemmatizer()
        self.classes = []
        self.documents = []
        self.modelPickleFileName = 'resumeBot.pkl'
        self.output = []
        print("vectorizer starts")
        self.classes, self.intents = self.loadData()
        X = self.vectorizer.fit_transform([self.cleanText(pattern) for pattern,tag in self.documents])
        Y = self.tfTransformer.fit_transform(X)
        #self.svmClassifier.fit(Y, list(tag for pattern,tag in self.documents))
        print("vectorizer ends; model starts")
        self.clfrNB = MultinomialNB(alpha = 0.1,class_prior=None, fit_prior=True)
        self.clfrNB.fit(Y, list(tag for pattern,tag in self.documents))
        print("model ends")
        # reset underlying graph data
        #tf.reset_default_graph()
        # Build neural network
        #net = tflearn.input_data(shape=[None,len(Y.toarray().tolist()[0])])
        #net = tflearn.fully_connected(net, 8)
        #net = tflearn.fully_connected(net, 8)
        #net = tflearn.fully_connected(net, 32)
        #net = tflearn.fully_connected(net, len(self.output[0]), activation='softmax')
        #net = tflearn.regression(net)

        # Define model and setup tensorboard
        #self.model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
        # Start training (apply gradient descent algorithm)
        #self.model.fit(Y.toarray().tolist(), self.output, n_epoch=1000, batch_size=8)
        #model.save('model.tflearn')

    def cleanText(self,summary):
        summary = summary.lower()
        try:
            parser = GingerIt()
            summary = parser.parse(summary+' ? ')['result']
        except:
            pass

        summary = summary.replace('*','').replace('-',' ').replace('/',' ').replace("'",' ')
        summary_new = [word for word,tag in nltk.pos_tag(nltk.word_tokenize(summary.lower())) if tag not in ['CC','DT','EX','MD','PRP','IN','PRP$','RP','TO','UH']]
        tokens_summary = [str.lower().strip(string.punctuation) for str in summary_new if str not in self.stopwords]
        #tokens_summary = [str.lower().strip(string.punctuation) for str in summary.split()]
        lemma_summary = [self.lemmatizer.lemmatize(token) for token in tokens_summary if len(token) > 0]
        return(' '.join(word for word in lemma_summary))


    def loadData(self):
        classes = []
        # loop through each sentence in our intents patterns
        intents = self.loadIntents()['intents']
        for intent in intents:
            for pattern in intent['patterns']:
                self.documents.append((pattern, intent['tag']))
                if intent['tag'] not in classes:
                    classes.append(intent['tag'])

        classes = sorted(list(set(classes)))

        output_empty = [0] * len(classes)
        for doc in self.documents:
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1
            self.output.append(output_row)
        return (classes,intents)

    def checkIsItName(self,userMessage):
        response = ''
        nlp = en_core_web_sm.load()
        doc = nlp(userMessage)
        isEntity = False
        for entry in doc.ents:
            if entry.label_ == 'PERSON':
                response = 'Hi '+entry.text[0].upper()+entry.text[1:]+", Nice to meet you. You can ask questions to know about Raja Singh's professional career."
                isEntity = True
                break
        if not isEntity:
            tags = nltk.pos_tag(nltk.word_tokenize(userMessage))
            currentIndex = 0
            flag = False
            for word, tag in tags:
                if(tag == 'VBZ' or tag == 'VBP'):
                    flag = True
                    break
                currentIndex+=1
            if flag:
                name = ''
                for index in range(currentIndex+1,len(tags)):
                    if tags[index][1] == '.':
                        break
                    else:
                        print(tags[index])
                        name += ' '+tags[index][0]
                response = 'Hi'+name[0].upper()+name[1:]+", Nice to meet you. You can ask questions to know about Raja Singh's professional career."
            else:
                response = 'Hi '+userMessage[0].upper()+userMessage[1:]+", Nice to meet you. You can ask questions to know about Raja Singh's professional career."
        return response

    def checkPattern(self,actualMessage,currentContext, previousContext,prev_datatype,curr_datatype,sessionId,modelResponse):
        response = ''
        pos_tokens = nltk.pos_tag(nltk.word_tokenize(actualMessage.lower()))
        patternTag = '-->'.join(word+'->'+tag for word,tag in pos_tokens)
        if patternTag == 'who->WP-->are->VBP-->you->PRP-->?->.' or patternTag == 'what->WP-->is->VBZ-->your->PRP$-->name->NN-->?->.':
            response = 'I am a bot. I am here to help you to know about Raja Singh Ravi. <br/> Ask Questions related to his professional career'
            return response
        #modelIntent = currentContext
        #context = ''
        #if modelIntent == 'unknown':
        #    context = previousContext
        print("Check pattern function")
        print("actualMessage: "+ actualMessage.lower())
        print("current cxt: "+ currentContext)
        print("previous cxt: "+ previousContext)
        print("previous datatype: "+ prev_datatype)
        isPrevContextUsed = False
        if (currentContext == 'unknown' and previousContext == 'master') or (currentContext == 'master'):

            for word,token in pos_tokens:
                if((token == 'WRB' and word=='where') or (word=='location')):
                    response = self.intents[5]['specifics'][0]['location']
                    if currentContext == 'unknown':
                        isPrevContextUsed = True
                    break
                elif(token=='WDT'):
                    response = self.intents[5]['specifics'][0]['university']
                    if currentContext == 'unknown':
                        isPrevContextUsed = True
                    break
                elif((token == 'WRB' and word=='when') or (word == 'year')) :
                    response = self.intents[5]['specifics'][0]['year']
                    if currentContext == 'unknown':
                        isPrevContextUsed = True
                    break
                elif(word == 'gpa'):
                    response = self.intents[5]['specifics'][0]['gpa']
                    if currentContext == 'unknown':
                        isPrevContextUsed = True
                    break
                elif(word == 'major'):
                    response = self.intents[5]['specifics'][0]['major']
                    if currentContext == 'unknown':
                        isPrevContextUsed = True
                    break


        if (currentContext == 'unknown' and previousContext == 'bachelor') or (currentContext == 'bachelor'):

            for word,token in pos_tokens:
                if((token == 'WRB' and word=='where') or (word=='location')):
                    response = self.intents[4]['specifics'][0]['location']
                    if currentContext == 'unknown':
                        isPrevContextUsed = True
                    break
                elif(token=='WDT'):
                    response = self.intents[4]['specifics'][0]['university']
                    if currentContext == 'unknown':
                        isPrevContextUsed = True
                    break
                elif((token == 'WRB' and word=='when') or (word == 'year')) :
                    response = self.intents[4]['specifics'][0]['year']
                    if currentContext == 'unknown':
                        isPrevContextUsed = True
                    break
                elif(word == 'gpa'):
                    response = self.intents[4]['specifics'][0]['gpa']
                    if currentContext == 'unknown':
                        isPrevContextUsed = True
                    break
                elif(word == 'major'):
                    response = self.intents[4]['specifics'][0]['major']
                    if currentContext == 'unknown':
                        isPrevContextUsed = True
                    break


        if (currentContext == 'unknown' and previousContext == 'skills') or (currentContext == 'skills'):
            if "machine learning" in actualMessage:
                response = self.intents[7]['specifics'][0]['machine learning'][0]
                if currentContext == 'unknown':
                    isPrevContextUsed = True
            elif "big data" in actualMessage:
                response = self.intents[7]['specifics'][0]['big data'][0]
                if currentContext == 'unknown':
                    isPrevContextUsed = True
            elif "reporting" in actualMessage:
                response = self.intents[7]['specifics'][0]['reporting'][0]
                if currentContext == 'unknown':
                    isPrevContextUsed = True


        if (currentContext == 'unknown' and previousContext == 'experience1') or (currentContext == 'experience1'):

            for word,token in pos_tokens:
                if((token == 'WRB' and word=='where') or (word=='location')):
                    response = self.intents[9]['specifics'][0]['location'][0]
                    if currentContext == 'unknown':
                        isPrevContextUsed = True
                    break
                elif(((token == 'WRB' and word=='when') and len([True for word,tag in pos_tokens  if 'start' in word])>0) or (len([True for word,tag in pos_tokens  if 'start' in word])>0 and len([True for word,tag in pos_tokens  if 'date' in word])>0)) :
                    response = self.intents[9]['specifics'][0]['start_date'][0]
                    if currentContext == 'unknown':
                        isPrevContextUsed = True
                    break
                elif(((token == 'WRB' and word=='when') and len([True for word,tag in pos_tokens  if 'end' in word])>0) or (len([True for word,tag in pos_tokens  if 'end' in word])>0 and len([True for word,tag in pos_tokens  if 'date' in word])>0)) :
                    response = self.intents[9]['specifics'][0]['end_date'][0]
                    if currentContext == 'unknown':
                        isPrevContextUsed = True
                    break
                elif((word == 'duration') or ((token == 'WRB' and word=='how') and  len([True for word,tag in pos_tokens  if 'long' in word])>0) ):
                    response = self.intents[9]['specifics'][0]['duration'][0]
                    if currentContext == 'unknown':
                        isPrevContextUsed = True
                    break
                elif(word == 'project' or word=='do'):
                    response = self.intents[9]['specifics'][0]['project'][0]
                    if currentContext == 'unknown':
                        isPrevContextUsed = True
                    break
                elif(token=='WDT' and word == 'company'):
                    response = self.intents[9]['specifics'][0]['company'][0]
                    if currentContext == 'unknown':
                        isPrevContextUsed = True
                    break

        if (currentContext == 'unknown' and previousContext == 'experience2') or (currentContext == 'experience2'):

            for word,token in pos_tokens:
                if((token == 'WRB' and word=='where') or (word=='location')):
                    response = self.intents[10]['specifics'][0]['location'][0]
                    if currentContext == 'unknown':
                        isPrevContextUsed = True
                    break
                elif(((token == 'WRB' and word=='when') and len([True for word,tag in pos_tokens  if 'start' in word])>0) or (len([True for word,tag in pos_tokens  if 'start' in word])>0 and len([True for word,tag in pos_tokens  if 'date' in word])>0)) :
                    response = self.intents[10]['specifics'][0]['start_date'][0]
                    if currentContext == 'unknown':
                        isPrevContextUsed = True
                    break
                elif(((token == 'WRB' and word=='when') and len([True for word,tag in pos_tokens  if 'end' in word])>0) or (len([True for word,tag in pos_tokens  if 'end' in word])>0 and len([True for word,tag in pos_tokens  if 'date' in word])>0)) :
                    response = self.intents[10]['specifics'][0]['end_date'][0]
                    if currentContext == 'unknown':
                        isPrevContextUsed = True
                    break
                elif((word == 'duration') or ((token == 'WRB' and word=='how') and  len([True for word,tag in pos_tokens  if 'long' in word])>0) ):
                    response = self.intents[10]['specifics'][0]['duration'][0]
                    if currentContext == 'unknown':
                        isPrevContextUsed = True
                    break
                elif(word == 'project' or word=='do'):
                    response = self.intents[10]['specifics'][0]['project'][0]
                    if currentContext == 'unknown':
                        isPrevContextUsed = True
                    break
                elif(token=='WDT' and word == 'company'):
                    response = self.intents[10]['specifics'][0]['company'][0]
                    if currentContext == 'unknown':
                        isPrevContextUsed = True
                    break

        if (currentContext == 'unknown' and previousContext == 'experience3') or (currentContext == 'experience3'):

            for word,token in pos_tokens:
                if((token == 'WRB' and word=='where') or (word=='location')):
                    response = self.intents[11]['specifics'][0]['location'][0]
                    if currentContext == 'unknown':
                        isPrevContextUsed = True
                    break
                elif(((token == 'WRB' and word=='when') and len([True for word,tag in pos_tokens  if 'start' in word])>0) or (len([True for word,tag in pos_tokens  if 'start' in word])>0 and len([True for word,tag in pos_tokens  if 'date' in word])>0)) :
                    response = self.intents[11]['specifics'][0]['start_date'][0]
                    if currentContext == 'unknown':
                        isPrevContextUsed = True
                    break
                elif(((token == 'WRB' and word=='when') and len([True for word,tag in pos_tokens  if 'end' in word])>0) or (len([True for word,tag in pos_tokens  if 'end' in word])>0 and len([True for word,tag in pos_tokens  if 'date' in word])>0)) :
                    response = self.intents[11]['specifics'][0]['end_date'][0]
                    if currentContext == 'unknown':
                        isPrevContextUsed = True
                    break
                elif((word == 'duration') or ((token == 'WRB' and word=='how') and  len([True for word,tag in pos_tokens  if 'long' in word])>0) ):
                    response = self.intents[11]['specifics'][0]['duration'][0]
                    if currentContext == 'unknown':
                        isPrevContextUsed = True
                    break
                elif(word == 'project' or word=='do'):
                    response = self.intents[11]['specifics'][0]['project'][0]
                    if currentContext == 'unknown':
                        isPrevContextUsed = True
                    break
                elif(token=='WDT' and word == 'company'):
                    response = self.intents[11]['specifics'][0]['company'][0]
                    if currentContext == 'unknown':
                        isPrevContextUsed = True
                    break

        if (prev_datatype == 'list' and previousContext == 'projects' and (currentContext == 'projects' or currentContext == 'unknown')):

            nlp = en_core_web_sm.load()
            nlp_ner = nlp(actualMessage.lower())
            ner_tags = [(i, i.label_, i.label) for i in nlp_ner.ents]
            cardinal_tags = {1:['one','1','first','1st'],2:['two','2','second','2nd'],3:['three','3','third','3rd'],4:['four','4','4th','fourth']}
            cardinal_list = []
            projects = self.intents[16]['specifics']
            for text, ner, label in ner_tags:
                for key,value in cardinal_tags.items():
                    if str(text).lower() in value:
                        cardinal_list.append(key)
            if len(cardinal_list) > 0:
                for entry in cardinal_list:
                    response += projects[entry-1] + '<br/>'
            curr_datatype = prev_datatype
            if currentContext == 'unknown':
                isPrevContextUsed = True
            print(response)

        print(response)
        if(response == None or response == ''):
            if (modelResponse == '' or modelResponse == None):
                response = self.intents[6]['responses'][0]
            else:
                response = modelResponse

        if isPrevContextUsed:
            currentContext = previousContext

        sessionId = self.dbInsert(sessionId,actualMessage,response,currentContext,previousContext)
        print("Response:  "+response)

        result = []
        result.append(response)
        result.append(currentContext)
        result.append(sessionId)
        result.append(curr_datatype)

        return result


    def getModelResult(self,actualMessage,previousContext,sessionId):
        message = self.cleanText(actualMessage)
        test_X = self.vectorizer.transform([message])
        test_Y = self.tfTransformer.transform(test_X)
        self.vectorizer._validate_vocabulary()
        ft = self.vectorizer.get_feature_names()
        result = list(map(lambda row:dict(zip(ft,row)),test_Y.toarray()))
        #print(result)
        #print(list(result[0].values()))

        #predicted_svm = svmClassifier.predict(test_Y)
        #intentTag = predicted_svm[0]
        #print(svmClassifier.predict_proba(test_Y))
        #print(svmClassifier.classes_)


        predicted_nb = self.clfrNB.predict(test_Y)
        probs = self.clfrNB.predict_proba(test_Y)
        print("naive bayes prob")
        print(self.clfrNB.classes_)
        print(probs[0].tolist())
        if max(probs[0].tolist()) > 0.2:
            max_index = probs[0].tolist().index(max(probs[0].tolist()))
            print("Naive Bayes: "+self.clfrNB.classes_[max_index])
            print("Score: "+str(max(probs[0].tolist())))
            intentTag = self.clfrNB.classes_[max_index]
        else:
            intentTag = 'unknown'

        #result_array = self.model.predict([list(result[0].values())]).tolist()[0]
        #for item,score in zip(self.classes,result_array):
        #    print(item+': '+str(score))
        #print(self.classes)

        #if max(result_array) > 0.6:
        #    intentTag = self.classes[result_array.index(max(result_array))]
        #else:
        #    intentTag = 'unknown'


        #predicted_svm = self.svmClassifier.predict(test_Y)
        #intentTag = predicted_svm[0]
        #print('SVM:'+intentTag)
        return intentTag


    def getBotResponse(self,actualMessage,previousContext,prev_datatype,sessionId):
        print("Get bot response method")
        print('msg '+actualMessage)
        print('prev coxt: '+previousContext)
        print('prev datatype: '+prev_datatype)
        print('session: '+sessionId)
        response = ''
        result = []
        if sessionId == '' or sessionId == None:
            response = self.checkIsItName(actualMessage)
            context = ''
            dataType = ''
            result = []
            sessionId = self.dbInsert(sessionId,actualMessage,response,'','')
            result.append(response)
            result.append(context)
            result.append(sessionId)
            result.append(dataType)

            return result
        else:
            #try:
                #parser = GingerIt()
                #actualMessage = parser.parse(actualMessage+' ')['result']
            #except:
                #pass
            modelIntent = self.getModelResult(actualMessage,previousContext,sessionId)
            for intent in self.intents:
                if intent['tag'] == modelIntent:
                    response = random.choice(intent['responses'])
                    #print("Response:  "+responseMsg)
                    context = intent['context']
                    dataType = intent['type']

                    print("Tag:  "+modelIntent)
                    print("Context:  "+context)
                    print("Data Type:  "+dataType)
                    result = self.checkPattern(actualMessage,context,previousContext,prev_datatype,dataType,sessionId,response)
                    return result

    def dbDelete(self,query):
        conn = sqlite3.connect('resume.db')
        c = conn.cursor()
        c.execute(query) #"DELETE FROM conversations"
        conn.commit()
        conn.close()

    def dbSurveyCreate():
        conn = sqlite3.connect('resume.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS survey (sessionId integer PRIMARY KEY, userMessage text, botResponse text, rating text, insertedDate timestamp)''')
        conn.commit()
        conn.close()

    def dbInsertSurvey(self,sessionId,userMessage,botResponse,rating):
        conn = sqlite3.connect('resume.db')
        c = conn.cursor()
        insertValues = (sessionId,userMessage,botResponse,rating,datetime.datetime.now())
        print(insertValues)
        try:
            c.execute('''INSERT INTO survey VALUES (?,?,?,?,?)''',insertValues)
            conn.commit()
            conn.close()
        except:
            conn.close()


    def dbCreate(self):
        conn = sqlite3.connect('resume.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS conversations (sessionId integer PRIMARY KEY, userMessage text, botResponse text, responseContext text, previousResponseContext text, insertedDate timestamp)''')
        conn.commit()
        conn.close()

    def dbInsert(self,sessionId,userMessage,botResponse,respContext,prevContext):
        conn = sqlite3.connect('resume.db')
        c = conn.cursor()
        try:
            if sessionId == None or sessionId == '':
                c.execute('''SELECT MAX(sessionId) FROM conversations''')
                sessionId = c.fetchone()[0]+1
            elif sessionId == 0:
                sessionId = 1
            insertValues = (sessionId,userMessage,botResponse,respContext,prevContext,datetime.datetime.now())
            print(insertValues)
            c.execute('''INSERT INTO conversations VALUES (?,?,?,?,?,?)''',insertValues)
            conn.commit()
            conn.close()
        except:
            conn.close()
        return sessionId


    def dbSelect(self,sessionId):
        session = (sessionId,)
        conn = sqlite3.connect('resume.db')
        c = conn.cursor()
        c.execute('''SELECT * FROM conversations where sessionId = ?''',session)
        all_rows = c.fetchall()
        conn.commit()
        conn.close()
        return all_rows

    def loadIntents(self):
        with open('profile.json') as json_data:
            self.intents = json.load(json_data)
        return self.intents

app = Flask(__name__)
resumeBot = ResumeBot()

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/getResponse',methods=['GET'])
def getResponse():
    actualMessage = ''
    previousContext = ''
    sessionId = ''
    previousDatatype = ''
    if 'userMessage' in request.args:
        actualMessage = request.args['userMessage']
    if 'previousContext' in request.args:
        previousContext = request.args['previousContext']
    if 'sessionId' in request.args:
        sessionId = request.args['sessionId']
    if 'previousDatatype' in request.args:
        previousDatatype = request.args['previousDatatype']

    print(previousDatatype)
    result = resumeBot.getBotResponse(actualMessage,previousContext,previousDatatype,sessionId)
    print("Actual Message: "+actualMessage)
    print("Bot Response: "+result[0])
    return jsonify({"result":result})

@app.route('/storeSurvey',methods=['GET'])
def storeSurvey():
    sessionId = ''
    userMessage = ''
    botResponse = ''
    rating = ''
    if 'userMessage' in request.args:
        userMessage = request.args['userMessage']
    if 'botResponse' in request.args:
        botResponse = request.args['botResponse']
    if 'sessionId' in request.args:
        sessionId = request.args['sessionId']
    if 'rating' in request.args:
        rating = request.args['rating']
    print(rating)
    print(userMessage)
    print(botResponse)
    print(sessionId)
    resumeBot.dbInsertSurvey(sessionId,userMessage,botResponse,rating)
    return "success"




if __name__ == "__main__":
    handler = RotatingFileHandler('foo.log', maxBytes=10000, backupCount=1)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    try:
        app.run(host='0.0.0.0',port=5001)
    finally:
        print("Exit")
