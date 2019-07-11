#Implemention Natural Language Processing-
"""
Created on Fri Apr  6 12:27:52 2018

@author: RajeshMylsamy
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, jsonify, request
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
    
app = Flask(__name__)

@app.route('/segment', methods =['POST'])
def predict():

    # Importing the dataset
    dataset = pd.read_csv('quest.csv', delimiter = '\t',quoting = 3)
    
    #Cleaning the texts
    corpus = []
    for i in range(0,41):
        corpus.append(cleanseName(dataset['Review'][i]))
        
    #Creating the Bag of words model
    from sklearn.feature_extraction.text import CountVectorizer
    cv=CountVectorizer(max_features=1500) 
    X=cv.fit_transform(corpus).toarray()
    y=dataset.iloc[:,1].values
    
    
    #Fitting the callifier to the training set
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X,y)
    
    
    requestJson = json.loads(json.dumps(request.get_json()))
    question = requestJson["question"]
    cleansedQuestion = [cleanseName(question)]
    #Prediction the input set results
    vect = cv.transform(cleansedQuestion).toarray()
    my_prediction = classifier.predict(vect) 
    areaOfProduct = str(my_prediction[0]) 
    return jsonify({'question': question,'cleansedQuestion': cleansedQuestion[0], "areaOfProdcut" : areaOfProduct}), 201

@app.route('/answer', methods =['POST'])
def answer():
    requestJson = json.loads(json.dumps(request.get_json()))
    question = requestJson["question"]    
    return jsonify({'question': question,'answer': question}), 201

def cleanseName(input):
    nltk.download('stopwords')
    review = re.sub('[^a-zA-Z]',' ', input)
    review= review.lower()
    review= review.split()
    ps = PorterStemmer()
    review= [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    return review

if __name__ == '__main__':
    app.run(debug=True)






