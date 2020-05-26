from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

#==
"""
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import Word
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import nltk
nltk.download('wordnet')

import seaborn as sn
import matplotlib.pyplot as plt"""
#==
app = Flask(__name__)

def clean_str(string):
  
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", "", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"'", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[0-9]\w+|[0-9]","", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def check_news_type(news_article):
    news_article = [' '.join([Word(word).lemmatize() for word in clean_str(news_article).split()])]
    features = vect.transform(news_article)
    return str(model.predict(features)[0])



@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

#---------------------------------------------------------------------------------------------------
	"""data = pd.read_csv('data/categ_dataset.csv', encoding='cp1252')
	x = data['news'].tolist()
	y = data['type'].tolist()
	for index,value in enumerate(x):
		x[index] = ' '.join([Word(word).lemmatize() for word in clean_str(value).split()])
	vect = TfidfVectorizer(stop_words='english',min_df=2)

	X = vect.fit_transform(x)
	Y = np.array(y)
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
	model = RandomForestClassifier(n_estimators=300, max_depth=150,n_jobs=1)
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	acc = accuracy_score(y_test,y_pred)

	pickle.dump(model,open('catego_model.sav','wb'))

	pickle.dump(vect,open('catego_vect.sav','wb'))"""


#---------------------------------------------------------------------------------------------------
	cv = pickle.load(open('data/catego_vect.sav', 'rb'))
	clf = pickle.load(open('data/catego_model.sav', 'rb'))

	if request.method == 'POST':
		text_input = request.form['text_input']
		data = [text_input]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)[0]
		print("111111111111111111111111111111111111",my_prediction)

		if my_prediction == 'sport':
			cv_sport = pickle.load(open('data/sport_vect.sav', 'rb'))
			clf_sport = pickle.load(open('data/sport_model.sav', 'rb'))
			text_input = request.form['text_input']
			data = [text_input]
			vect = cv_sport.transform(data).toarray()
			my_prediction_t = clf_sport.predict(vect)
			print("22222222222222222222222222222222",my_prediction_t)


		if my_prediction == 'business':
			cv_business = pickle.load(open('data/business_vect.sav', 'rb'))
			clf_business = pickle.load(open('data/business_model.sav', 'rb'))
			text_input = request.form['text_input']
			data = [text_input]
			vect = cv_business.transform(data).toarray()
			my_prediction_t = clf_business.predict(vect)
			print("22222222222222222222222222222222",my_prediction_t)

		if my_prediction == 'entertainment':
			cv_entertainment = pickle.load(open('data/entertainment_vect.sav', 'rb'))
			clf_entertainment = pickle.load(open('data/entertainment_model.sav', 'rb'))
			text_input = request.form['text_input']
			data = [text_input]
			vect = cv_entertainment.transform(data).toarray()
			my_prediction_t = clf_entertainment.predict(vect)
			print("22222222222222222222222222222222",my_prediction_t)

		if my_prediction == 'politics':
			cv_politics = pickle.load(open('data/politics_vect.sav', 'rb'))
			clf_politics = pickle.load(open('data/politics_model.sav', 'rb'))
			text_input = request.form['text_input']
			data = [text_input]
			vect = cv_politics.transform(data).toarray()
			my_prediction_t = clf_politics.predict(vect)
			print("22222222222222222222222222222222",my_prediction_t)


		if my_prediction == 'tech':
			cv_tech = pickle.load(open('data/tech_vect.sav', 'rb'))
			clf_tech = pickle.load(open('data/tech_model.sav', 'rb'))
			text_input = request.form['text_input']
			data = [text_input]
			vect = cv_tech.transform(data).toarray()
			my_prediction_t = clf_tech.predict(vect)
			print("22222222222222222222222222222222",my_prediction_t)




#---------------------------------------------------------------------------------------------------

	return render_template('result.html',prediction = my_prediction_t, catego= my_prediction)



if __name__ == '__main__':
	app.run(debug=True)