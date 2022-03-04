from flask import Flask, render_template, jsonify
import pandas as pd
import regex as re,string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from textblob import TextBlob
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.classify import NaiveBayesClassifier
import random

def clean_text(text):
    text = text.lower()
    text = re.sub('@', '', text)
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub(r"[^a-zA-Z ]+", "", text)
    
    #Tokenize the data
    text = nltk.word_tokenize(text)
    #Remove stopwords
    text = [w for w in text if w not in sw]
    return text

def ClassifyMBTI(input):
  input=clean_text(input)
  word_freq= nltk.FreqDist(input)
  wdict={}
  for k in word_freq:
    wdict[k]=word_freq[k]
  mbti=''
  for i in "INTJ":
   if (IntroExtro.classify(wdict) == 'introvert'):
        mbti += 'I'
  if (IntroExtro.classify(wdict) == 'extrovert'):
        mbti += 'E'
  if (SensorIntutive.classify(wdict) == 'sensor'):
        mbti += 'S'
  if (SensorIntutive.classify(wdict) == 'intutive'):
        mbti += 'N'
  if (ThinkerFeeler.classify(wdict) == 'thinker'):
        mbti += 'T'
  if (ThinkerFeeler.classify(wdict) == 'feeler'):
        mbti += 'F'
  if (JudgerPerceiver.classify(wdict) == 'judger'):
        mbti += 'J'
  if (JudgerPerceiver.classify(wdict) == 'perceiver'):
        mbti += 'P'
  return mbti

def lem(text):
    text = [lemmatizer.lemmatize(t) for t in text]
    text = [lemmatizer.lemmatize(t, 'v') for t in text]
    return text

sw = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

#--------------------------------------Creating Model------------------------------------------------------------------
#Read csv file
df=pd.read_csv("mbti_1.csv")
#Clean the posts
df['clean_posts']=df['posts'].apply(lambda x: clean_text(x))
#Lemmatize posts
df['clean_posts'] = df['clean_posts'].apply(lambda x: lem(x))

# Add Peronality Type Columns
df['eori']=[x[0] for x in df['type']]
df['nors']=[x[1] for x in df['type']]
df['torf']=[x[2] for x in df['type']]
df['jorp']=[x[3] for x in df['type']]

#Extrovert Introvert words  extraction
E_df = df[df['eori']=='E']
E_df.reset_index(inplace=True)

I_df = df[df['eori']=='I']
I_df.reset_index(inplace=True)

all_Ewords=[]        
for i in range(len(E_df)):
  all_Ewords = all_Ewords + E_df['clean_posts'][i]

nlp_words = nltk.FreqDist(all_Ewords)

all_Iwords=[]        
for i in range(len(I_df)):
  all_Iwords = all_Iwords + I_df['clean_posts'][i]

nlp_wordsI = nltk.FreqDist(all_Iwords)

# Naive Bayes Classification analysis for Extroverts(E) and Introverts(I)

# Find words (i.e keys) unique to Extroverts(E) and Introverts(I)
Ekeys = nlp_words.keys()
Ikeys = nlp_wordsI.keys()
#difference = Ekeys - IKeys
print(f"Total Extrover Words - {len(nlp_words)} ; Extrovert only words- {len(Ekeys - Ikeys)}")
print(f"Total Introvert Words - {len(nlp_wordsI)} ; Introvert only words - {len(Ikeys - Ekeys)}")
Eonlykeys = Ekeys - Ikeys
Ionlykeys = Ikeys - Ekeys

#Create list words and corresponding classification (I or E)
EIfeatures=[]
for k in Eonlykeys:
  wdict={}
  wdict[k]=nlp_words[k]
  EIfeatures += [(wdict,'extrovert')]

for k in Ionlykeys:
  wdict={}
  wdict[k]=nlp_wordsI[k]
  EIfeatures += [(wdict,'introvert')]

random.shuffle(EIfeatures)

#Split into Test and Train data sets
x_train,x_test, = train_test_split(EIfeatures)

#Train NaiveBayes classifier
IntroExtro = NaiveBayesClassifier.train(x_train)
print(f"Model Score Accuracy on train data {nltk.classify.util.accuracy(IntroExtro, x_train)*100}%")
print(f"Model Score Accuracy on test data {nltk.classify.util.accuracy(IntroExtro, x_test)*100}%")

#Sensor and Intutive analysis
S_df = df[df['nors']=='S']
S_df.reset_index(inplace=True)

N_df = df[df['nors']=='N']
N_df.reset_index(inplace=True)

all_Swords=[]        
for i in range(len(S_df)):
  all_Swords = all_Swords + S_df['clean_posts'][i]

all_Nwords=[]        
for i in range(len(N_df)):
  all_Nwords = all_Nwords + N_df['clean_posts'][i]

nlp_Swords = nltk.FreqDist(all_Swords)
nlp_Nwords = nltk.FreqDist(all_Nwords)
Skeys = nlp_Swords.keys()
Nkeys = nlp_Nwords.keys()

print(f"Total Sensor Words - {len(nlp_Swords)} ; Sensor only words- {len(Skeys - Nkeys)}")
print(f"Total Intuitive Words - {len(nlp_Nwords)} ; Intuitive only words - {len(Nkeys - Skeys)}")
Sonlykeys = Skeys - Nkeys
Nonlykeys = Nkeys - Skeys

SNfeatures=[]
for k in Sonlykeys:
  wdict={}
  wdict[k]=nlp_Swords[k]
  SNfeatures += [(wdict,'sensor')]

for k in Nonlykeys:
  wdict={}
  wdict[k]=nlp_Nwords[k]
  SNfeatures += [(wdict,'intutive')]
  
random.shuffle(SNfeatures)

x_train,x_test, = train_test_split(SNfeatures)
SensorIntutive = NaiveBayesClassifier.train(x_train)
print(f"Model Score Accuracy on train data {nltk.classify.util.accuracy(SensorIntutive, x_train)*100}%")
print(f"Model Score Accuracy on test data {nltk.classify.util.accuracy(SensorIntutive, x_test)*100}%")

#Thinkers and Feelers Analysis

T_df = df[df['torf']=='T']
T_df.reset_index(inplace=True)

F_df = df[df['torf']=='F']
F_df.reset_index(inplace=True)


all_Twords=[]        
for i in range(len(T_df)):
  all_Twords = all_Twords + T_df['clean_posts'][i]

all_Fwords=[]        
for i in range(len(F_df)):
  all_Fwords = all_Fwords + F_df['clean_posts'][i]

nlp_Twords = nltk.FreqDist(all_Twords)
nlp_Fwords = nltk.FreqDist(all_Fwords)

Tkeys = nlp_Twords.keys()
Fkeys = nlp_Fwords.keys()

print(f"Total Thinker Words - {len(nlp_Twords)} ; Thinker only words- {len(Tkeys - Fkeys)}")
print(f"Total Feeler Words - {len(nlp_Fwords)} ; Feeler only words - {len(Fkeys - Tkeys)}")
Tonlykeys = Tkeys - Fkeys
Fonlykeys = Fkeys - Tkeys

TFfeatures=[]
for k in Tonlykeys:
  wdict={}
  wdict[k]=nlp_Twords[k]
  TFfeatures += [(wdict,'thinker')]

for k in Fonlykeys:
  wdict={}
  wdict[k]=nlp_Fwords[k]
  TFfeatures += [(wdict,'feeler')]
  
random.shuffle(TFfeatures)

x_train,x_test, = train_test_split(TFfeatures)
ThinkerFeeler = NaiveBayesClassifier.train(x_train)
print(f"Model Score Accuracy on train data {nltk.classify.util.accuracy(ThinkerFeeler, x_train)*100}%")
print(f"Model Score Accuracy on test data {nltk.classify.util.accuracy(ThinkerFeeler, x_test)*100}%")

#Judgers(J) and Perceivers(P) anlaysis

J_df = df[df['jorp']=='J']
J_df.reset_index(inplace=True)

P_df = df[df['jorp']=='P']
P_df.reset_index(inplace=True)


all_Jwords=[]        
for i in range(len(J_df)):
  all_Jwords = all_Jwords + J_df['clean_posts'][i]

all_Pwords=[]        
for i in range(len(P_df)):
  all_Pwords = all_Pwords + P_df['clean_posts'][i]

nlp_Jwords = nltk.FreqDist(all_Jwords)
nlp_Pwords = nltk.FreqDist(all_Pwords)

Jkeys = nlp_Jwords.keys()
Pkeys = nlp_Pwords.keys()
#difference = Ekeys - IKeys
print(f"Total Judger Words - {len(nlp_Jwords)} ; Judger only words- {len(Jkeys - Pkeys)}")
print(f"Total Perceiver Words - {len(nlp_Pwords)} ; Perceiver only words - {len(Pkeys - Jkeys)}")
Jonlykeys = Jkeys - Pkeys
Ponlykeys = Pkeys - Jkeys

JPfeatures=[]
for k in Jonlykeys:
  wdict={}
  wdict[k]=nlp_Jwords[k]
  JPfeatures += [(wdict,'judger')]

for k in Ponlykeys:
  wdict={}
  wdict[k]=nlp_Pwords[k]
  JPfeatures += [(wdict,'perceiver')]
  
random.shuffle(JPfeatures)

x_train,x_test, = train_test_split(JPfeatures)

JudgerPerceiver = NaiveBayesClassifier.train(x_train)
print(f"Model Score Accuracy on train data {nltk.classify.util.accuracy(JudgerPerceiver, x_train)*100}%")
print(f"Model Score Accuracy on test data {nltk.classify.util.accuracy(JudgerPerceiver, x_test)*100}%")

#------------------------------------------------------------------------------------------------------------------------
app = Flask(__name__)
#-------------------------------------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/mbti_predict/<text2predict>")
def predict(text2predict):
    out1=ClassifyMBTI(text2predict)
    ptype=""
    for i in out1:
        if i not in ptype:
            ptype+=i
    return jsonify(ptype)


if __name__ == "__main__":
    app.run(debug=False)