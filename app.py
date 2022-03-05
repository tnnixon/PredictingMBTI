import regex as re,string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from flask import Flask, render_template, jsonify

nltk.download('punkt')
nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')


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


#------------------------------------------------------------------------------------------------------------------------
app = Flask(__name__)
#-------------------------------------------------------------------------------------------------------

# Read stored models

IntroExtro = pickle.load(open('m1-intro-extro.pkl', 'rb'))
SensorIntutive = pickle.load(open('m1-senso-intuit.pkl', 'rb'))
ThinkerFeeler = pickle.load(open('m1-think-feel.pkl', 'rb'))
JudgerPerceiver = pickle.load(open('m1-judge-percv.pkl', 'rb'))

# Flask routes

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