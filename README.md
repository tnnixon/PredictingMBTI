# Predicting MBTI Personality
<h1> Overview </h1>
<p> 
  The Myers Briggs Type Indicator (or MBTI for short) is a personality type system that divides everyone into 16 distinct personality types across 4 axis:
  <ul>
    <li> Introversion (I) — Extroversion (E) </li>
    <li> Intuition (N) — Sensing (S) </li>
    <li> Thinking (T) — Feeling (F) </li>
    <li> Judging (J) — Perceiving (P) </li>
  </ul>
The overall goal for the outcome is to be able to predict a person’s personality type based on some text that they have written
The dataset contains ~8600 observations (people), where each observation gives a person’s:
<ul> 
  <li> Myers-Briggs personality type (as a 4-letter code) </li>
  <li> An excerpt containing the last 50 posts on their PersonalityCafe forum (each entry separated by “|||”) </li>
  </ul>
  Data from https://www.kaggle.com/datasnaek/mbti-type was used for supervised ML<br>
  Presentation: https://docs.google.com/presentation/d/1MS5NI7GGJzh51OxxgC4yXFF7KudZNkmj/edit?usp=sharing&ouid=108140751856213571221&rtpof=true&sd=true
</p>
<h1> Method Summary </h1>
<p>
  <ul>
  <li> Cleaned the text of all punctuation  and stopwords (using nlt.stopwords).</li>
  <li> Tokenized (using nlkt.word_toeknize) and lemmatized (using nltk.WordNetLemmatizer).</li>
  <li> Created a bag of words for each personality type (I,E,N,S,T,F,j & P) </li>
  <li> Ran word frequencies (using nltk.FreqDist) and NaiveBayesClassifier to classsify and predict. To Classify and predict E&I,N&S,T&F,J&P are grouped together.</li>
  </ul>
 Below is the accuracy table
 
| Perosnality Group| Train Accuarcy %|Test Accuracy %
| ------------- |:-------------:| -----:|
| Introvert - Extrovert| 100 | 78 |
| Sensor - Intuitives| 100      |   88.29 |
| Thinking - Feelers | 100      |    52.83 |
| Judgers - Perceivers| 100      |   61.58 |

<h1> Other Comparision/Analysis done </h1>
<p> 
  Below are some other comparision/Analsyis done on the data-set
 <ul>
   <li> Emoji Counts by Personality Type </li>
   <li> Most commonly used Words </li>
   <li> Sentiment Analysisusing TextBlob </li>
   <li> Comparision Frequent Words used by Trait </li>
   <li> Comparision Biagram frequency of  Words used by Trait </li>
   <li> Linear Regression </li>
   <li> Random Forest Classifier </li>
   
   
   
</ul>
</p>
