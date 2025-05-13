# Import Libraries
import streamlit as st
import joblib
from keras.models import load_model
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import gensim
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np 
from nltk.data import find

def safe_download(resource):
    try:
        find(resource)
    except LookupError:
        nltk.download(resource.split("/")[-1])

safe_download('tokenizers/punkt')
safe_download('corpora/stopwords')
safe_download('tokenizers/punkt_tab')



# Load model and Word2Vec
model = load_model('lstm_model.keras')  # Load the LSTM model
word2vec = Word2Vec.load('word2vec.model')  # Load Word2Vec model

stop_words = set(stopwords.words('english'))  # Set of stopwords
lemmatizer = WordNetLemmatizer()  # Lemmatizer to reduce words to root form


# Clean input text
def handle_negation(text):
    negations = [
        "don't", "isn't", "aren't", "didn't", "can't", "won't", "never", 
        "no", "nothing", "none", "nobody", "neither", "nowhere", 
        "without", "hardly", "scarcely", "barely", "not", "doesn't", "wasn't", 
        "weren't", "shouldn't", "wouldn't", "couldn't", "hasn't", "haven't"
    ]
    for neg in negations:
        pattern = r'\b' + re.escape(neg) + r'\b'
        text = re.sub(pattern, 'NOT', text)  # Replace negations with 'NOT'
    return text

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = handle_negation(text)  # Handle negation first
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove punctuation and non-alphanumeric characters
    tokens = nltk.word_tokenize(text)  # Tokenize text
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatize and remove stopwords
    return ' '.join(tokens)  # Join tokens back into a string


# Sentence into vector using Word2Vec
def sent_to_vec(sentence):
    return [word2vec.wv[word] if word in word2vec.wv else np.zeros(word2vec.vector_size) for word in sentence]


# Streamlit UI
st.title("🎬 Movie Review Sentiment Analysis")

user_input = st.text_area("Enter a movie review", "")

if st.button("Predict"):
    if user_input:
        cleaned = clean_text(user_input)  # Clean the input review
        new_sentence = word_tokenize(cleaned)  # Tokenize the cleaned sentence
        new_sentence_vector = sent_to_vec(new_sentence)  # Convert sentence into word vectors
        new_sentence_padded = pad_sequences([new_sentence_vector], maxlen=50, padding='post', dtype='float32')  # Pad sequence to fixed length
        
        # Predict sentiment
        prediction = model.predict(new_sentence_padded)[0]  # Get the sentiment prediction

        if prediction > 0.5:  # Positive sentiment if prediction > 0.5
            st.success("😊 Positive Sentiment")
        else:
            st.warning("😞 Negative Sentiment")
    else:
        st.warning("Please enter a review.")  # Alert if no input is given
