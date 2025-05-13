# Import Libraray

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
import nltk

import nltk

import nltk

# Download the necessary NLTK resources if not already present
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading punkt...")
        nltk.download('punkt')
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading punkt_tab...")
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading stopwords...")
        nltk.download('stopwords')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading wordnet...")
        nltk.download('wordnet')

# Ensure resources are available
download_nltk_resources()




# Load model and Word2Vec
model = load_model('lstm_model.keras')
word2vec = Word2Vec.load('word2vec.model')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Clean input text
def handle_negation(text):
    # Use regex to match negations as full words
    negations = [
        "don't", "isn't", "aren't", "didn't", "can't", "won't", "never", 
        "no", "nothing", "none", "nobody", "neither", "nowhere", 
        "without", "hardly", "scarcely", "barely", "not", "doesn't", "wasn't", 
        "weren't", "shouldn't", "wouldn't", "couldn't", "hasn't", "haven't"
    ]
    for neg in negations:
        # Use regex to replace whole word negation only
        pattern = r'\b' + re.escape(neg) + r'\b'
        text = re.sub(pattern, 'NOT', text)
    return text

def clean_text(text):
    text = text.lower()
    text = handle_negation(text)  # Handle negation first
    text = re.sub(r'<.*?>', '', text)  # Remove HTML
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove punctuation
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)


# Sentence into vector
def sent_to_vec(sentence):
    return [word2vec.wv[word] if word in word2vec.wv else np.zeros(word2vec.vector_size) for word in sentence]

# Streamlit UI
st.title("🎬 Movie Review Sentiment Analysis")

user_input = st.text_area("Enter a movie review", "")

if st.button("Predict"):
    if user_input:
        cleaned = clean_text(user_input)
        new_sentence = word_tokenize(cleaned)  # Tokenize the cleaned sentence
        new_sentence_vector = sent_to_vec(new_sentence)  # Convert to word vectors
        new_sentence_padded = pad_sequences([new_sentence_vector], maxlen=50, padding='post', dtype='float32')  # Pad the sequence
        prediction = model.predict(new_sentence_padded)[0]

        if prediction > 0.5:
            st.success("😊 Positive Sentiment")
        else:
            st.warning("😞 Negative Sentiment")
    else:
        st.warning("Please enter a review.")
