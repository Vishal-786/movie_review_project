import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Manually download required data
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)


# Load model and vectorizer
model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Clean input text
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.title("🎬 Movie Review Sentiment Analysis")

user_input = st.text_area("Enter a movie review", "")

if st.button("Predict"):
    if user_input:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        sentiment = "😊 Positive" if prediction == 1 else "😠 Negative"
        st.success(f"Sentiment: {sentiment}")
    else:
        st.warning("Please enter a review.")
