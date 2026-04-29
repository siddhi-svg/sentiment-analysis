import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# Clean function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return ' '.join(words)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# UI
st.title("🎬 Sentiment Analysis App")

user_input = st.text_area("Enter a movie review:")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]

    if prediction == 1:
        st.success("Positive 😊")
    else:
        st.error("Negative 😠")