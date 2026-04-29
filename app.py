import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os

# Download stopwords (only once)
nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

# -------------------------------
# Text Cleaning Function
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    return ' '.join(words)

# -------------------------------
# Load Dataset (Safer Path)
# -------------------------------
file_path = r"C:\Users\siddh\Downloads\archive (12)\IMDB Dataset.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError("Dataset not found. Check your file path!")

df = pd.read_csv(file_path)

# -------------------------------
# Preprocessing
# -------------------------------
df['clean_review'] = df['review'].apply(clean_text)
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_review'], df['label'], test_size=0.2, random_state=42
)

# -------------------------------
# Vectorization
# -------------------------------
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------------
# Model Training
# -------------------------------
model = LogisticRegression(max_iter=1000, C=5.0)
model.fit(X_train_vec, y_train)

# -------------------------------
# Evaluation
# -------------------------------
preds = model.predict(X_test_vec)
print(f"\n✅ Accuracy: {accuracy_score(y_test, preds):.4f}\n")
print(classification_report(y_test, preds))

# -------------------------------
# Save Model & Vectorizer
# -------------------------------
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\n✅ Model and vectorizer saved successfully!")

# -------------------------------
# Test Prediction Function
# -------------------------------
def predict_sentiment(text):
    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]
    return "Positive 😊" if prediction == 1 else "Negative 😠"

# -------------------------------
# Example Test
# -------------------------------
sample = "This movie was absolutely fantastic!"
print("\nSample Prediction:", predict_sentiment(sample))