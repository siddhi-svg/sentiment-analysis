# 🎬 Sentiment Analysis Web App

🚀 **Live Demo:** https://siddh-sentiment-analysis.streamlit.app/

---

## 📌 Overview

This project is a **Machine Learning-based Sentiment Analysis Web Application** that predicts whether a movie review is **Positive 😊 or Negative 😠**.

The model is trained on the **IMDB Dataset** and deployed using **Streamlit Cloud** for real-time predictions.

---

## ✨ Features

* 🔍 Real-time sentiment prediction
* 🧠 Machine Learning model (Logistic Regression)
* 📝 Text preprocessing (cleaning + stopword removal)
* ⚡ Fast and interactive UI using Streamlit
* 🌐 Fully deployed and accessible online

---

## 🛠️ Tech Stack

* Python 🐍
* Pandas & NumPy
* Scikit-learn
* NLTK
* Streamlit

---

## 📊 Model Details

* Algorithm: Logistic Regression
* Feature Extraction: TF-IDF Vectorizer (uni-grams + bi-grams)
* Dataset: IMDB Movie Reviews
* Accuracy: ~89%

---

## ⚙️ How It Works

1. User enters a movie review
2. Text is cleaned (lowercase, punctuation removed, stopwords filtered)
3. TF-IDF vectorization converts text into numerical features
4. Model predicts sentiment
5. Result is displayed instantly on the UI

---

## 🚀 Run Locally

```bash
git clone https://github.com/siddhi-svg/sentiment-analysis
cd sentiment-analysis
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 📸 Demo Preview

<img width="1564" height="718" alt="image" src="https://github.com/user-attachments/assets/8aefe8ee-ed77-4810-acc3-abd32050cf1b" />

---

## ⚠️ Limitations

* May misclassify mixed sentiment reviews (e.g., "good but slow")
* Does not fully understand context like deep learning models

---

## 🔮 Future Improvements

* Use advanced models like LSTM / BERT
* Add confidence score visualization
* Improve UI/UX
* Support multi-language reviews

---

## 👩‍💻 Author

**Siddhi Grover**
BCA (AI & Data Science)

---

## 🌟 Show Your Support

If you like this project, ⭐ star the repo!
