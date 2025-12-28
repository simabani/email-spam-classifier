# src/save_model.py

import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump
import os

# --- Preprocessing function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- Load dataset ---
df = pd.read_csv("data/SMSSpamCollection", sep='\t', header=None, names=['label', 'message'])
df['clean_message'] = df['message'].apply(clean_text)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# --- TF-IDF Vectorization ---
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=3, max_df=0.95)
X = vectorizer.fit_transform(df['clean_message'])
y = df['label_num']

# --- Train model ---
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# --- Ensure models/ folder exists ---
os.makedirs("models", exist_ok=True)

# --- Save model and vectorizer ---
dump(model, "models/spam_classifier_model.pkl")
dump(vectorizer, "models/spam_vectorizer.pkl")

print("✅ Model and vectorizer saved to 'models/' folder.")
