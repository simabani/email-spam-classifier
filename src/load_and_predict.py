# src/load_and_predict.py

import re
import string
from joblib import load

# --- Text cleaning function (same as in training) ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- Load saved model and vectorizer ---
model = load("models/spam_classifier_model.pkl")
vectorizer = load("models/spam_vectorizer.pkl")

# --- Sample new messages to classify ---
new_messages = [
    "WINNER! Claim your $1000 Walmart gift card now.",
    "Hey, are we still meeting at 6?",
    "URGENT! You’ve won a free ticket to Bahamas!",
    "Don’t forget to submit the report."
]

# --- Preprocess and vectorize ---
cleaned_messages = [clean_text(msg) for msg in new_messages]
X_new = vectorizer.transform(cleaned_messages)

# --- Predict and display results ---
predictions = model.predict(X_new)

for msg, pred in zip(new_messages, predictions):
    label = "SPAM" if pred == 1 else "HAM"
    print(f"[{label}] {msg}")

