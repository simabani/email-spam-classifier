# src/cli_predict.py

import re
import string
from joblib import load

# --- Preprocessing function (same as in training) ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- Load model and vectorizer ---
model = load("models/spam_classifier_model.pkl")
vectorizer = load("models/spam_vectorizer.pkl")

print("📨 SMS Spam Classifier (type 'exit' to quit)\n")

# --- Loop for user input ---
while True:
    message = input("Enter a message: ")
    
    if message.lower() == "exit":
        print("👋 Exiting classifier.")
        break

    cleaned = clean_text(message)
    X_input = vectorizer.transform([cleaned])
    prediction = model.predict(X_input)[0]

    label = "SPAM 🚨" if prediction == 1 else "HAM ✅"
    print(f"→ Prediction: {label}\n")

