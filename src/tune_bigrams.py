# src/tune_bigrams.py

import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# --- Load and preprocess the SMS dataset ---
df = pd.read_csv("data/SMSSpamCollection", sep='\t', header=None, names=['label', 'message'])

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.strip()

df['clean_message'] = df['message'].apply(clean_text)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# --- Vectorize with bigrams and stopword removal ---
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),   # Unigrams + Bigrams
    min_df=3,             # Ignore very rare terms
    max_df=0.95           # Ignore very common terms
)

X = vectorizer.fit_transform(df['clean_message'])
y = df['label_num']

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train Logistic Regression ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- Evaluate the model ---
y_pred = model.predict(X_test)

print("📊 Evaluation with bigrams and stopword removal:\n")
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

