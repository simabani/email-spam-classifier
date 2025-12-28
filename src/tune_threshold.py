# src/tune_threshold.py

import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# --- Load the SMS spam dataset ---
df = pd.read_csv("data/SMSSpamCollection", sep='\t', header=None, names=['label', 'message'])

# --- Clean the messages ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['clean_message'] = df['message'].apply(clean_text)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# --- TF-IDF vectorization ---
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_message'])
y = df['label_num']

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Logistic Regression model ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --- Predict probabilities ---
y_proba = model.predict_proba(X_test)[:, 1]

# --- Try multiple thresholds ---
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

print("🔎 Tuning classification thresholds:\n")

for thresh in thresholds:
    y_pred = (y_proba >= thresh).astype(int)
    print(f"\n📍 Threshold: {thresh}")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

