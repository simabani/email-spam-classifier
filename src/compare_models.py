# src/compare_models.py

import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# --- Load the SMS spam dataset ---
df = pd.read_csv("data/SMSSpamCollection", sep='\t', header=None, names=['label', 'message'])

# --- Preprocess the text ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['clean_message'] = df['message'].apply(clean_text)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# --- Vectorize with bigrams and stopword removal ---
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95
)

X = vectorizer.fit_transform(df['clean_message'])
y = df['label_num']

# --- Split the data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train Logistic Regression ---
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# --- Train Multinomial Naive Bayes ---
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)

# --- Evaluate both models ---
print("📊 Logistic Regression Performance:\n")
print(classification_report(y_test, lr_preds, target_names=['ham', 'spam']))

print("\n📊 Multinomial Naive Bayes Performance:\n")
print(classification_report(y_test, nb_preds, target_names=['ham', 'spam']))

