# src/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
import string
import re

# Optional: download stopwords if not already done
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load the dataset
df = pd.read_csv("data/SMSSpamCollection", sep='\t', header=None, names=['label', 'message'])

# Clean and normalize text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = re.sub(r'\s+', ' ', text)  # remove extra whitespace
    tokens = text.strip().split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

df['clean_message'] = df['message'].apply(clean_text)

# Encode the labels: ham = 0, spam = 1
label_encoder = LabelEncoder()
df['label_num'] = label_encoder.fit_transform(df['label'])

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_message'])
y = df['label_num']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Debug output
print("✅ Preprocessing complete.")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

