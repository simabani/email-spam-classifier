import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# === Load dataset ===
data_path = "data/SMSSpamCollection"
texts, labels = [], []
with open(data_path, "r", encoding="utf-8") as file:
    for line in file:
        label, text = line.strip().split("\t")
        texts.append(text)
        labels.append(label)

# === Preprocessing ===
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# === Train model ===
model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

# === Generate classification report ===
report = classification_report(y_test, y_pred, output_dict=True)

# === Plot ===
class_labels = [label for label in report.keys() if label not in ('accuracy', 'macro avg', 'weighted avg')]
metrics = ['precision', 'recall', 'f1-score']

data = []
for metric in metrics:
    data.append([report[label][metric] for label in class_labels])

x = np.arange(len(class_labels))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))
for i, metric in enumerate(metrics):
    ax.bar(x + i * width, data[i], width, label=metric)

ax.set_ylabel('Score')
ax.set_title('Classification Metrics by Class')
ax.set_xticks(x + width)
ax.set_xticklabels(class_labels)
ax.set_ylim(0, 1.1)
ax.legend()

for i in range(len(metrics)):
    for j in range(len(class_labels)):
        score = data[i][j]
        ax.text(j + i * width, score + 0.02, f"{score:.2f}", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()
