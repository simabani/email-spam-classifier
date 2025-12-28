# 📧 Email Spam Classifier

Classify SMS messages as **spam** or **ham** using supervised learning.

This project builds a **complete machine learning pipeline** that processes real‑world text messages and predicts whether they are spam (e.g. *“You won $1000!”*) or ham (e.g. *“Meeting at 3 PM”*), using **Logistic Regression** and **Multinomial Naive Bayes**.

---

## 🎯 Features

- ✅ Text preprocessing and TF‑IDF vectorization  
- ✅ Train/test split and evaluation  
- ✅ Model fine‑tuning (threshold adjustment, bigrams)  
- ✅ Model comparison (Logistic Regression vs Naive Bayes)  
- ✅ Save/load models with `joblib`  
- ✅ Command‑line interface (CLI) for real‑time prediction  

---

## 🚀 Getting Started

### Clone the repository
```python
git clone https://github.com/simabani/email-spam-classifier.git
cd email-spam-classifier
```

## 🖥️ Use the CLI Tool

```python
python src/cli_predict.py
```

Enter a message: You’ve won a free cruise!
→ Prediction: SPAM 🚨

Enter a message: Are you free for lunch?
→ Prediction: HAM ✅

Type **“Exit”** to quit.

## 📚 Dataset

- Source: UCI SMS Spam Collection
- Format: label<TAB>message

## 📦 Dependencies
```python
- pandas
- scikit-learn
- joblib
```

## 👩🏻‍💻 Author
Sima Bani
- Senior AI Product Manager @ Zoom
- [Linkedin](https://www.linkedin.com/in/simaban/)
- [See My Portfolio](https://sima-personal-portfolio.lovable.app/)

 
