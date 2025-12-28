🙋‍♀️ Author

**Sima Bani**, Senior AI Product Manager
🔗 [LinkedIn](https://www.linkedin.com/in/simaban/)
💼 [Personal portfolio](https://sima-personal-portfolio.lovable.app/) 

📧 Email Spam Classifier

Classify SMS messages as spam or ham using supervised learning.

This project builds a complete ML pipeline that processes real-world messages and predicts whether they are spam (e.g. "You won $1000!") or ham (e.g. "Meeting at 3 PM"), using Logistic Regression and Multinomial Naive Bayes.

🎯 Features

✅ Text preprocessing and TF-IDF vectorization
✅ Train/test split and evaluation
✅ Model fine-tuning (threshold, bigrams)
✅ Model comparison (LogReg vs Naive Bayes)
✅ Save/load models with joblib
✅ CLI interface for real-time prediction

📁 Project Structure
email-spam-classifier/
├── data/
│   └── SMSSpamCollection         # Raw dataset (TSV)
├── models/
│   ├── spam_classifier_model.pkl
│   └── spam_vectorizer.pkl
├── src/
│   ├── train_models.py           # Step 4: Train and evaluate
│   ├── tune_threshold.py         # Step 6.1: Adjust classification threshold
│   ├── tune_bigrams.py           # Step 6.2: Improve features with bigrams
│   ├── compare_models.py         # Step 6.3: Compare NB vs LR
│   ├── save_model.py             # Step 7: Save model/vectorizer
│   ├── load_and_predict.py       # Step 7: Load and classify new messages
│   └── cli_predict.py            # Step 8: Command-line prediction tool
├── README.md
└── requirements.txt

🚀 Getting Started
# Clone the repository
git clone https://github.com/simabani/email-spam-classifier.git
cd email-spam-classifier

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

📊 Train and Evaluate
python src/train_models.py

🧪 Fine-Tuning (Step 6)
🔺 Threshold tuning:
python src/tune_threshold.py

🧱 Add bigrams:
python src/tune_bigrams.py

⚖️ Compare classifiers:
python src/compare_models.py

💾 Save the Model
python src/save_model.py

🔁 Load and Predict New Messages
python src/load_and_predict.py

🖥️ Use the CLI Tool
python src/cli_predict.py

Example usage:

Enter a message: You’ve won a free cruise!
→ Prediction: SPAM 🚨

Enter a message: Are you free for lunch?
→ Prediction: HAM ✅

Type exit to quit.

📚 Dataset

Source: UCI SMS Spam Collection
Format: label<TAB>message

📦 Dependencies
pandas
scikit-learn
joblib

Install via:

pip install -r requirements.txt

✅ Project Status
✅ Model training and evaluation
✅ Fine-tuning with threshold & bigrams
✅ Model comparison complete
✅ CLI-based prediction working

🚀 Ready for web or API deployment
