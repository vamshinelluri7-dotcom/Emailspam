A simple Machine Learning project that classifies emails as Spam or Not Spam using Python, Scikit-learn, and Naive Bayes. This project demonstrates text preprocessing, feature extraction, and model evaluation for email spam detection.
 Features

Detects whether an email is spam or ham (not spam)

Uses Bag of Words for text representation

Implements Naive Bayes Classifier for high accuracy

Easy to test with custom email inputs

Simple and clean code — great for beginners and college projects

 Technologies Used

Python 3

Pandas

Scikit-learn

NumPy

 Dataset

Dataset used: SMS Spam Collection Dataset

Rename the downloaded file to spam.csv and place it in your project folder.

🧩 How It Works

Load and clean the dataset

Convert text into numerical vectors using CountVectorizer

Train the model using MultinomialNB

Test the model and evaluate performance

Predict new email messages for spam detection

Installation
git clone https://github.com/your-username/email-spam-classifier.git
cd email-spam-classifier
pip install pandas numpy scikit-learn

Usage

Run the Python script:

python spam_classifier.py


Example output:

 Accuracy: 98%
 Prediction: Spam

 Results

Achieved accuracy around 97–99%

Performs well on text-based spam detection tasks

 Author

Your Name
 your.email@example.com

 LinkedIn
