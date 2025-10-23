# Email Spam Classifier using Naive Bayes

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1️⃣ Load Dataset
# You can download 'spam.csv' from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
data = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
data.columns = ['label', 'message']

# 2️⃣ Convert labels to numeric
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# 3️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42)

# 4️⃣ Convert text to numerical vectors (Bag of Words)
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5️⃣ Train Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 6️⃣ Predict
y_pred = model.predict(X_test_vec)

# 7️⃣ Evaluate
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📈 Classification Report:\n", classification_report(y_test, y_pred))

# 8️⃣ Test with your own input
test_email = ["Congratulations! You've won a $1000 Walmart gift card. Click here to claim now."]
test_vec = vectorizer.transform(test_email)
print("\n🕵️‍♂️ Prediction:", "Spam" if model.predict(test_vec)[0] == 1 else "Not Spam")

