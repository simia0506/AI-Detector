import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from data_processing import process_dataset

script_dir = os.path.dirname(os.path.realpath(__file__))  # Get the folder where train.py is located
data_file_path = os.path.join(script_dir, '../ai_vs_human_dataset.csv')  # Relative path to the dataset

# Check if the file exists before proceeding
if not os.path.exists(data_file_path):
    raise FileNotFoundError(f"Dataset file not found: {data_file_path}")

data = pd.read_csv("ai_vs_human_dataset.csv")


X_vals = data.iloc[:, :-1].values
y_vals = data.iloc[:, -1].values

vectorizer = TfidfVectorizer(stop_words='english')  # Ignore common English stop words
X_vals = vectorizer.fit_transform(X_vals.ravel())   

data = data.dropna()
X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size = 0.25, random_state = 0)

classifier = LogisticRegression(random_state = 0, max_iter = 1000)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))