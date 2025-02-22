import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

from data_processing import process_dataset

data = pd.read_csv("processed_dataset.csv")
X_vals = data.iloc[:, :-1].values
y_vals = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size = 0.25, random_state = 0)
classifier = LogisticRegression(random_state = 0, max_iter = 1000)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(accuracy_score(y_test, y_pred))