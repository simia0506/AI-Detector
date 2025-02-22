# import pandas as pd
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import TfidfVectorizer

# from data_processing import process_dataset

# script_dir = os.path.dirname(os.path.realpath(__file__))  # Get the folder where train.py is located
# data_file_path = os.path.join(script_dir, '../ai_vs_human_dataset.csv')  # Relative path to the dataset

# # Check if the file exists before proceeding
# if not os.path.exists(data_file_path):
#     raise FileNotFoundError(f"Dataset file not found: {data_file_path}")

# data = pd.read_csv("ai_vs_human_dataset.csv")


# X_vals = data.iloc[:, :-1].values
# y_vals = data.iloc[:, -1].values

# vectorizer = TfidfVectorizer(stop_words='english')  # Ignore common English stop words
# X_vals = vectorizer.fit_transform(X_vals.ravel())   

# data = data.dropna()
# X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size = 0.25, random_state = 0)

# classifier = LogisticRegression(random_state = 0, max_iter = 1000)
# classifier.fit(X_train, y_train)

# y_pred = classifier.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# def process_dataset (input_fp, output_fp):
#     df = pd.read_csv(input_fp)
    
#     #rename generated label to Label
#     #dependent on csv file format
#     df.rename(columns={"text": "Text", "generated": "Label"}, inplace = True)

#     feature_list = []
#     for text in df["Text"]:
#         features = extract_featuress(text)
#         feature_list.append(features)
    
#     # feature_dicts = []
#     # for feature in feature_list:
#     #     feature_dicts.append(feature)
    
#     feature_df = pd.DataFrame(feature_list)

#     feature_df["Label"] = df["Label"]

#     # labels = []
#     # for label in df["Label"]:
#     #     labels.append(label)
    
#     # feature_df["Label"] = labels

#     feature_df.to_csv(output_fp, index = False)


# import pandas as pd
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import TfidfVectorizer

# from data_processing import process_dataset

# script_dir = os.path.dirname(os.path.realpath(__file__))  # Get the folder where train.py is located
# # data_file_path = os.path.join(script_dir, '../ai_vs_human_dataset.csv')  # Relative path to the dataset

# data_file_path = os.path.join(script_dir, 'out.csv')  # Relative path to the dataset


# process_dataset("ai_vs_human_dataset.csv", data_file_path)

# # Check if the file exists before proceeding
# if not os.path.exists(data_file_path):
#     raise FileNotFoundError(f"Dataset file not found: {data_file_path}")

# # data = pd.read_csv("ai_vs_human_dataset.csv")
# data = pd.read_csv(data_file_path)


# # X_vals = data.iloc[:, :-1].values
# # y_vals = data.iloc[:, -1].values

# X_vals = data.drop(columns=["Label"])
# y_vals = data["Label"]

# # vectorizer = TfidfVectorizer(stop_words='english')  # Ignore common English stop words
# # X_vals = vectorizer.fit_transform(X_vals.ravel())   

# # data = data.dropna()
# X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, test_size = 0.25, random_state = 0)

# classifier = LogisticRegression(random_state = 0, max_iter = 1000)
# classifier.fit(X_train, y_train)

# y_pred = classifier.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))


import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from data_processing import process_dataset, extract_features  # Assuming extract_features is defined elsewhere
import numpy as np  # For random sampling

# Define paths
script_dir = os.path.dirname(os.path.realpath(__file__))  # Get the folder where train.py is located
data_file_path = os.path.join(script_dir, '../ai_vs_human_dataset.csv')  # Path to dataset

# Check if the file exists before proceeding
if not os.path.exists(data_file_path):
    raise FileNotFoundError(f"Dataset file not found: {data_file_path}")

# Debugging statement to confirm data file existence
print(f"Data file found: {data_file_path}")

# Load the full dataset
try:
    processed_data = pd.read_csv(data_file_path)
    print(f"Processed data loaded: {processed_data.shape[0]} rows")
except Exception as e:
    print(f"Error loading processed data: {e}")
    exit()

# Randomly sample 1/4 of the data
sampled_data = processed_data.sample(frac=0.25, random_state=42)  # Adjust the fraction for the desired amount (0.25 = 1/4)

print(f"Sampled {len(sampled_data)} rows from the dataset.")

# Ensure the required columns are present (if the process_dataset function expects specific ones)
if 'text' not in sampled_data.columns or 'generated' not in sampled_data.columns:
    raise ValueError("Expected columns 'text' and 'generated' are missing in the sampled data.")

# Preprocessing: Extract features using the extracted features or vectorizer
X_vals = sampled_data['text']
y_vals = sampled_data['generated']

try:
    vectorizer = TfidfVectorizer(stop_words='english')  # Removing common English stop words
    X_vals_transformed = vectorizer.fit_transform(X_vals)  # Transform the text data
    print("Data vectorized successfully.")
except Exception as e:
    print(f"Error during vectorization: {e}")
    exit()

# Split data into train and test sets
try:
    X_train, X_test, y_train, y_test = train_test_split(X_vals_transformed, y_vals, test_size=0.25, random_state=0)
    print("Data split into train and test sets.")
except Exception as e:
    print(f"Error during data split: {e}")
    exit()

# Train a Logistic Regression classifier
try:
    classifier = LogisticRegression(random_state=0, max_iter=1000)
    classifier.fit(X_train, y_train)
    print("Classifier trained successfully.")
except Exception as e:
    print(f"Error during model training: {e}")
    exit()

# Predict on the test set
try:
    y_pred = classifier.predict(X_test)
    print("Predictions made on test set.")
except Exception as e:
    print(f"Error during prediction: {e}")
    exit()

# Print accuracy and classification report
try:
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
except Exception as e:
    print(f"Error during evaluation: {e}")
    exit()

# Now, process the dataset and use the trained model for predictions
output_file_path = os.path.join(script_dir, 'predicted_output.csv')

# Use the classifier to process data, predict labels and save the results
def process_and_predict(input_df, output_fp, classifier, vectorizer):
    try:
        # Ensure that the input dataframe has the correct columns
        print(f"Processing input data with {input_df.shape[0]} rows.")
        
        # Feature extraction using the same vectorizer as training
        X_vals = vectorizer.transform(input_df['text'])  # Transform new text data into the same feature space

        # Predict labels using the trained classifier
        predictions = classifier.predict(X_vals)
        
        # Add predictions as a new column
        input_df['Predicted_Label'] = predictions

        # Save the dataframe with predictions
        input_df.to_csv(output_fp, index=False)
        print(f"Processed dataset saved to {output_fp}")
    except Exception as e:
        print(f"Error during prediction and saving: {e}")

# Now, process the dataset and get predictions
process_and_predict(sampled_data, output_file_path, classifier, vectorizer)
