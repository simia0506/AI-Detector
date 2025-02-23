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


# import pandas as pd
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.feature_extraction.text import TfidfVectorizer
# from data_processing import process_dataset, extract_features  # Assuming extract_features is defined elsewhere
# import numpy as np  # For random sampling

# # Define paths
# script_dir = os.path.dirname(os.path.realpath(__file__))  # Get the folder where train.py is located
# data_file_path = os.path.join(script_dir, '../ai_vs_human_dataset.csv')  # Path to dataset

# # Check if the file exists before proceeding
# if not os.path.exists(data_file_path):
#     raise FileNotFoundError(f"Dataset file not found: {data_file_path}")

# # Debugging statement to confirm data file existence
# print(f"Data file found: {data_file_path}")

# # Load the full dataset
# try:
#     processed_data = pd.read_csv(data_file_path)
#     print(f"Processed data loaded: {processed_data.shape[0]} rows")
# except Exception as e:
#     print(f"Error loading processed data: {e}")
#     exit()

# # Randomly sample 1/4 of the data
# sampled_data = processed_data.sample(frac=0.25, random_state=42)  # Adjust the fraction for the desired amount (0.25 = 1/4)

# print(f"Sampled {len(sampled_data)} rows from the dataset.")

# # Ensure the required columns are present (if the process_dataset function expects specific ones)
# if 'text' not in sampled_data.columns or 'generated' not in sampled_data.columns:
#     raise ValueError("Expected columns 'text' and 'generated' are missing in the sampled data.")

# # Preprocessing: Extract features using the extracted features or vectorizer
# X_vals = sampled_data['text']
# y_vals = sampled_data['generated']

# try:
#     vectorizer = TfidfVectorizer(stop_words='english')  # Removing common English stop words
#     X_vals_transformed = vectorizer.fit_transform(X_vals)  # Transform the text data
#     print("Data vectorized successfully.")
# except Exception as e:
#     print(f"Error during vectorization: {e}")
#     exit()

# # Split data into train and test sets
# try:
#     X_train, X_test, y_train, y_test = train_test_split(X_vals_transformed, y_vals, test_size=0.25, random_state=0)
#     print("Data split into train and test sets.")
# except Exception as e:
#     print(f"Error during data split: {e}")
#     exit()

# # Train a Logistic Regression classifier
# try:
#     classifier = LogisticRegression(random_state=0, max_iter=1000)
#     classifier.fit(X_train, y_train)
#     print("Classifier trained successfully.")
# except Exception as e:
#     print(f"Error during model training: {e}")
#     exit()

# # Predict on the test set
# try:
#     y_pred = classifier.predict(X_test)
#     print("Predictions made on test set.")
# except Exception as e:
#     print(f"Error during prediction: {e}")
#     exit()

# # Print accuracy and classification report
# try:
#     print("Accuracy:", accuracy_score(y_test, y_pred))
#     print(classification_report(y_test, y_pred))
# except Exception as e:
#     print(f"Error during evaluation: {e}")
#     exit()

# # Now, process the dataset and use the trained model for predictions
# output_file_path = os.path.join(script_dir, 'predicted_output.csv')

# # Use the classifier to process data, predict labels and save the results
# def process_and_predict(input_df, output_fp, classifier, vectorizer):
#     try:
#         # Ensure that the input dataframe has the correct columns
#         print(f"Processing input data with {input_df.shape[0]} rows.")
        
#         # Feature extraction using the same vectorizer as training
#         X_vals = vectorizer.transform(input_df['text'])  # Transform new text data into the same feature space

#         # Predict labels using the trained classifier
#         predictions = classifier.predict(X_vals)
        
#         # Add predictions as a new column
#         input_df['Predicted_Label'] = predictions

#         # Save the dataframe with predictions
#         input_df.to_csv(output_fp, index=False)
#         print(f"Processed dataset saved to {output_fp}")
#     except Exception as e:
#         print(f"Error during prediction and saving: {e}")

# # Now, process the dataset and get predictions
# process_and_predict(sampled_data, output_file_path, classifier, vectorizer)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib  # To save the trained model
#from criteria import extract_features  # Import your feature extraction function
from src.criteria import extract_features
# Load the dataset
df = pd.read_csv('../ai_vs_human_dataset.csv')

# Check the columns
print("Columns in dataset:", df.columns)

# Create a function to process 100 samples at a time
def process_batch(batch_size=100):
    # Select a random subset of the dataset
    df_sample = df.sample(n=batch_size, random_state=42)  # Process only 100 samples at a time

    # Check the sample size
    print(f"Sample size: {len(df_sample)}")

    # Assuming the dataset has columns: 'text' (the text sample) and 'generated' (either '1' for AI, '0' for Human)
    X = df_sample['text']
    y = df_sample['generated']  # 'generated' is assumed to be binary (1 = AI, 0 = Human)

    # Convert the text data into feature vectors using the extract_features function
    X_features = X.apply(lambda text: extract_features(text))  # Uses your extract_features function

    # Flatten the list of dictionaries into a DataFrame
    # Convert each dictionary into a row of values
    X_df = pd.DataFrame(list(X_features))

    # Check if the DataFrame has any missing values
    if X_df.isnull().values.any():
        print("Warning: Missing values detected in features, filling with 0s.")
        X_df = X_df.fillna(0)  # Fill missing values with 0

    # Ensure X_df contains only numerical values
    X_df = X_df.apply(pd.to_numeric, errors='coerce').fillna(0)  # Convert non-numeric values to 0

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

    # Initialize the model (Random Forest Classifier here)
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model using the feature vectors extracted from the text
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)

    # Print the classification report and accuracy
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Save the trained model to a file
    joblib.dump(model, 'ai_detection_model.pkl')

    # Optionally, save the feature extraction function as well
    joblib.dump(extract_features, 'feature_extraction_function.pkl')

# Process a batch of 100 samples at a time


def load_model():
    model = joblib.load('ai_detection_model.pkl')  # Load the trained model
    return model

# def predict_with_model(model, features):
#     # Assuming features is a DataFrame or list of numerical features
#     prediction = model.predict(features)  # Predict class
#     probability = model.predict_proba(features)  # Probability for each class
#     return probability[0][1]  # Assuming class 1 is "AI"
def predict_with_model(model, features):
    # Convert features to a 2D array if it's not already
    if isinstance(features, dict):  # If features is a dictionary, convert to a DataFrame
        features = pd.DataFrame([features])
    elif isinstance(features, list) or isinstance(features, np.ndarray):  
        features = np.array(features).reshape(1, -1)  # Ensure it's a 2D array
    
    # Predict class probabilities
    probability = model.predict_proba(features)
    prediction = model.predict(features)

    # Generate a report on the feature importance or values
    feature_report = {}
    feature_names = list(features.columns)  # Column names of the features
    feature_importance = model.feature_importances_  # Get feature importances

    for i, feature_name in enumerate(feature_names):
        feature_report[feature_name] = {
            "value": features[feature_name].iloc[0],
            "importance": feature_importance[i]
        }
    
    # Normalize the probability to percentage
    ai_probability = float(probability[0][1]) * 100

    # Return the prediction, probability, and feature report
    return ai_probability, prediction[0], feature_report
process_batch(batch_size=100)
# Load the pre-trained model from file
# def load_model():
#     # Replace with actual model path
#     model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'src', 'ai_detection_model.pkl')
#     return joblib.load(model_path)

# # Make predictions using the loaded model
# def predict_with_model(model, input_text):
#     # Extract features from the input text
#     features = extract_features(input_text)
    
#     # Make a prediction with the model
#     prediction = model.predict([features])  # Assuming the model expects a list of features
#     probability = model.predict_proba([features])[0][1]  # Probability of the positive class (AI-generated)
    
#     return probability