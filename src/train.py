import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib  

from src.criteria import extract_features

df = pd.read_csv('../ai_vs_human_dataset.csv')


print("Columns in dataset:", df.columns)


def process_batch(batch_size=100):
    
    df_sample = df.sample(n=batch_size, random_state=42)  

    
    print(f"Sample size: {len(df_sample)}")

    
    X = df_sample['text']
    y = df_sample['generated']  

    
    X_features = X.apply(lambda text: extract_features(text))  

    
    X_df = pd.DataFrame(list(X_features))

    
    if X_df.isnull().values.any():
        print("Warning: Missing values detected in features, filling with 0s.")
        X_df = X_df.fillna(0)  

    
    X_df = X_df.apply(pd.to_numeric, errors='coerce').fillna(0)  


    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)

    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    
    joblib.dump(model, 'ai_detection_model.pkl')

   
    joblib.dump(extract_features, 'feature_extraction_function.pkl')




def load_model():
    model = joblib.load('ai_detection_model.pkl') 
    return model


def predict_with_model(model, features):

    if isinstance(features, dict): 
        features = pd.DataFrame([features])
    elif isinstance(features, list) or isinstance(features, np.ndarray):  
        features = np.array(features).reshape(1, -1)  
    
    
    probability = model.predict_proba(features)
    prediction = model.predict(features)

    
    feature_report = {}
    feature_names = list(features.columns)  
    feature_importance = model.feature_importances_  

    for i, feature_name in enumerate(feature_names):
        feature_report[feature_name] = {
            "value": features[feature_name].iloc[0],
            "importance": feature_importance[i]
        }
    
    
    ai_probability = float(probability[0][1]) * 100

 
    return ai_probability, prediction[0], feature_report
process_batch(batch_size=100)
