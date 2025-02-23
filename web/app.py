import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



from flask import Flask, render_template, request
from src.train import load_model, predict_with_model  # Importing the correct functions from train.py
from src.criteria import extract_features  # Import the feature extraction function
import pandas as pd
app = Flask(__name__)
# model = joblib.load('ai_detection_model.pkl')
# extract_features = joblib.load('feature_extraction_function.pkl')
# Load the model at the start of the application
model = load_model()  # Load the pre-trained model once when the app starts

@app.route("/", methods=["GET", "POST"])
def home():
    result = None  # Initialize result to None on page load
    if request.method == "POST":
        user_input = request.form["user_input"]  # Capturing the user input

        if user_input:  # If the user has entered something new
            # Extract features from the user input
            features = extract_features(user_input)
            
            # Predict using the loaded model
            probability = predict_with_model(model, features)
            features_df = pd.DataFrame([features])
            
            # Format the result as a percentage
            result = f"Probability that the text is AI-generated: {round(probability, 2) * 100}%"
        else:
            # If there's no input, keep result as None (clear the result)
            result = None

    # Render the index.html template and pass the result to it
    return render_template("index.html", result=result)

# if __name__ == "__main__":
#     app.run(debug=True)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
