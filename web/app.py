import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



from flask import Flask, render_template, request
from src.train import load_model, predict_with_model  
from src.criteria import extract_features  
import pandas as pd
app = Flask(__name__)

model = load_model()  
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    feature_report = None
    if request.method == "POST":
        user_input = request.form["user_input"]
        
        
        features = extract_features(user_input)
        
        
        ai_probability, prediction, feature_report = predict_with_model(model, features)
        
        
        result = f"Probability that the text is AI-generated: {round(ai_probability, 2)}%"
        
        
        explanation = ""
        if prediction == 1:
            explanation += "This text is predicted to be AI-generated. Here are the factors that contributed to this conclusion:\n"
        else:
            explanation += "This text is predicted to be Human-generated. Here are the factors that contributed to this conclusion:\n"
        
        for feature, details in feature_report.items():
            explanation += f"- {feature}: Value = {details['value']}, Importance = {details['importance']}\n"

        feature_report = explanation
    
    return render_template("index.html", result=result, explanation=feature_report)
