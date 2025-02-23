# AI-Detector  
This years theme is Canadian issues and with that in mind we decided to develop an AI  detection app. An app that detects AI-generated text can help prevent the spread of  misleading information, especially in areas like government policies, health advisories, or  community discussions.  
With the rise of social media, AI-generated content has become increasingly prevalent,  leading to potential threats in terms of fake news and disinformation. This app can be used by Canadians to ensure the authenticity of online content, making it easier to distinguish  between human-generated and AI-generated text.  

With this theme in mind enjoy our AI detection web application. We would like to be  considered for the best AI app category

## AI-Detection Web Application  

This web app is designed to detect whether a given text is written by AI or a  human. It uses machine learning models to classify text and return a  probability percentage based on the analysis. Additionally it will declare  whether the text is predicted to be human generated or AI-generated, and the  factors that contributed to the computed conclusion, alongside their  importance scores. The backend is built with Python and Flask, and the  frontend allows users to input text and get instant results.  

## Features  
Classifies text as AI-generated or human-written.  

Provides a probability percentage indicating the likelihood that the text is AI-generated.

Built with Flask for the web framework and scikit-learn for the machine learning model.

Provides breakdown of factors contributing to the computed conclusion   

## Requirements
- Flask
- scikit-learn
- pandas
- numpy
- joblib 

## Prerequisites
Python 3.8+: Ensure Python 3.8 or above is installed  
Pip: Ensure that pip is installed to manage packages

## Installation
Follow these steps to set up and run the app locally  

### 1. Clone the Repository

Visit the repository on [GitHub](https://github.com/simia0506/AI-Detector.git):  

#### Clone this repository in your local machine:

```
git clone https://github.com/simia0506/AI-Detector.git  
cd AI-Detector
```

### 2. Create a Virtual Environment
#### Create a virtual environment to manage dependencies

`python3 -m venv venv`  

#### Activate the Virtual Environment
Mac: `source venv/bin/activate`  
Windows: `venv\Scripts\activate`

### 3. Install the Required Packages

`pip install flask scikit-learn pandas numpy joblib`  

### 4. Train the Model  
#### Run the training script to train the AI vs Human classification model

`python -m src.train`

This will train the model using the data in the ai_vs_human_dataset.csv file  and save the model as ai_detection_model.pkl

### 5. Run the Flask Web Application
To start the web app, run the Flask server:  

`python -m web.app`

This will start the server, and you should see output similar to this:  

 * Running on http://127.0.0.1:5000 (Press CTRL+C to quit)  

 ### 6. Access the Web Application  
 Open your web browser and go to http://127.0.0.1:5000

 You should see the web interface where you can enter text and get the  probability that it was generated by AI

 ## Usage
 1. Enter a passage of text into the texbox  
 2. Click "Analyze Text" to get the probability percentage indicating how  likely it is that the text is AI-generated, and the conclusion on whether it  is human or AI generated
 3. The results will be displayed in percentage format  

## Folder Structure
Here is a breakdown of the projects folder structure  - 

```
AI-Detector/
│
├── src/                         # Contains the main Python code and model
│   ├── __init__.py              # Initializes the src package
│   ├── criteria.py              # Text analysis and feature extraction
│   ├── preprocess.py            # Text preprocessing (tokenization, etc.)
│   ├── train.py                 # Model training script
│   ├── main.py                  # Model loading and prediction logic
│
├── web/                         # Contains the Flask app
│   ├── app.py                   # Main Flask app
│   ├── static/                  # Contains CSS and other static files
│   │   └── style.css            # Stylesheet for the web pages
│   ├── templates/               # Contains HTML templates
│   │   ├── index.html           # Homepage template
│   │   └── result.html          # Result page template
│
├── ai_vs_human_dataset.csv      # Dataset used for training the model
├── ai_detection_model.pkl       # Trained machine learning model
├── ai_keywords.txt              # List of AI-related keywords
├── ai_phrases.txt               # List of AI-related phrases
└── README.md                    # This file
```