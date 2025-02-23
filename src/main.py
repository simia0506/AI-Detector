import pandas as pd
import nltk
# nltk.download('averaged_perceptron_tagger') // pos tags
from preprocess import preprocess_text, seperate_by_sentence
from criteria import (
    extract_features
)


def print_section_header(title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50 + "\n")

# Sample data for testing:
data = {
    'text': [
        "I love using AI. It's a really fun tool to work with!",
        "She endeavored to enhance her coding skills, transforming lines of binary into innovative solutions.",
        "AI's improved so much! It's crazy to see how much it can do now!",
        "Artificial Intelligence has advanced significantly, demonstrating impressive capabilities across a wide range of applications."
    ],
    'label': [1, 0, 1, 0]  # 1 = human written, 0 = AI-generated
}

df = pd.DataFrame(data)

# Loop through each text entry and extract features
for index, row in df.iterrows():
    text = row['text']
    label = row['label']
    
    print_section_header(f"Processing Text {index + 1}")
    print("Raw Text:")
    print(text)
    
    # Print the preprocessed text (after stopword removal)
    processed_text = preprocess_text(text)
    print("\nPreprocessed Text (Stopwords Removed):")
    print(processed_text)
    
    print_section_header("Extracted Features")
    features = extract_features(text)
    for key, value in features.items():
        print(f"{key}: {value}")
    
    print("\n" + "-" * 50 + "\n")
