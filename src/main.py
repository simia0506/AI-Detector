import pandas as pd
import nltk
from preprocess import preprocess_text, seperate_by_sentence
from criteria import sentence_length_variation, sentiment_analysis, common_ai_keywords, common_ai_phrases, repetition

data = {
        'text': [
            "I love using AI. It's a really fun tool to work with!",
            "She endeavored to enhance her coding skills, transforming lines of binary into innovative solutions.",
            "AI's improved so much! It's crazy to see how much it can do now!",
            "Artificial Intelligence has advanced significantly, demonstrating impressive capabilities across a wide range of applications."
        ],
        'label': [1, 0, 1, 0]  # 1 = human written, 0 = AI
    }

df = pd.DataFrame(data)
# print(df)
# print()

# SEPERATE BY SENTENCE
# Create a list of lists where each sublist contains sentences for a particular text entry
separated_sentences = [seperate_by_sentence(text) for text in data['text']]
for i, sentences in enumerate(separated_sentences):
    print(f"Quote {i+1}: {sentences}")

for i, text in enumerate(data['text']):
    print(f"Text {i+1}:")
    sentiment_score = sentiment_analysis(text)
    
    # Show sentiment score for the entire text
    print(f"  Sentiment Score: {sentiment_score}")
    print()

sentence_lengths_results = [sentence_length_variation(text) for text in data['text']]
print(sentence_lengths_results)

for i, text in enumerate(data['text']):
    ai_phrase_count = common_ai_phrases(text)
    ai_keyword_count = common_ai_keywords(text)
    
    print(f"Text {i + 1}: {ai_phrase_count} AI-related phrases found, {ai_keyword_count} AI-related keywords found")




# REMOVE STOP WORDS AND CHANGE TO LOWERCASE, THEN CHECK REPETITION
for i, text in enumerate(data['text']):
    # Preprocess the text (remove stopwords and convert to lowercase)
    processed_text = preprocess_text(text)
    print(f"\nProcessed Text {i + 1}:")
    print(processed_text)

    # Check for repetition in the processed text
    repeated_words = repetition(processed_text)
    print(f"Repeated Words in Text {i + 1}:")
    print(repeated_words)
    print()




