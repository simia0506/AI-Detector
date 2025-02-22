# nltk.download('vader_lexicon') // sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
import statistics
import spacy
nlp = spacy.load("en_core_web_sm")


# for repetition
from collections import Counter

ai_keywords = [
    "adhere",
    "beacon",
    "binary",
    "bustling",
    "convey",
    "delve",
    "elevate",
    "embark",
    "emerge",
    "endeavor",
    "enhance",
    "enlighten",
    "esteemed",
    "explore",
    "foster",
    "groundbreaking",
    "imperative",
    "indelible",
    "interplay",
    "journey",
    "leverage",
    "multifaceted",
    "nuance",
    "pivotal",
    "plethora",
    "realm",
    "refrain",
    "resonate",
    "robust",
    "seamless",
    "tapestry",
    "testament",
    "underscore",
    "unleash",
    "whimsical"
]
ai_phrases = [
    "a complex multifaceted",
    "a nuanced understanding",
    "a profound implication",
    "a significant implication",
    "a stark reminder",
    "add depth to",
    "an unwavering commitment",
    "crucial role in understanding",
    "delve deeper into",
    "finding a contribution",
    "gain a comprehensive understanding",
    "left an indelible mark",
    "make an informed decision in regard to",
    "meticulous attention to",
    "navigate the complex",
    "offer a valuable",
    "open a new avenue",
    "play a crucial role in determining",
    "play a significant role in shaping",
    "provide a comprehensive overview",
    "provide a valuable insight",
    "shed light on",
    "the complex interplay",
    "the multifaceted nature",
    "the relentless pursuit",
    "the transformative power",
    "underscore the importance",
    "underscore the need",
    "vital role in shaping"
]
sia = SentimentIntensityAnalyzer()
def sentiment_analysis(text):
    sentiment_score = sia.polarity_scores(text)
    return sentiment_score

def sentence_length_variation(text):
    sentences = sent_tokenize(text)
    sentence_lengths = []
    for sentence in sentences:
        current = []
        for word in word_tokenize(sentence):
            if word.isalpha():
                current.append(word)
        sentence_lengths.append(len(current))

    if len(sentence_lengths) > 1:
        variation = statistics.stdev(sentence_lengths)
    else:
        variation = 0
    return variation

def common_ai_phrases(text):
    count = 0
    text = text.lower()
    for i in ai_phrases:
        count += text.count(i)
    return count

# def sentence_structure_variation(text):

    
# def perplexity_level(text):
    # ai has low purplexity (predictable)

def repetition(text):
    splittext = text.split()
    counts = Counter(splittext)

    repeated_words = {}
    for i, frequency in counts.items():
        if frequency > 1:
            repeated_words[i] = frequency
    return repeated_words

        


# REMOVE STOPWORDS
# def lemmatization(text):
# def detect_formality(text):

def repetition(text):
    if type(text) == list:  # check if list
        splittext = text  # if its a list use it directly
    else:
        splittext = text.split() # other wise split

    counts = Counter(text)
    repeated_words = {}

    for i, frequency in counts.items():
        if frequency > 1:
            repeated_words[i] = frequency
    return repeated_words

def common_ai_keywords(text):
    count = 0
    text = text.lower()
    for i in ai_keywords:
        count += text.count(i)
    return count