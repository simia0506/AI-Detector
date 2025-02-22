# nltk.download('stopwords') // stop word removal
from nltk.corpus import stopwords
# nltk.download('punkt') // split text into tokens
from nltk.tokenize import word_tokenize, sent_tokenize

def preprocess_text(text):
    preprocessed_text = []
    stop_words = set(stopwords.words('english'))
    for i in text.split():
        i = i.lower()
        if i not in stop_words and i.isalpha():
            preprocessed_text.append(i)
    return preprocessed_text

def seperate_by_sentence(text):
    return sent_tokenize(text)