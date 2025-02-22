# nltk.download('vader_lexicon') // sentiment analysis

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
import statistics
import spacy
nlp = spacy.load("en_core_web_sm")

# structure variation
import statistics
from nltk import pos_tag

# for repetition
from collections import Counter
from preprocess import preprocess_text, seperate_by_sentence
import math
from nltk.util import bigrams
from nltk import FreqDist
from nltk.probability import MLEProbDist

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
    if isinstance(text, list):
        text = " ".join(text)
    text = text.lower()
    for i in ai_phrases:
        count += text.count(i)
    return count

def sentence_complexity(text):
    sentences = sent_tokenize(text)
    complexity_scores = []
    
    for sentence in sentences:
        words = word_tokenize(sentence)
        pos_tags = pos_tag(words)

        tags_set = set()
        for word, tag in pos_tags:
            tags_set.add(tag)

        if len(pos_tags) > 0:
            complexity_score = len(tags_set) / len(pos_tags) if len(pos_tags) != 0 else 0  # check for zero
        else:
            complexity_score = 0
        
        complexity_scores.append(complexity_score)

    if len(complexity_scores) == 0:
        return 0
    else:
        total_complexity = 0

        for score in complexity_scores:
            total_complexity += score
        return total_complexity / len(complexity_scores) if len(complexity_scores) != 0 else 0
# - AI-generated sentences are often less complex with fewer unique parts of speech (e.g., adjectives, adverbs).
# - Human-written text is typically more diverse, involving multiple sentence structures and a wide range of vocabulary and grammatical elements.


def sentence_starter_variation(text):
    start_types = {"pronoun": 0, "verb": 0, "adverb": 0, "noun": 0, "conjunction": 0}
    sentences = sent_tokenize(text)

    for sentence in sentences:
        words = word_tokenize(sentence)
        if words:
            first_word = words[0].lower()
            tagged_pos = pos_tag([first_word])[0][1]  # Renamed to 'tagged_pos'

            if tagged_pos in ["PRP", "PRP$", "WP", "WP$"]:  # Pronoun
                start_types["pronoun"] += 1
            elif tagged_pos.startswith("VB"):  # Verb
                start_types["verb"] += 1
            elif tagged_pos.startswith("RB"):  # Adverb
                start_types["adverb"] += 1
            elif tagged_pos.startswith("NN"):  # Noun
                start_types["noun"] += 1
            elif tagged_pos.startswith("CC"):  # Conjunction
                start_types["conjunction"] += 1

    total_sentences = len(sentences)
    result = {}
    for start_type, count in start_types.items():
        result[start_type] = count / total_sentences  # Calculating the ratio

    return result
# - AI-generated content typically has fewer subordinate clauses (dependent clauses), making sentences more direct and less complex.
# - Humans use more subordinate clauses to create nuanced, complex thoughts, offering a greater variety of sentence types.
def subordinate_clause_ratio(text):
    if isinstance(text, list):
        text = " ".join(text)
    subordinate_conjunctions = {"although", "because", "since", "unless", "if", "while", "though"}
    sentences = sent_tokenize(text)
    total_clauses = 0
    subordinate_clauses = 0
    for i in sentences:
        words = word_tokenize(i)
        for j in words:
            if j.lower() in subordinate_conjunctions:
                subordinate_clauses += 1
            total_clauses += 1
    if total_clauses > 0:
        result = subordinate_clauses / total_clauses 
    else:
        result = 0

    return result

# - AI-generated text often tends to follow a more rigid and repetitive sentence structure (e.g., "SV" or "SVC") due to its lack of variety in sentence composition.
# - Human-written text tends to have more varied structures, such as a balance of "SVO" and "SV" patterns.
def sentence_structure_pattern(text):
    sentences = sent_tokenize(text)
    structure_patterns = []

    for sentence in sentences:
        words = word_tokenize(sentence)
        pos_tags = pos_tag(words) 
        pattern = [tag for _, tag in pos_tags]
        structure_patterns.append(" ".join(pattern))

    structure_pattern_frequency = {}
    for pattern in structure_patterns:
        if pattern in structure_pattern_frequency:
            structure_pattern_frequency[pattern] += 1
        else:
            structure_pattern_frequency[pattern] = 1

    return structure_pattern_frequency

def analyze_sentence_structure(text):
    return {
        "structure_pattern_frequency": sentence_structure_pattern(text),
        "subordinate_clause_ratio": subordinate_clause_ratio(text),
        "sentence_starter_variation": sentence_starter_variation(text),
        "sentence_complexity_score": sentence_complexity(text)
    }
    
def perplexity_level(text):
    # ai has low purplexity (predictable)
    #tokenize input text
    perplexity = 0
    tokens = nltk.word_tokenize(text)

    #Generate bigrams from tokens - pairs of 2 words
    bigram_list = list(bigrams(tokens))

    #create a frequency distrbution of the bigrams
    fdist = FreqDist(bigram_list)

    #create a bigram model using the Max likelihood estimation (MLE)
    bigram_model = MLEProbDist(fdist)

    preplexity = 0.0
    n = len(tokens)

    #loop thorugh tokens starting from the 2nd token
    for i in range (1, n):
        prev = tokens [i - 1]
        curr = tokens [i]

        #get the prob of the bigram
        prob = bigram_model.prob((prev, curr))

        #if the prob is 0, assign a little value to avoid math.log(0)
        if prob == 0:
            prob = 1e-10
        
        #add the log of the probability ot the perplexity sum
        perplexity += math.log(prob)
    
    #calculate the final perplexity
    perplexity = math.exp(-1 * perplexity / n)

    return perplexity



# REMOVE STOPWORDS
def lemmatization(text):
    doc = nlp(text)
    lemm_words = []

    for token in doc:
        if token.is_alpha:
            lemm_words.append(token.lemma_)
    
    # lemm_words = " ".join(lemm_words)

    unique_lemm_words = len(lemm_words) / len(doc) if len(doc) != 0 else 0  # check for zero

    return unique_lemm_words

# def detect_formality(text):
def detect_formality(text):
    doc = nlp(text)

    formal_words = {"moreover", "therefore", "thus", "hence", "furthermore", "consequently", "notwithstanding", "whereas", "henceforth"}
    informal_words = {"gonna", "wanna", "gotta", "ya", "dunno", "lemme", "gimme", "kinda", "sorta", "ain't", "bro", "dude", "omg", "lol"}

    word_counter = Counter()

    for token in doc:
        if token.is_alpha:
            word_counter[token.text.lower()] += 1
    
    formal = 0
    for word in formal_words:
        if word in word_counter:
            formal += word_counter[word]
    
    informal = 0
    for word in informal_words:
        if word in word_counter:
            informal += word_counter[word]
    
    pronoun = 0
    for token in doc:
        if token.pos == "PRON":
            pronoun += 1

    contraction = 0
    for token in doc:
        if "'" in token.text:
            contraction += 1

    formality_score = (formal - informal) - (pronoun + contraction)

    # formality = "Neutral"

    # if formality_score > 2:
    #     formality = "Formal"
    # elif formality_score < -2:
    #     formality = "Informal"
    
    return formality_score

# def repetition(text):
#     if type(text) == list:  # check if list
#         splittext = text  # if its a list use it directly
#     else:
#         splittext = text.split() # other wise split

#     counts = Counter(text)
#     repeated_words = {}

#     for i, frequency in counts.items():
#         if frequency > 1:
#             repeated_words[i] = frequency
#     return repeated_words

def common_ai_keywords(text):

    count = 0
    if isinstance(text, list):
        text = " ".join(text)
    text = text.lower()
    for i in ai_keywords:
        count += text.count(i)
    return count

# def extract_features(text):
#     #dictionary to store the feature values
#     features = {}

#     #sentiment
#     sentiment = sentiment_analysis(text)
#     #compund is a single number between -1 and 1 that gives the overall sentiment of the text
#     #-1 => negative, 0 => neutral, +1 => positive
#     features["sentinment_compound"] = sentiment["compound"]

#     #AI phrase count
#     phrase_count = common_ai_phrases(text)
#     features["ai_phrase_count"] = phrase_count

#     #AI keyword count
#     features["ai_keyword_count"] = common_ai_keywords(text)

#     #Perplexity
#     features["perplexity"] = perplexity_level(text)

#     #formality
#     features["formality"] = detect_formality(text)

#     #lemmatization score (ratio of unique lemmas)
#     features["lemma_ratio"] = lemmatization(text)

#     #sentence length variation
#     features["sentence_length_variation"] = sentence_length_variation(text)

#     #sentence complexity
#     features["sentence_complexity"] = sentence_complexity(text)

#     #sentence starter variation
#     features["sentence_starter_variation"] = sentence_starter_variation(text)

#     #subordinate clause ratio
#     features["subordinate_clause_ratio"] = subordinate_clause_ratio(text)

#     #sentence structure pattern
#     features["sentence_structure_pattern"] = sentence_structure_pattern(text)

# def extract_features(text):
#     # First, define raw text and processed text (after stopword removal)
#     raw_text = text
#     processed_text = preprocess_text(text)  # Assumes this function removes stopwords and returns a clean string

#     features = {}

#     # Features computed on the raw (unprocessed) text:
#     features["sentiment_compound"] = sentiment_analysis(raw_text)["compound"]
#     features["sentence_length_variation"] = sentence_length_variation(raw_text)
#     features["sentence_complexity"] = sentence_complexity(raw_text)
#     features["sentence_starter_variation"] = sentence_starter_variation(raw_text)
#     features["subordinate_clause_ratio"] = subordinate_clause_ratio(raw_text)
#     features["sentence_structure_pattern"] = sentence_structure_pattern(raw_text)
#     features["perplexity"] = perplexity_level(raw_text)
#     features["ai_phrase_count"] = common_ai_phrases(raw_text)

#     # Features computed on the processed text (after stopword removal):
#     features["ai_keyword_count"] = common_ai_keywords(processed_text)
#     features["lemma_ratio"] = lemmatization(processed_text)
#     features["formality"] = detect_formality(processed_text)

#     return features


#     return features

def extract_features(text):
    # Raw text for features computed before stopword removal.
    raw_text = text
    
    # Process text: preprocess_text returns a list of tokens (stopwords removed).
    processed_tokens = preprocess_text(text)
    # Join tokens into a string for functions that expect a string.
    processed_text = " ".join(processed_tokens)
    
    features = {}
    
    # Features computed on raw (unprocessed) text:
    features["ai_phrase_count"] = common_ai_phrases(raw_text)  # Compute AI phrase count on raw text
    features["sentiment_compound"] = sentiment_analysis(raw_text)["compound"]
    features["sentence_length_variation"] = sentence_length_variation(raw_text)
    features["sentence_complexity"] = sentence_complexity(raw_text)
    features["sentence_starter_variation"] = sentence_starter_variation(raw_text)
    features["subordinate_clause_ratio"] = subordinate_clause_ratio(raw_text)
    features["sentence_structure_pattern"] = sentence_structure_pattern(raw_text)
    features["perplexity"] = perplexity_level(raw_text)
    
    # Features computed on processed text (after stopword removal):
    features["ai_keyword_count"] = common_ai_keywords(processed_text)
    features["lemma_ratio"] = lemmatization(processed_text)
    features["formality"] = detect_formality(processed_text)
    
    return features
