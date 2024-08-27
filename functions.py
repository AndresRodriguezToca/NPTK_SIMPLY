import textstat

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer

# GET READABILITY SCORE
def get_readability_score(text):
    readability = textstat.flesch_reading_ease(text)
    return readability

# GET SENTIMENT SCORE
def get_sentiment_score(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

# GET LEXICAL DIVERSITY
def get_lexical_diversity(text):
    words = word_tokenize(text)
    unique_words = set(words)
    lexical_diversity = len(unique_words) / len(words)
    return lexical_diversity

# GRAMMATICAL COMPLEXITY
def get_average_sentence_length(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    avg_sentence_length = len(words) / len(sentences)
    return avg_sentence_length

# INFORMATION DENSITY
def get_information_density(text):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    content_words = [word for word, pos in pos_tags if pos.startswith(('NN', 'VB', 'JJ', 'RB'))]
    density = len(content_words) / len(words)
    return density

# NAMED ENTITY DENSITY
def get_named_entity_density(text):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    named_entities = ne_chunk(pos_tags)
    named_entity_count = sum(1 for chunk in named_entities if hasattr(chunk, 'label'))
    density = named_entity_count / len(words)
    return density

# FUNCTION THAT PROCESS DATA (EXPECTS A STRING)
def preprocess_text(text):
    # TOKENIZE
    words = word_tokenize(text.lower())
    
    # REMOVE STOPWORDS
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    
    # RETURN PROCESSED TEXT
    return filtered_words