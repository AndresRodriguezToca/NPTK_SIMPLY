
import functions


from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, ne_chunk

# FUNCTION THAT ANALIZE TEXT
def analyze_text(text):

    # PREPARE RETURN JSON
    result = {
        "text_length": len(text),
        "sentence_count": len(sent_tokenize(text)),
        "word_count": len(word_tokenize(text)),
        "named_entity_count": 0,
        "unique_word_count": 0,
        "is_meaningful": False,
        "contextual_analysis": "",
        "readability_score": functions.get_readability_score(text),
        "sentiment_score": functions.get_sentiment_score(text),
        "lexical_diversity": functions.get_lexical_diversity(text),
        "grammar_complexity": functions.get_average_sentence_length(text),
        "information_density": functions.get_information_density(text),
        "named_entity_density": functions.get_named_entity_density(text),
        "named_entities": [],
        "unique_words": [],
    }
    
    # TOKENIZE INTO SENTENCES
    sentences = sent_tokenize(text)
    result['sentence_count'] = len(sentences)
    
    # CHECK IF TEXT IS SHORTER THAN EXPECTED
    if len(sentences) < 3:
        result['contextual_analysis'] = "Text is too short to be contextually meaningful."
        return result
    
    # POS TAGGING AND NER (NAMED ENTITY RECOGNITION)
    words = word_tokenize(text)
    result['word_count'] = len(words)
    
    pos_tags = pos_tag(words)
    named_entities_tree = ne_chunk(pos_tags)
    
    # EXTRACT NER AND COUNT
    named_entities = []
    for chunk in named_entities_tree:
        if hasattr(chunk, 'label'):
            entity = " ".join(c[0] for c in chunk)
            named_entities.append(entity)
    result['named_entities'] = named_entities
    result['named_entity_count'] = len(named_entities)
    
    # UNIQUE WORDS
    unique_words = set(functions.preprocess_text(text))
    result['unique_words'] = list(unique_words)
    result['unique_word_count'] = len(unique_words)
    
    # CONDITIONS
    meaningful_conditions = [
        result['text_length'] > 50,
        result['named_entity_count'] > 3,           
        result['unique_word_count'] > 10,
        result['word_count'] > 10,
        result['readability_score'] > 80,
        result['lexical_diversity'] > 0.6,
        result['grammar_complexity'] > 5,
        result['information_density'] > 0.5,
        result['information_density'] > 0.05,
    ]

    if all(meaningful_conditions):
        result['is_meaningful'] = True
        result['contextual_analysis'] = "Text is contextually valid and meaningful."
    else:
        result['contextual_analysis'] = "Text lacks sufficient context or detail."
    
    return result
