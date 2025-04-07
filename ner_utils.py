# ner_utils.py

import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")


def extract_named_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]
