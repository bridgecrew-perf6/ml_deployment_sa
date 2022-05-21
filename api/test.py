import pandas as pd 
import numpy as np
import spacy
import enchant
import string
from .utils import load_tfidf, load_model


d = enchant.Dict('en_US')
nlp = spacy.load('en_core_web_sm')
punctuation = string.punctuation + '.'
vec = load_tfidf(r'api\tfidf.pkl')
model = load_model(r'api\multinomialNB.pkl')


def data_preprocessing(input_text):
    ml_data = pd.DataFrame({'text':pd.Series(input_text)})
    ml_data.text = ml_data.text.apply(lambda x: ' '.join([token.text.lower() for token in nlp(x)]))
    ml_data.text = ml_data.text.str.replace(r'^@\w+','', regex = True)
    ml_data.text = ml_data.text.str.replace(r'\d+','', regex = True)
    ml_data.text = ml_data.text.apply(lambda x: ' '.join([token.lemma_.lower() for token in nlp(x) if d.check(token.lemma_)]))
    ml_data.text = ml_data.text.apply(lambda x: ' '.join([token.text for token in nlp(x) if token.text not in punctuation]).strip())
    ml_data.text = ml_data.text.apply(lambda x: ' '.join([token.text for token in nlp(x) if token.text not in nlp.Defaults.stop_words]).strip())

    return ml_data.text[0]


def convert_text_to_tfidf(text):
    text = [text]
    tfidf_X = vec.transform(text)
    return tfidf_X

