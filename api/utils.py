import pandas as pd
import numpy as np
import pickle
from pathlib import PurePath

def load_model(path):
    assert type(path) == str
    model = pickle.load(open(path, 'rb'))
    return model


def load_tfidf(path):
    assert type(path) == str
    tfidf = pickle.load(open(path, 'rb'))
    return tfidf
