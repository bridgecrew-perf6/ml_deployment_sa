from fastapi import FastAPI, status
from .utils import load_model
from . import schemas
from .test import data_preprocessing, convert_text_to_tfidf, model

app = FastAPI()

text_to_int_label = {'positive' :0, 'neutral' :1, 'negative' :2}
int_to_text_label = {j:i for i, j in text_to_int_label.items()}

@app.post("/predict")
def predict(txt: schemas.predictCreate):
    txt = txt.text
    text = data_preprocessing(txt)
    tfidf_x = convert_text_to_tfidf(text)
    y_test = model.predict(tfidf_x.todense())
    return int_to_text_label[y_test.tolist()[0]]