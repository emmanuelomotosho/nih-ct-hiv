from flask import Flask, jsonify, request

import pickle
import re
import string



from sklearn.feature_selection import chi2, SelectKBest
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

model_payload_path = "models/cancer_hiv.pickle"
with open(model_payload_path, 'rb') as f:
    model_payload = pickle.load(f)

REMOVE_PUNC = str.maketrans({key: None for key in string.punctuation})

def filter_study(ec):
    """take one study and returns a filtered version with only relevant lines included"""
    lines = []
    segments = re.split(
        r'\n+|(?:[A-Za-z0-9\(\)]{2,}\. +)|(?:[0-9]+\. +)|(?:[A-Z][A-Za-z]+ )+?[A-Z][A-Za-z]+: +|; +| (?=[A-Z][a-z])',
        ec, flags=re.MULTILINE)
    for i, l in enumerate(segments):
        l = l.strip()
        if l:
            l = l.translate(REMOVE_PUNC).strip()
            if l:
                lines.append(l)
    return '\n'.join(lines)


@app.route("/", methods=['POST'])
def predict():


    text=[""]

    text[0] = request.get_json()['x']


    vectorizer = model_payload['vectorizer']


    text[0] = filter_study(text[0])

    tr = vectorizer.transform(text)

    chi2_best = model_payload['chi2_best']

    tr = chi2_best.transform(tr)



    val =str(model_payload['model'].predict(tr[0])[0])

    return val, 200
