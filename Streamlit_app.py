import streamlit as stl
import pickle
import nltk
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
import string

model = pickle.load(open('Sucide_model.pkl', 'rb'))
cv = pickle.load(open('vectorixer.pkl', 'rb'))
stl.title('Suicided Classifier')
input_message = stl.text_area('Enter the note')


def transform_text(text):
    lem = WordNetLemmatizer()
    text = text.lower()
    text = nltk.word_tokenize(text)

    a = []
    for i in text:
        if i.isalnum():
            a.append(i)

    text = a[:]
    a.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            a.append(i)

    text = a[:]
    a.clear()

    for i in text:
        a.append(lem.lemmatize(i))

    return " ".join(a)


if stl.button('Predict'):
    transform_note = transform_text(input_message)
    vector = cv.transform([transform_note])
    result = model.predict(vector)
    if result == 1:
        stl.header('Not Suicided')
    else:
        stl.header(' Suicided')
