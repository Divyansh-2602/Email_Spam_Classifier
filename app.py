# app.py
import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Text preprocessing
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load saved model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

# Streamlit UI
st.title("Email Spam Classifier")

input_mail = st.text_area("Enter the mail")

if st.button("Predict"):

    transformed_mail = transform_text(input_mail)
    vector_input = cv.transform([transformed_mail])


    result = model.predict(vector_input)[0]

    if result == 1:
        st.error("Spam Mail")
    else:
        st.success("Not Spam")
