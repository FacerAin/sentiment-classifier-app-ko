import streamlit as st
from transformers import pipeline


def display_result(label, score):
    if label == "positive":
        color = "Green"
    else:
        color = "Red"
    label_text = f'<p style="color:{color}; font-size: 20px;">{label}</p>'
    st.markdown(label_text, unsafe_allow_html=True)
    st.text(f"Score: {score}")


classifier = pipeline(
    "sentiment-analysis", model="sangrimlee/bert-base-multilingual-cased-nsmc"
)


st.title("Korean Sentiment Classifier")
sentence = st.text_input(label="Input Sentence")
run = st.button("Go!")

if run:
    st.write("Results")
    results = classifier(sentence)[0]
    label = results["label"]
    score = results["score"]
    display_result(label, score)
