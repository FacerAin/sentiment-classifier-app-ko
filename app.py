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

with st.form(key="form"):
    sentence = st.text_input(label="Input Sentence", placeholder="나는 오늘 기분이 좋다.")
    submit = st.form_submit_button("Go!")

if submit:
    st.write("Results")
    with st.spinner("두뇌 풀가동!"):
        results = classifier(sentence)[0]
    label = results["label"]
    score = results["score"]
    display_result(label, score)
