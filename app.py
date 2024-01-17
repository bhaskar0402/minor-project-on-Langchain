import os
from langchain.llms import OpenAI
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

def get_openai_response(question):
    llm=OpenAI(temperature=0.8)
    response=llm(question)
    return response

st.set_page_config(page_title="Q&A demo")
st.header("langchain Application")
input=st.text_input("input:",key="input")
response=get_openai_response(input)

submit=st.button("ask the question")


if submit:
    st.subheader("the response is")
    st.write(response)
