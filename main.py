from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.llms import Ollama
import streamlit as st
import os
load_dotenv()


key = os.getenv("GOOGLE_API_KEY")
llm = Ollama(model="llama2")
genai.configure(api_key = key)

model = genai.GenerativeModel('gemini-pro')


st.header("Do Any Task 	:panda_face:")
select_task = st.selectbox("Please Select Specific Task", ['Summarization', 'Named Entity Recognition', 'Language Transation', 'Question Answering'], index = None, placeholder="Select one Task")
select_model = st.selectbox("Please Select LLM Model", ['Google Gemini', 'Ollama Llama2 (local)'], index = None, placeholder="Select one Model")

if select_task == 'Summarization':
    word = st.slider("Word Length", 20, 100)
if select_task == 'Language Transation':
    transcribe_language = st.text_input("Enter the Language to transceibe")

input = st.text_area("Input")

submit = st.button("Submit")

if input:
    if submit:
        if select_model == 'Google Gemini':
            if select_task == 'Summarization':
                response = model.generate_content(f"Summarize this text{input} into {word} words")
            if select_task == 'Named Entity Recognition':
                response = model.generate_content(f"find named entity recognition from this text{input}")
            if select_task == 'Language Transation':
                response = model.generate_content(f"translate the text {input} into {transcribe_language}")
            if select_task == 'Question Answering':
                response = model.generate_content(input)
            
            output = st.text_area("Output", value=response.text)

        if select_model == 'Ollama Llama2 (local)':
            if select_task == 'Summarization':
                response = llm.invoke(f"Summarize this text{input} into {word} words")
            if select_task == 'Named Entity Recognition':
                response = llm.invoke(f"find named entity recognition from this text{input}")
            if select_task == 'Language Transation':
                response = llm.invoke(f"translate the text {input} into {transcribe_language}")
            if select_task == 'Question Answering':
                response = llm.invoke(input)

            output = st.text_area("Output", value=response)



