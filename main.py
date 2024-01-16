import os

from langchain.llms import OpenAI
#from openai import OpenAI
from constants import openai_key
#from langchain_community import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain , SequentialChain , SimpleSequentialChain
#from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory

import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key

st.title("Celebrity Search Results")
input_text=st.text_input("search the topic u want")

first_input_prompt=PromptTemplate(
    input_variables=['name'],
    template="tell me about celebrity {name}"
)

person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')

llm=OpenAI(temperature=0.8)
chain=LLMChain(
    llm=llm , prompt=first_input_prompt, verbose=True,output_key='person'
)

second_input_prompt=PromptTemplate(
    input_variables=['person'],
    template="when was {person} born ? "
)

chain2=LLMChain(
    llm=llm , prompt=second_input_prompt,verbose=True,output_key='dob'
)

third_input_prompt=PromptTemplate(
    input_variables=['dob'],
    template="mention 5 major events happend around {dob} in the world"
)

chain3=LLMChain(llm=llm, prompt=third_input_prompt , verbose=True ,output_key='description')

parent_chain=SequentialChain(
    chains=[chain,chain2,chain3],input_variables=['name'],output_variables=['person','dob','description'], verbose=True
)

if input_text:
    st.write(parent_chain({'name':input_text}))
