from fastapi import FastAPI, Request
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
import os
from dotenv import load_dotenv
import streamlit as st
import json


def load_conversation_history(username):
    try:
        with open(f"conversation_history_{username}.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []
    
def save_conversation_history(username, conversation_history):
    with open(f"conversation_history_{username}.json", "w") as file:
        json.dump(conversation_history, file)


def extract_text_from_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def extract_text_from_txt(txt):
    text = txt.read().decode("utf-8")
    return text


def extract_text_from_journal():
    with open('reddit_data.txt', 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def authenticate_user():
    user_name = st.text_input("Enter your User Name:")
    password = st.text_input("Enter your password:", type="password")

    if user_name == "supriya" and password == "hello":
        return user_name
    else:
        return None

app = FastAPI() 
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")  # Make sure this is set

# Initialize these outside the endpoint for efficiency
conversation_memory = ConversationBufferMemory(memory_key="history")
conversation_chain = ConversationChain(llm=OpenAI(), memory=conversation_memory)

# Optionally preload your knowledge base 
knowledge_base = None
if os.path.exists("reddit_data.txt"):  # Adjust if needed for your knowledge base
    text = extract_text_from_journal()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

@app.post("/api/chat")
async def chatbot_endpoint(request: Request):
    user_input = await request.json()
    username = user_input.get("username") 
    user_question = user_input.get("question")

    def generate_response(user_question, username):
        conversation_history = load_conversation_history(username)
        history = [chat['message'] for chat in conversation_history]

        conversation_memory.clear()
        for message in history:
            conversation_memory.save_context({"input": message}, {"output": ""})

        docs = knowledge_base.similarity_search(user_question) if knowledge_base else []

        with get_openai_callback() as cb:
            response = conversation_chain.predict(input=user_question)
            print(cb)
        return response 

    response = generate_response(user_question, username)
    return {"response": response}
