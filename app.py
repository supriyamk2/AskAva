import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryBufferMemory

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks import get_openai_callback
import os
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


def main():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    hide_streamlit_style = """
    <style>
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.image("logo.png",width=320)

    username = authenticate_user()

    if username:
        st.title("Optty: Powered by Advanced Legal AI.")
        st.write("Get help for your immigration questions from a legal expert")
        
        
        
        conversation_history = load_conversation_history(username)
        conversation_memory = ConversationBufferMemory(memory_key = "history")
        #conversation_memory = ConversationSummaryBufferMemory(llm=OpenAI(), max_token_limit=200)
        conversation_chain = ConversationChain(
            
            llm = OpenAI(),
            memory = conversation_memory
        )
        

        # Add API key input if needed
        #api_key = st.text_input("Enter your User Id:", type="password")
        #os.environ["OPENAI_API_KEY"] = api_key

        #if not api_key:
          #  st.warning("Please enter your User Id to continue.")
        #else:

        file_type = st.selectbox("Choose the file type", options=["Knowledge Base", "Your Document"])
        file = None
        text = None

        if file_type == "PDF":
            file = st.file_uploader("Upload your PDF", type="pdf")
            if file is not None:
                text = extract_text_from_pdf(file)
        elif file_type == "TXT":
            file = st.file_uploader("Upload your TXT", type="txt")
            if file is not None:
                text = extract_text_from_txt(file)
        elif file_type == "Knowledge Base":
            text = extract_text_from_journal()

        if file is not None or file_type == "Knowledge Base":
            # split into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
            )
            chunks = text_splitter.split_text(text)

            # create embeddings
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            col1, col2 = st.columns([3, 1])  # 3:1 ratio gives more space to the chat

            with col1:
                st.header(f"Hey {username}! How can I help?") 

                if 'current_session_history' not in st.session_state:
                    st.session_state['current_session_history'] = []

                #llm = OpenAI()
                #chain = load_qa_chain(llm, chain_type="stuff")
                chat_model = ChatGroq(
                    temperature=0,
                    model_name="gemma-7b-it",
                    api_key=groq_api_key
                )
                            
                def handle_user_question(user_question, username):
                    conversation_history = load_conversation_history(username)
                    history_str = "\n".join([f"{'User' if chat['is_user'] else 'Optty'}: {chat['message']}" for chat in conversation_history])

                    system_prompt = "You are a helpful immigration assistant who remembers name."
                    prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", system_prompt), 
                            ("human", history_str),
                            ("human", user_question)
                        ]
                    )
                    response = chat_model.invoke(prompt)
                    return response

                def generate_response(user_question, username):
                    docs = knowledge_base.similarity_search(user_question)
                    conversation_history = load_conversation_history(username)
                    history = [chat['message'] for chat in conversation_history]
                    
                    # Update the ConversationBufferMemory with the loaded history
                    conversation_memory.clear()
                    for message in history:
                        conversation_memory.save_context({"input": message}, {"output": ""})
                    
                    with get_openai_callback() as cb:
                        response = conversation_chain.predict(input=user_question)
                        print(cb)
                    
                    return response
                def is_fact_based_question(user_question):
                    fact_keywords = ["what", "how", "when", "define", "explain",'do you',"my name is"] 
                    if any(word in user_question.lower() for word in fact_keywords):
                        return True
                    else:
                        return False
                def route_question(user_question, username):
                    if is_fact_based_question(user_question):  # Implement your intent detection
                        return generate_response(user_question, username)
                    else:
                        return handle_user_question(user_question, username)

                user_question = st.text_input("Get help for your immigration questions from a legal expert", key="input")
                if user_question:
                    
                    response = route_question(user_question,username)
                    conversation_history.append({"message": user_question, "is_user": True})
                    conversation_history.append({"message": response, "is_user": False})
                    save_conversation_history(username, conversation_history)
                    #conversation_memory.save_context({"input": user_question}, {"output": response})
                    st.session_state['current_session_history'].append({"message": user_question, "is_user": True})
                    st.session_state['current_session_history'].append({"message": response, "is_user": False})
                    #st.markdown(f"**Ask Ava:** {response}") 
                

                for chat in st.session_state['current_session_history']:
                    if chat['is_user']:
                        st.markdown(f"**You:** {chat['message']}")
                    else:
                        st.markdown(f"**Optty:** {chat['message']}")

    else:
        st.warning("Invalid username or password.")

if __name__ == '__main__':
    main()
