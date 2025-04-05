import streamlit as st
from langchain_community.llms import Ollama

# Let's initialize LLama via Ollama
@st.cache_resource
def load_llm():
    return Ollama(model = "llama3:8b", temperature = 0.75)
llm = load_llm()

# Async response generation 
async def generate_response(prompt):
    return await llm.invoke(prompt)

# Initialzie session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "What can I help you with today?"}] 

if "questions" not in st.session_state:
    st.session_state.questions = [] # List to store user questions

# Display chat history
st.title("Local Llama Chat")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and processing
if prompt := st.chat_input("Your message"):
    # Save user question
    st.session_state.questions.append(prompt)
    # Save question to file for persistence
    with open("questions.txt", "a") as f:
        f.write(prompt + "\n")
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        response = llm.invoke(prompt)
        st.markdown(response)

    # Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": response}) 

if st.checkbox("Show saved questions"):
    st.subheader("User Questions")
    for question in st.session_state.questions:
        st.markdown(f" - {question}")