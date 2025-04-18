import os
import asyncio
import streamlit as st

from PyPDF2 import PdfReader
from langchain_community.llms import Ollama

# Let's initialize sidebar state in session state
if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = "expanded" # Default state

# Configure page with current sidebar state
st.set_page_config(
    page_title="Local Llama Chat",
    initial_sidebar_state=st.session_state.sidebar_state,
)

# Add this near the top of your app
st.markdown("""
<style>
    .stButton button {
        width: 100%;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin-bottom: 10px;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        padding-top: 0rem;
    }
</style>
""", unsafe_allow_html=True)

# Chat histories management
# @st.cache_resource
def load_llm():
    return Ollama(model = "llama3:8b", temperature = 0.75, num_thread = 4)
llm = load_llm()

# Initialzie session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "What can I help you with today?"}] 

if "questions" not in st.session_state:
    st.session_state.questions = [] # List to store user questions

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = None # Variable to store PDF text

# Display chat history
st.title("Local Llama Chat")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Place uploader above chat input
uploaded_files = st.file_uploader("Upload files for context", 
                                  type=["pdf", "txt", "docx", "jpg", "png"], 
                                  accept_multiple_files=True)

prompt = st.chat_input("Ask anything")

# Then adapt your processing logic to handle files from uploaded_files variable
if prompt:
        
        # Handle different types of input (text-only or text + files)
        if isinstance(prompt, dict) or hasattr(prompt, 'text'):
            user_text = prompt.text if hasattr(prompt, 'text') else ""
            uploaded_files = prompt.files if hasattr(prompt, 'files') else []

            # If no text but files, show a default message
            if not user_text and uploaded_files:
                user_text = "📎 Uploaded files"
   
            # Save question to list and file
            st.session_state.questions.append(user_text)
            with open("questions.txt", "a") as f:
                f.write(f'{user_text}\n')

            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": user_text})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_text)

                # Display file names if files were uploaded
                if uploaded_files:
                    file_list = '.'.join([f.name for f in uploaded_files])
                    st.markdown(f"Uploaded: {file_list}")
            # Process uploaded files
            context_text = ""
            for uploaded_file in uploaded_files:
                if uploaded_file.type == "application/pdf":
                    # Process PDF file
                    pdf_reader = PdfReader(uploaded_file)
                    pdf_text = ""
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            pdf_text += page_text
                    
                    # Store the PDF text 
                    st.session_state.pdf_text[uploaded_file.name] = pdf_text
                    context_text += pdf_text
                elif uploaded_file.type in ["text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                    # Process text files
                    try:
                        file_text = uploaded_file.read().decode("utf-8")
                        st.session_state
                        context_text += file_text
                    except Exception as e:
                        st.error(f"Error reading file {uploaded_file.name}: {e}")
            
            # Generate response with context from uploaded files or existing context 
            with st.chat_message("assistant"):
                if context_text:
                    context = context_text[:3000]
                    full_promomt = f'Context from uploaded document(s): {context} \n\n User question: {user_text} \nAnswer based only on the provided context:'
                elif st.session_state.pdf_text:
                    # Use existing context from previously uploaded files
                    combined_text = "\n".join(st.session_state.pdf_text.values())
                    context = combined_text[:3000]
                    full_promomt = f'Context from previously uploaded document(s): {context} \n\n User question: {user_text} \nAnswer based only on the provided context:'
                else: 
                    # No context available, just ask the question
                    full_prompt = user_text
                response = llm.invoke(full_prompt)
                st.markdown(response)

                # Store assistant response
                st.session_state.messages.append({"role": "assistant", "content": response})
        else: 
        # Legacy handling (text-only, no file)
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.questions.append(prompt)

            with st.chat_message("user"):
                st.markdown(prompt)
            # Generate response
            with st.chat_message("assistant"):
                if st.session_state.pdf_text:
                    # Use context from previously uploaded files
                    combined_text = "\n".join(st.session_state.pdf_text.values())
                    context = combined_text[:3000]
                    full_prompt = f'Context: {context}\n\nUser: {prompt}\nAssistant:'
                else:
                    full_prompt = prompt
                response = llm.invoke(full_prompt)
                st.markdown(response)

                # Store assistant response
                st.session_state.messages.append({"role": "assistant", "content": response})

# Show saved questions 
if st.checkbox("show saved questions"):
    st.subheader("User Questions")
    for question in st.session_state.questions:
        st.markdown(f' - {question}')