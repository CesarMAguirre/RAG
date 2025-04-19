import os
import asyncio
import streamlit as st
from typing import List, Dict, Any, Optional

from PyPDF2 import PdfReader
from langchain_community.llms import Ollama

# Configure page with current sidebar state
if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = "expanded"  # Default state

st.set_page_config(
    page_title="Local Llama Chat",
    initial_sidebar_state=st.session_state.sidebar_state,
)

# Styling
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

# LLM Configuration
@st.cache_resource
def load_llm():
    """Load the LLM with optimized settings"""
    return Ollama(
        model="llama3:8b", 
        temperature=0.75,
        num_thread=4,
        num_gpu=1,  
        num_predict=256  # Limit token generation for faster responses
    )

llm = load_llm()

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "What can I help you with today?"}]

if "questions" not in st.session_state:
    st.session_state.questions = []  # List to store user questions

if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = {}  # Dictionary to store PDF text by filename

if "vectorstores" not in st.session_state:
    st.session_state.vectorstores = {}  # For document retrieval

# Helper functions
def extract_pdf_text(pdf_file, max_pages: Optional[int] = None) -> str:
    """Extract text from PDF more efficiently with page limiting"""
    try:
        pdf_reader = PdfReader(pdf_file)
        pages_to_read = min(len(pdf_reader.pages), max_pages) if max_pages else len(pdf_reader.pages)
        
        pdf_text = ""
        for i in range(pages_to_read):
            page = pdf_reader.pages[i]
            page_text = page.extract_text()
            if page_text:
                pdf_text += f"\n--- Page {i+1} ---\n{page_text}"
        
        return pdf_text
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return ""

def process_file(uploaded_file) -> str:
    """Process various file types and return content"""
    if uploaded_file.type == "application/pdf":
        # Process first 10 pages for speed (adjust as needed)
        return extract_pdf_text(uploaded_file, max_pages=10)
    
    elif uploaded_file.type in ["text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        try:
            file_text = uploaded_file.read().decode("utf-8")
            return file_text
        except Exception as e:
            st.error(f"Error reading file {uploaded_file.name}: {e}")
            return ""
    return ""

async def generate_response(prompt: str) -> str:
    """Generate response asynchronously"""
    try:
        # Using ainvoke for better async performance if available
        if hasattr(llm, 'ainvoke'):
            return await llm.ainvoke(prompt)
        else:
            # Fallback to synchronous invoke
            return llm.invoke(prompt)
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "I'm having trouble processing your request. Please try again."

# UI Components
st.title("Cesar's Local Llama Chat")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# File uploader
with st.expander("Upload files for context", expanded=False):
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT or DOCX files", 
        type=["pdf", "txt", "docx"], 
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files:
        files_str = ", ".join([f.name for f in uploaded_files])
        st.success(f"Files ready: {files_str}")
        
        # Process button to manually trigger file processing
        if st.button("Process Files"):
            progress_bar = st.progress(0)
            for i, file in enumerate(uploaded_files):
                with st.spinner(f"Processing {file.name}..."):
                    file_text = process_file(file)
                    if file_text:
                        st.session_state.pdf_text[file.name] = file_text
                progress_bar.progress((i + 1) / len(uploaded_files))
            st.success("All files processed!")

# Chat input
prompt = st.chat_input("Ask anything")

if prompt:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.questions.append(prompt)
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process any newly uploaded files first
    if uploaded_files:  # Check if uploaded_files is not None
        new_files = [f for f in uploaded_files if f.name not in st.session_state.pdf_text]
        for file in new_files:
            file_text = process_file(file)
            if file_text:
                st.session_state.pdf_text[file.name] = file_text
    
    # Generate response with context from uploaded files
    with st.chat_message("assistant"):
        # Create a placeholder for streaming response
        message_placeholder = st.empty()
        
        # Determine the context to use
        context_text = ""
        if st.session_state.pdf_text:
            # Use the first 2000 chars from each document, up to 3 documents for speed
            doc_count = 0
            for filename, text in st.session_state.pdf_text.items():
                if doc_count >= 3:  # Limit to 3 documents for speed
                    break
                context_text += f"\n--- From {filename} ---\n{text[:2000]}"
                doc_count += 1
            
            full_prompt = f"""Context from documents:
{context_text.strip()[:4000]}

User question: {prompt}

Answer based only on the provided context. If the context doesn't contain relevant information, say so:"""
        else:
            full_prompt = prompt
        
        # Stream the response for better UX
        try:
            if hasattr(llm, 'stream'):
                full_response = ""
                # Display a spinner while waiting for the first chunk
                with st.spinner("Generating response..."):
                    try:
                        for chunk in llm.stream(full_prompt):
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)
                    except Exception as e:
                        st.error(f"Error streaming response: {e}")
                        full_response = "I encountered an error while generating a response. Please try again."
                        message_placeholder.markdown(full_response)
            else:
                # Fallback to async response
                with st.spinner("Generating response..."):
                    full_response = asyncio.run(generate_response(full_prompt))
                    message_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            full_response = "I encountered an error. Please check if Ollama is running properly."
            message_placeholder.markdown(full_response)
        
        # Store assistant response
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Save questions to file
        with open("questions.txt", "a") as f:
            f.write(f'{prompt}\n')

# Show saved questions in an expander
with st.expander("Saved Questions", expanded=False):
    if st.session_state.questions:
        for i, question in enumerate(st.session_state.questions):
            st.markdown(f'{i+1}. {question}')
    else:
        st.info("No questions saved yet.")

# Add settings in sidebar
with st.sidebar:
    st.header("Settings")
    
    # Model settings
    st.subheader("Model Configuration")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.75, 0.05)
    token_limit = st.slider("Response Token Limit", 100, 2048, 512, 50)
    
    # Apply settings
    if st.button("Apply Settings"):
        # Recreate the LLM with new settings
        llm = Ollama(
            model="llama3:8b", 
            temperature=temperature,
            num_thread=4,
            num_gpu=1,
            num_predict=token_limit
        )
        st.success("Settings applied!")
    
    # Clear conversation
    if st.button("Clear Conversation"):
        st.session_state.messages = [{"role": "assistant", "content": "What can I help you with today?"}]
        st.experimental_rerun()