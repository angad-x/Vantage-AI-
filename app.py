import streamlit as st
import os
import gc
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

FAISS_PATH = "faiss_index"

# 1. Page Config
st.set_page_config(page_title="Vantage Intelligence", page_icon="🛰️", layout="wide")

# 2. Professional "Slate & Indigo" CSS
st.markdown("""
    <style>
    /* Global Background & Font */
    .stApp {
        background-color: #09090b; /* Zinc 950 */
        color: #fafafa;
    }

    /* Sidebar - Clean & Minimal */
    section[data-testid="stSidebar"] {
        background-color: #121214;
        border-right: 1px solid #27272a;
    }

    /* Message Bubbles */
    [data-testid="stChatMessageUser"] {
        background-color: #27272a !important; 
        border: 1px solid #3f3f46;
        border-radius: 10px;
    }

    [data-testid="stChatMessageAssistant"] {
        background-color: #18181b !important;
        border: 1px solid #27272a;
        border-radius: 10px;
    }

    /* Chat Input Bar */
    div[data-testid="stChatInput"] {
        border-radius: 8px !important;
        border: 1px solid #3f3f46 !important;
        background-color: #18181b !important;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        background-color: #6366f1;
        color: white;
        border-radius: 6px;
        border: none;
        font-weight: 500;
    }
    
    .stButton>button:hover {
        background-color: #4f46e5;
        border: none;
        color: white;
    }

    /* File Uploader Customization */
    div[data-testid="stFileUploader"] section {
        background-color: #18181b;
        border: 1px dashed #3f3f46;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Vantage Header
st.markdown("""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 5px;">
        <div style="background-color: #6366f1; padding: 10px; border-radius: 8px;">
            <span style="font-size: 1.5rem;">🛰️</span>
        </div>
        <div style="line-height: 1;">
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -1.5px; color: #ffffff;">
                VANTAGE<span style="color: #6366f1;">.</span>
            </h1>
            <p style="margin: 0; font-size: 0.7rem; color: #71717a; text-transform: uppercase; letter-spacing: 2px; font-weight: 500;">
                Document Intelligence Layer
            </p>
        </div>
    </div>
    <hr style="border: 0; border-top: 1px solid #27272a; margin-bottom: 25px;">
""", unsafe_allow_html=True)

# --- CACHING LOGIC ---
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

@st.cache_resource
def load_rag_chain(_vector_db):
    llm = OllamaLLM(model="mistral", num_ctx=2048, temperature=0.1)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are Vantage, a professional intelligence assistant. Answer concisely based on the context.\n\nContext: {context}"),
        ("human", "{input}"),
    ])
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(_vector_db.as_retriever(search_kwargs={"k": 2}), combine_docs_chain)

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 📚 Library") 
    uploaded_file = st.file_uploader("Upload PDF Source", type="pdf", label_visibility="collapsed")
    
    st.divider()
    
    # Side-by-side technical buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Flush 🗑️", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("Re-index 🔄", use_container_width=True):
            if os.path.exists(FAISS_PATH):
                shutil.rmtree(FAISS_PATH)
            st.session_state.clear()
            st.rerun()

    # This creates a flexible spacer that pushes everything below it to the bottom
    st.html("""
        <div style="flex-grow: 1;"></div>
        <div style="padding-top: 370px;"> <hr style="border: 0; border-top: 1px solid #27272a; margin-bottom: 20px;">
            <div style="text-align: center; padding-bottom: 20px;">
                <p style="margin: 0; font-size: 0.7rem; color: #71717a; letter-spacing: 1px; text-transform: uppercase;">
                    Powered by
                </p>
                <p style="margin: 0; font-size: 0.85rem; font-weight: 700; color: #6366f1; border: 1px solid #6366f1; display: inline-block; padding: 2px 8px; border-radius: 4px; margin-top: 5px;">
                    MISTRAL-7B-V0.3
                </p>
                <p style="margin: 5px 0 0 0; font-size: 0.6rem; color: #3f3f46; letter-spacing: 0.5px;">
                    OLLAMA LOCAL ENGINE
                </p>
            </div>
        </div>
    """)

# --- MAIN LOGIC ---
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    if 'vector_db' not in st.session_state:
        embeddings = load_embeddings()
        if os.path.exists(FAISS_PATH):
            st.session_state.vector_db = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        else:
            with st.status("Initializing Vantage Index...", expanded=True) as status:
                loader = PyPDFLoader("temp.pdf")
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                final_docs = splitter.split_documents(docs)
                st.session_state.vector_db = FAISS.from_documents(final_docs, embeddings)
                st.session_state.vector_db.save_local(FAISS_PATH)
                status.update(label="Index Ready", state="complete", expanded=False)
                gc.collect()

    rag_chain = load_rag_chain(st.session_state.vector_db)

    # Display History
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🛰️"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Query the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🛰️"):
            try:
                with st.spinner("Analyzing..."):
                    response = rag_chain.invoke({"input": prompt})
                    answer = response["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error("Engine Offline: Please ensure Ollama is running.")
else:
    st.info("Awaiting source document upload to begin analysis.")