import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from transformers import pipeline

# =========================
# UI CONFIG
# =========================
st.set_page_config(page_title="AI Chatbot", layout="wide")
st.title("🤖 AI Support Chatbot")

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.chat-user {
    background-color: #DCF8C6;
    padding: 10px;
    border-radius: 10px;
    margin: 6px 0;
    text-align: right;
}
.chat-bot {
    background-color: #F1F0F0;
    padding: 10px;
    border-radius: 10px;
    margin: 6px 0;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = None

# =========================
# LOAD VECTORSTORE
# =========================
@st.cache_resource
def load_vectorstore():
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # IMPORTANT FIX:
    # index.faiss + index.pkl are in ROOT, so use "."
    vectorstore = FAISS.load_local(
        ".",
        embeddings=embedding,
        allow_dangerous_deserialization=True
    )

    return vectorstore

# =========================
# LOAD LLM
# =========================
@st.cache_resource
def load_llm():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        framework="pt",
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        num_beams=4,
        repetition_penalty=2.5
    )

    return HuggingFacePipeline(pipeline=pipe)

# =========================
# BUILD QA CHAIN
# =========================
@st.cache_resource
def load_qa_chain(_vectorstore, _llm):

    retriever = _vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    st.session_state.memory = memory

    prompt_template = """
You are a helpful AI assistant.

Use ONLY the given context to answer.
If you don't know, say you don't know.

Context:
{context}

Question:
{question}

Answer in 2-3 clear sentences.
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=_llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    return chain

# =========================
# LOAD EVERYTHING
# =========================
vectorstore = load_vectorstore()
llm = load_llm()
qa_chain = load_qa_chain(vectorstore, llm)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Settings")
st.sidebar.write("Model: FLAN-T5 Base")
st.sidebar.write("Embeddings: MiniLM-L6-v2")
st.sidebar.write("Vector DB: FAISS")

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    if st.session_state.memory:
        st.session_state.memory.clear()
    st.rerun()

# =========================
# CHAT INPUT
# =========================
query = st.chat_input("Ask your question...")

if query:
    try:
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"question": query})
            answer = result.get("answer", "Sorry, I couldn't find an answer.")

    except Exception as e:
        answer = f"Error: {str(e)}"

    st.session_state.chat_history.append(("user", query))
    st.session_state.chat_history.append(("bot", answer))

# =========================
# DISPLAY CHAT
# =========================
for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"<div class='chat-user'>{message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bot'>{message}</div>", unsafe_allow_html=True)
