import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from transformers import pipeline

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Support Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Support Chatbot")

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.chat-user {
    background-color: #DCF8C6;
    padding: 12px;
    border-radius: 12px;
    margin: 8px 0;
    text-align: right;
}

.chat-bot {
    background-color: #F1F0F0;
    padding: 12px;
    border-radius: 12px;
    margin: 8px 0;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION STATE INIT
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def clear_chat():
    st.session_state.chat_history = []

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Settings")

st.sidebar.write("Model: FLAN-T5 Base")
st.sidebar.write("Embeddings: MiniLM-L6-v2")
st.sidebar.write("Vector DB: FAISS")

if st.sidebar.button("🗑️ Clear Chat"):
    clear_chat()
    st.rerun()

# =========================
# LOAD VECTOR STORE
# =========================
@st.cache_resource
def load_vectorstore():
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        ".",
        embedding,
        index_name="index",
        allow_dangerous_deserialization=True
    )

    return vectorstore

# =========================
# LOAD LLM
# =========================
@st.cache_resource
def load_llm():
    pipe = pipeline(
        task="text-generation",
        model="google/flan-t5-base",
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.2
    )

    return HuggingFacePipeline(pipeline=pipe)
# =========================
# BUILD RAG CHAIN
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

    template = """
You are a helpful AI assistant.

Use ONLY the provided context.

If the answer is not available in the context,
say: "I don't know based on the provided information."

Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=_llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    return qa_chain

# =========================
# LOAD ALL RESOURCES
# =========================
try:
    vectorstore = load_vectorstore()
    llm = load_llm()
    qa_chain = load_qa_chain(vectorstore, llm)

except Exception as e:
    st.error(f"Startup Error: {e}")
    st.stop()

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
        st.markdown(
            f"<div class='chat-user'>{message}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='chat-bot'>{message}</div>",
            unsafe_allow_html=True
        )
