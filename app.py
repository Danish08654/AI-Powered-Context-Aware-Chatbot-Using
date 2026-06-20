import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

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
# SESSION STATE
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
# LOAD LLM (FIXED)
# =========================
@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200
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

Use ONLY the context below to answer.

If the answer is not in the context, say:
"I don't know based on the provided information."

Context:
{context}

Question:
{question}

Answer in a short clear sentence:
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
# LOAD EVERYTHING
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

            answer = result["answer"]

            # Safe output handling
            if isinstance(answer, list):
                answer = answer[0].get("generated_text", str(answer))

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
