import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from transformers import pipeline

#  UI CONFIG 
st.set_page_config(page_title="AI Chatbot", layout="wide")
st.title(" AI Support Chatbot")

#  CUSTOM CSS 
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

#  SESSION STATE 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# CACHE VECTORSTORE 
@st.cache_resource
def load_vectorstore():
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local(
        "faiss_index",
        embedding,
        allow_dangerous_deserialization=True
    )
    return vectorstore

#  CACHE LLM 
@st.cache_resource
def load_llm():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        framework="pt",
        max_new_tokens=200,
        min_new_tokens=30,     
        do_sample=True,           
        temperature=0.7,         
        num_beams=4,              
        early_stopping=True,
        repetition_penalty=2.5   
    )
    return HuggingFacePipeline(pipeline=pipe)

# CACHE CHAIN 
@st.cache_resource
def load_qa_chain(_vectorstore, _llm):
    retriever = _vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1}
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    template = """You are an AI assistant. Answer ONLY the question asked.
     Do NOT include information about other topics.
     Use only the most relevant part of the context.
     Answer in 2-3 complete sentences.

    Context: {context}

    Question: {question}

    Answer only about '{question}':"""

    prompt = PromptTemplate(
    template=template,
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

#  LOAD RESOURCES 
vectorstore = load_vectorstore()
llm = load_llm()
qa_chain = load_qa_chain(vectorstore, llm)

#  SIDEBAR 
st.sidebar.title("⚙️ Settings")
st.sidebar.write("**Model:** FLAN-T5 base")
st.sidebar.write("**Embeddings:** MiniLM-L6-v2")
st.sidebar.write("**Vector Store:** FAISS")

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    qa_chain.memory.clear()
    st.rerun()

#  CHAT INPUT 
query = st.chat_input("Ask your question...")

if query:
    try:
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"question": query})
            answer = result.get("answer", "Sorry, I couldn't find an answer.")
    except Exception as e:
        answer = f" Error: {str(e)}"

    st.session_state.chat_history.append(("user", query))
    st.session_state.chat_history.append(("bot", answer))

#  DISPLAY CHAT 
for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(
            f"<div class='chat-user'> {message}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='chat-bot'> {message}</div>",
            unsafe_allow_html=True
        )
