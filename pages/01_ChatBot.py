from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
import streamlit as st
import sqlite3
from langchain.memory import ConversationSummaryBufferMemory

from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

st.set_page_config(
    page_title="ChatBot",
    page_icon="📃",
)

# 기본 모델 -> gpt-3.5-turbo
llm = ChatOpenAI(
    temperature=0.1,
)

summaryBufferMemory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=5,
    return_messages=True,
)

def add_summary_to_memory(input, output):
    summaryBufferMemory.save_context({"input": input}, {"output": output})


def get_summary():
    return summaryBufferMemory.load_memory_variables({})

# SQLite 데이터베이스에 연결 (데이터베이스가 없으면 새로 생성)
conn = sqlite3.connect("sqlite3.db")
cursor = conn.cursor()

# 테이블 생성
cursor.execute("""
CREATE TABLE IF NOT EXISTS interactions (
    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT, 
    uuid TEXT,
    user_input TEXT,
    bot_response TEXT,
    summary TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# 데이터 삽입 함수
def insert_interaction(uuid, user_input, bot_response, summury):
    # 매개변수로 받은 id를 그대로 사용하여 데이터베이스에 저장
    cursor.execute("""
    INSERT INTO interactions (uuid, user_input, bot_response, summary) 
    VALUES (?, ?, ?, ?)
    """, (uuid, user_input, bot_response, summury))
    conn.commit()
    print("Data saved successfully.")

@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


st.title("ChatBot")

st.markdown(
    """
Welcome!
"""
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
        )
        response = chain.invoke(message)
        add_summary_to_memory(message, response.content)
        uuid = "123e4567-e89b-12d3-a456-426614174000"  # 샘플 UUID 값
        insert_interaction(uuid, message, response.content, str(get_summary()))
        send_message(response.content, "ai")

else:
    st.session_state["messages"] = []