import os
from typing import List
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


DATA_DIR = "data"
CHROMA_DIR = "chroma_db"

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def load_documents() -> List[Document]:
    documents = []

    os.makedirs(DATA_DIR, exist_ok=True)

    for filename in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, filename)

        if filename.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            docs = loader.load()
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs = loader.load()
        else:
            continue

        for doc in docs:
            doc.metadata["source"] = filename

            if "confidential" in filename.lower():
                doc.metadata["roles"] = "hr,admin,it"
            elif "it" in filename.lower():
                doc.metadata["roles"] = "employee,it,admin"
            else:
                doc.metadata["roles"] = "employee,hr,admin"

        documents.extend(docs)

    return documents


def build_vector_db():
    documents = load_documents()

    if not documents:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    return vector_db


def get_vector_db():
    if not os.path.exists(CHROMA_DIR):
        return build_vector_db()

    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )


def role_allowed(doc: Document, role: str) -> bool:
    allowed_roles = doc.metadata.get("roles", "employee,hr,admin,it")
    allowed_roles = [r.strip() for r in allowed_roles.split(",")]
    return role in allowed_roles


def answer_policy_question(question: str, role: str = "employee", chat_history=None) -> str:
    vector_db = get_vector_db()

    if vector_db is None:
        return "No policy documents found. Please add documents inside the data folder."

    docs = vector_db.similarity_search(question, k=8)
    allowed_docs = [doc for doc in docs if role_allowed(doc, role)]
    top_docs = allowed_docs[:3]

    if not top_docs:
        return "I could not find policy information available for your role."

    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source')}\nContent: {doc.page_content}"
            for doc in top_docs
        ]
    )

    history_text = ""
    if chat_history:
        history_text = "\n".join(
            [f"{m['role']}: {m['content']}" for m in chat_history[-6:]]
        )

    prompt = ChatPromptTemplate.from_template("""
You are an enterprise HR and IT policy assistant.

Use only the provided context.
Answer conversationally and clearly.
Include source file names.

User role:
{role}

Recent conversation:
{history}

Context:
{context}

Question:
{question}
""")

    chain = prompt | llm

    result = chain.invoke({
        "role": role,
        "history": history_text,
        "context": context,
        "question": question
    })

    return result.content