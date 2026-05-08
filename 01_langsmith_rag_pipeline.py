"""Step 1 — LangSmith-instrumented RAG Pipeline."""

import os
from pathlib import Path

import config  # loads .env and sets LangSmith env vars

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = config.LANGSMITH_API_KEY or ""
os.environ["LANGCHAIN_PROJECT"] = config.LANGSMITH_PROJECT
os.environ["LANGCHAIN_ENDPOINT"] = config.LANGSMITH_ENDPOINT

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

from qa_pairs import SAMPLE_QUESTIONS

KB_PATH = Path(__file__).parent / "data" / "knowledge_base.txt"


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=config.DEFAULT_LLM_MODEL,
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
        temperature=0,
    )


def make_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
    )


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Use ONLY the context below to answer the user's question. "
     "If the answer is not in the context, say you don't have enough information.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


def build_vectorstore() -> FAISS:
    text = KB_PATH.read_text()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"Split knowledge base into {len(chunks)} chunks")
    return FAISS.from_texts(chunks, make_embeddings())


def build_rag_chain(vectorstore: FAISS):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | make_llm()
        | StrOutputParser()
    )
    return chain, retriever


@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    return chain.invoke(question)


def main():
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)
    config.validate()

    vectorstore = build_vectorstore()
    chain, _ = build_rag_chain(vectorstore)

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        answer = ask(chain, question)
        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question[:60]}")
        print(f"       A: {answer[:120]}\n")

    print(f"OK {len(SAMPLE_QUESTIONS)} traces sent to LangSmith project '{os.environ['LANGCHAIN_PROJECT']}'")
    print("   Open https://smith.langchain.com to view traces.")


if __name__ == "__main__":
    main()
