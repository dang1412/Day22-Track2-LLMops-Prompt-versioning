"""Step 2 — Prompt Hub & A/B Routing."""

import os
import hashlib
from collections import Counter

import config

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = config.LANGSMITH_API_KEY or ""
os.environ["LANGCHAIN_PROJECT"] = config.LANGSMITH_PROJECT
os.environ["LANGCHAIN_ENDPOINT"] = config.LANGSMITH_ENDPOINT

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable

import importlib
_step1 = importlib.import_module("01_langsmith_rag_pipeline")
build_vectorstore = _step1.build_vectorstore
make_llm = _step1.make_llm
from qa_pairs import SAMPLE_QUESTIONS

SYSTEM_V1 = (
    "You are a helpful AI assistant. "
    "Answer the user's question using ONLY the provided context. "
    "Keep your answer concise (2-4 sentences). "
    "If the context does not contain the answer, say: 'I don't have enough information.'\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human", "{question}"),
])

SYSTEM_V2 = (
    "You are an expert AI tutor. Provide a structured, accurate answer.\n\n"
    "Instructions:\n"
    "1. Read the context carefully.\n"
    "2. Identify the key facts relevant to the question.\n"
    "3. Write a clear, well-organized answer (3-5 sentences).\n"
    "4. State explicitly if the context lacks sufficient information.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human", "{question}"),
])

PROMPT_V1_NAME = "day22-rag-prompt-v1"
PROMPT_V2_NAME = "day22-rag-prompt-v2"

LOCAL_PROMPTS = {PROMPT_V1_NAME: PROMPT_V1, PROMPT_V2_NAME: PROMPT_V2}


def push_prompts_to_hub(client: Client) -> None:
    for name, obj, desc in [
        (PROMPT_V1_NAME, PROMPT_V1, "V1 - concise answers"),
        (PROMPT_V2_NAME, PROMPT_V2, "V2 - structured expert answers"),
    ]:
        try:
            url = client.push_prompt(name, object=obj, description=desc)
            print(f"OK Pushed {name} -> {url}")
        except Exception as e:
            print(f"!! push {name}: {e}")


def pull_prompts_from_hub(client: Client) -> dict:
    prompts = {}
    for name in (PROMPT_V1_NAME, PROMPT_V2_NAME):
        try:
            prompts[name] = client.pull_prompt(name)
            print(f"<- Pulled '{name}' from Hub")
        except Exception as e:
            prompts[name] = LOCAL_PROMPTS[name]
            print(f"ii Using local fallback for '{name}' ({e})")
    return prompts


def get_prompt_version(request_id: str) -> str:
    h = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if h % 2 == 0 else PROMPT_V2_NAME


@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
    return {"question": question, "answer": answer, "version": version}


def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)
    config.validate()

    client = Client(api_key=config.LANGSMITH_API_KEY)
    push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client)

    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = make_llm()

    counts = Counter()
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt = prompts[version_key]
        result = ask_ab(retriever, llm, prompt, question, version_tag)
        counts[version_tag] += 1
        print(f"[{i+1:02d}] [prompt-{version_tag}] {question[:55]}")
        print(f"      -> {result['answer'][:100]}")

    print(f"\nRouting summary: v1={counts['v1']}  v2={counts['v2']}  total={sum(counts.values())}")


if __name__ == "__main__":
    main()
