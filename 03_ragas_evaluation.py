"""Step 3 — RAGAS Evaluation."""

import os
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import config

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = config.LANGSMITH_API_KEY or ""
os.environ["LANGCHAIN_PROJECT"] = config.LANGSMITH_PROJECT
os.environ["LANGCHAIN_ENDPOINT"] = config.LANGSMITH_ENDPOINT

import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

import importlib
_step1 = importlib.import_module("01_langsmith_rag_pipeline")
build_vectorstore = _step1.build_vectorstore
make_llm = _step1.make_llm
make_embeddings = _step1.make_embeddings

_step2 = importlib.import_module("02_prompt_hub_ab_routing")
PROMPT_V1 = _step2.PROMPT_V1
PROMPT_V2 = _step2.PROMPT_V2

from qa_pairs import QA_PAIRS

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}

REPORT_PATH = Path(__file__).parent / "data" / "ragas_report.json"


@traceable(name="rag-eval-query", tags=["ragas", "step3"])
def run_rag(retriever, llm, prompt, question: str) -> dict:
    docs = retriever.invoke(question)
    contexts = [d.page_content for d in docs]
    ctx_str = "\n\n".join(contexts)
    answer = (prompt | llm | StrOutputParser()).invoke({"context": ctx_str, "question": question})
    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, version: str) -> list:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = make_llm()
    prompt = PROMPTS[version]

    results = []
    print(f"\nRunning {len(QA_PAIRS)} questions with prompt {version}...")
    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])
        results.append({
            "question": qa["question"],
            "reference": qa["reference"],
            "answer": out["answer"],
            "contexts": out["contexts"],
        })
        print(f"  [{i:02d}/{len(QA_PAIRS)}] {qa['question'][:60]}")
    return results


def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    return EvaluationDataset(samples=samples)


def run_ragas_eval(rag_results: list, version: str) -> dict:
    print(f"\nRunning RAGAS evaluation for prompt {version}...")
    dataset = build_ragas_dataset(rag_results)

    llm_eval = make_llm()
    emb_eval = make_embeddings()

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
    )

    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw = result[key]
        clean = [v for v in raw if v is not None and not (isinstance(v, float) and np.isnan(v))]
        scores[key] = float(np.mean(clean)) if clean else 0.0

    for k, v in scores.items():
        star = " *" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")
    return scores


def main():
    print("=" * 60)
    print("  Step 3: RAGAS Evaluation")
    print("=" * 60)
    config.validate()

    vectorstore = build_vectorstore()

    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    print("\n" + "=" * 60)
    print("  V1 vs V2 Comparison")
    print("=" * 60)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2 = v1_scores[metric], v2_scores[metric]
        winner = "<- V1" if s1 > s2 else ("<- V2" if s2 > s1 else "tie")
        print(f"  {metric:30s}: V1={s1:.4f}  V2={s2:.4f}  {winner}")

    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\nOK Target met: faithfulness = {best_faith:.4f}")
    else:
        print(f"\n!! Below target ({best_faith:.4f}). Try adjusting chunking or prompts.")

    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "best_faithfulness": best_faith,
        "target_met": best_faith >= 0.8,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Saved {REPORT_PATH}")


if __name__ == "__main__":
    main()
