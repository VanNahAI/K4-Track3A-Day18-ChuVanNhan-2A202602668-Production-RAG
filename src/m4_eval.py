from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass
from math import isfinite

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
        raise ValueError("Evaluation inputs must have equal lengths")
    if any(not isinstance(value, str) for values in (questions, answers, ground_truths) for value in values):
        raise ValueError("Questions, answers and ground truths must contain strings")
    if any(not isinstance(group, list) or any(not isinstance(c, str) for c in group) for group in contexts):
        raise ValueError("contexts must contain lists of strings")
    empty = {**dict.fromkeys(METRICS, 0.0), "per_question": []}
    if not questions:
        return empty
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        dataset = Dataset.from_dict({"question": questions, "answer": answers,
                                     "contexts": contexts, "ground_truth": ground_truths})
        from langchain_openai import ChatOpenAI
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        if not OPENAI_API_KEY:
            raise ValueError("Configure OLLAMA_API_KEY before evaluation")
        from ragas.run_config import RunConfig

        run_config = RunConfig(timeout=300, max_workers=2, max_retries=1)
        llm = LangchainLLMWrapper(ChatOpenAI(
            model=LLM_MODEL, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL,
            temperature=0, timeout=300, max_retries=0, max_tokens=4096,
            model_kwargs={"reasoning_effort": "low"} if LLM_MODEL.startswith("gpt-oss") else {}))
        embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"))
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                           context_precision, context_recall],
                          llm=llm, embeddings=embeddings, run_config=run_config,
                          raise_exceptions=False)
        df = result.to_pandas()
        if len(df) != len(questions):
            raise ValueError("RAGAS returned an unexpected number of rows")
        per_question = []
        valid_scores = {name: [] for name in METRICS}
        failed_questions = []
        for _, row in df.iterrows():
            scores = {name: float(row[name]) for name in METRICS}
            invalid = [name for name, score in scores.items() if not isfinite(score) or not 0 <= score <= 1]
            for name, score in scores.items():
                if name not in invalid:
                    valid_scores[name].append(score)
            if invalid:
                failed_questions.append({"question": row["question"], "metrics": invalid})
                continue
            per_question.append(EvalResult(question=row["question"], answer=row["answer"],
                                contexts=list(row["contexts"]), ground_truth=row["ground_truth"], **scores))
        aggregate = {name: sum(values) / len(values) if values else 0.0
                     for name, values in valid_scores.items()}
        output = {**aggregate, "per_question": per_question,
                  "metric_counts": {name: len(values) for name, values in valid_scores.items()},
                  "failed_questions": failed_questions}
        if failed_questions:
            output["error"] = f"Incomplete evaluation: {len(failed_questions)}/{len(questions)} questions"
        return output
    except Exception as exc:
        print(f"RAGAS evaluation failed ({type(exc).__name__})")
        return {**empty, "error": f"RAGAS evaluation failed ({type(exc).__name__})"}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if isinstance(bottom_n, bool) or not isinstance(bottom_n, int) or bottom_n < 0:
        raise ValueError("bottom_n must be a non-negative integer")
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten system prompt, set temperature to 0"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25 keywords"),
        "context_precision": ("Too many irrelevant chunks", "Add cross-encoder reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template for direct answers"),
    }
    failures = []
    for result in eval_results:
        scores = {name: float(getattr(result, name)) for name in METRICS}
        if any(not isfinite(score) or not 0 <= score <= 1 for score in scores.values()):
            raise ValueError("Evaluation scores must be finite and between 0 and 1")
        worst_metric = min(scores, key=scores.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        failures.append({"question": result.question, "worst_metric": worst_metric,
                         "score": sum(scores.values()) / len(METRICS),
                         "diagnosis": diagnosis, "suggested_fix": suggested_fix})
    return sorted(failures, key=lambda failure: failure["score"])[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
