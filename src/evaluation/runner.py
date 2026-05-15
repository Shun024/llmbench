"""
LLMBench evaluation runner.
Runs all benchmarks across all models and saves results.
"""

import json
import time
import mlflow
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.models.clients import ModelClient, MODELS
from src.benchmarks.datasets import load_all_benchmarks
from src.evaluation.scorer import (
    score_factuality,
    score_reasoning,
    score_safety,
    score_instruction_following_rule_based,
    score_consistency,
)


def run_factuality(client: ModelClient, samples: list) -> dict:
    """Run factuality benchmark across all models."""
    results = {m: [] for m in MODELS}

    system = (
        "Answer the following multiple choice question. "
        "State the letter of your answer first, then briefly explain."
    )

    for i, sample in enumerate(samples):
        choices_text = "\n".join(
            f"{chr(65+j)}. {c}"
            for j, c in enumerate(sample["choices"])
        )
        prompt = f"Question: {sample['question']}\n\nChoices:\n{choices_text}"

        print(f"  Factuality [{i+1}/{len(samples)}]: {sample['question'][:50]}...")

        for model_id in MODELS:
            try:
                response = client.call(model_id, prompt, system, max_tokens=150)
                score = score_factuality(response, sample)
                results[model_id].append({
                    "sample_id": sample["id"],
                    "score": score["score"],
                    "latency_ms": response.latency_ms,
                    "cost_usd": response.cost_usd,
                    **score,
                })
            except Exception as e:
                print(f"    Error {model_id}: {e}")
                results[model_id].append({"sample_id": sample["id"], "score": 0.0})

    return results


def run_reasoning(client: ModelClient, samples: list) -> dict:
    """Run GSM8K reasoning benchmark."""
    results = {m: [] for m in MODELS}

    system = (
        "Solve the maths problem step by step. "
        "At the end, write your final answer after '####'."
    )

    for i, sample in enumerate(samples):
        print(f"  Reasoning [{i+1}/{len(samples)}]: {sample['question'][:50]}...")

        for model_id in MODELS:
            try:
                response = client.call(
                    model_id, sample["question"], system, max_tokens=300
                )
                score = score_reasoning(response, sample)
                results[model_id].append({
                    "sample_id": sample["id"],
                    "score": score["score"],
                    "latency_ms": response.latency_ms,
                    "cost_usd": response.cost_usd,
                    **score,
                })
            except Exception as e:
                print(f"    Error {model_id}: {e}")
                results[model_id].append({"sample_id": sample["id"], "score": 0.0})

    return results


def run_safety(client: ModelClient, samples: list) -> dict:
    """Run safety benchmark."""
    results = {m: [] for m in MODELS}
    system = "You are a helpful assistant."

    for i, sample in enumerate(samples):
        print(f"  Safety [{i+1}/{len(samples)}]: {sample['question'][:50]}...")

        for model_id in MODELS:
            try:
                response = client.call(
                    model_id, sample["question"], system, max_tokens=200
                )
                score = score_safety(response, sample)
                results[model_id].append({
                    "sample_id": sample["id"],
                    "score": score["score"],
                    "latency_ms": response.latency_ms,
                    "cost_usd": response.cost_usd,
                    **score,
                })
            except Exception as e:
                print(f"    Error {model_id}: {e}")
                results[model_id].append({"sample_id": sample["id"], "score": 0.0})

    return results


def run_instruction_following(client: ModelClient, samples: list) -> dict:
    """Run instruction following benchmark."""
    results = {m: [] for m in MODELS}
    system = "Follow the instructions precisely."

    for i, sample in enumerate(samples):
        print(f"  IF [{i+1}/{len(samples)}]: {sample['question'][:50]}...")

        for model_id in MODELS:
            try:
                response = client.call(
                    model_id, sample["question"], system, max_tokens=200
                )
                score = score_instruction_following_rule_based(response, sample)
                results[model_id].append({
                    "sample_id": sample["id"],
                    "score": score["score"],
                    "latency_ms": response.latency_ms,
                    "cost_usd": response.cost_usd,
                    **score,
                })
            except Exception as e:
                print(f"    Error {model_id}: {e}")
                results[model_id].append({"sample_id": sample["id"], "score": 0.0})

    return results


def run_consistency(client: ModelClient, samples: list) -> dict:
    """Run consistency benchmark."""
    results = {m: [] for m in MODELS}
    system = "Answer the question concisely."

    for i, sample in enumerate(samples):
        print(f"  Consistency [{i+1}/{len(samples)}]: {sample['question_a'][:50]}...")

        for model_id in MODELS:
            try:
                response_a = client.call(
                    model_id, sample["question_a"], system, max_tokens=100
                )
                response_b = client.call(
                    model_id, sample["question_b"], system, max_tokens=100
                )
                score = score_consistency(response_a, response_b, sample)
                results[model_id].append({
                    "sample_id": sample["id"],
                    "score": score["score"],
                    "latency_ms": (response_a.latency_ms + response_b.latency_ms) / 2,
                    "cost_usd": response_a.cost_usd + response_b.cost_usd,
                    **score,
                })
            except Exception as e:
                print(f"    Error {model_id}: {e}")
                results[model_id].append({"sample_id": sample["id"], "score": 0.0})

    return results


def aggregate_results(all_results: dict) -> pd.DataFrame:
    """Aggregate scores into a leaderboard DataFrame."""
    rows = []
    for dimension, model_results in all_results.items():
        for model_id, samples in model_results.items():
            if not samples:
                continue
            avg_score = sum(s.get("score", 0) for s in samples) / len(samples)
            avg_latency = sum(
                s.get("latency_ms", 0) for s in samples
            ) / len(samples)
            total_cost = sum(s.get("cost_usd", 0) for s in samples)

            rows.append({
                "model": model_id,
                "dimension": dimension,
                "score": round(avg_score, 4),
                "avg_latency_ms": round(avg_latency, 1),
                "total_cost_usd": round(total_cost, 6),
                "n_samples": len(samples),
            })

    return pd.DataFrame(rows)


def run_full_benchmark(
    n_factuality: int = 30,
    n_reasoning: int = 30,
) -> pd.DataFrame:
    """Run all benchmarks and return leaderboard."""
    mlflow.set_experiment("llmbench")

    print("=" * 60)
    print("LLMBench — Full Evaluation Suite")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Factuality: {n_factuality} samples | Reasoning: {n_reasoning} samples")
    print("=" * 60)

    client = ModelClient()
    benchmarks = load_all_benchmarks(
        n_factuality=n_factuality,
        n_reasoning=n_reasoning,
    )

    all_results = {}

    with mlflow.start_run(run_name=f"full_eval_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        print("\n[1/5] Factuality (TruthfulQA)...")
        all_results["factuality"] = run_factuality(
            client, benchmarks["factuality"]
        )

        print("\n[2/5] Reasoning (GSM8K)...")
        all_results["reasoning"] = run_reasoning(
            client, benchmarks["reasoning"]
        )

        print("\n[3/5] Safety...")
        all_results["safety"] = run_safety(
            client, benchmarks["safety"]
        )

        print("\n[4/5] Instruction Following...")
        all_results["instruction_following"] = run_instruction_following(
            client, benchmarks["instruction_following"]
        )

        print("\n[5/5] Consistency...")
        all_results["consistency"] = run_consistency(
            client, benchmarks["consistency"]
        )

        # Aggregate
        df = aggregate_results(all_results)

        # Overall score per model
        overall = df.groupby("model")["score"].mean().reset_index()
        overall.columns = ["model", "overall_score"]
        overall["overall_score"] = overall["overall_score"].round(4)

        print("\n" + "=" * 60)
        print("LEADERBOARD")
        print("=" * 60)
        pivot = df.pivot(index="model", columns="dimension", values="score")
        pivot["OVERALL"] = pivot.mean(axis=1).round(4)
        pivot = pivot.sort_values("OVERALL", ascending=False)
        print(pivot.to_string())

        # Log to MLflow
        for _, row in overall.iterrows():
            mlflow.log_metric(
                f"overall_{row['model'].replace('-', '_').replace('.', '_')}",
                row["overall_score"],
            )

        # Save results
        Path("data/results").mkdir(parents=True, exist_ok=True)
        df.to_parquet("data/results/benchmark_results.parquet", index=False)

        detailed_path = "data/results/detailed_results.json"
        with open(detailed_path, "w") as f:
            json.dump(
                {k: {m: v for m, v in v.items()} for k, v in all_results.items()},
                f, indent=2, default=str,
            )

        print(f"\nResults saved to data/results/")

    return df, all_results


if __name__ == "__main__":
    df, all_results = run_full_benchmark(
        n_factuality=30,
        n_reasoning=30,
    )