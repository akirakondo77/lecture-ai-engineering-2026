"""応答時間とトークン数を計測し、複数のモデル・質問で比較するスクリプト。

使い方:
    uv run python lectures/01/scratch/benchmark.py
"""

import json
import os
import sys
import time

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from google import genai
from google.genai import errors

from app.config import api_error_message, get_api_key

# ---------- 設定 ----------
MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
]

# 短い質問と長い質問
QUESTIONS = [
    # 短い質問
    {"label": "短_LoRA", "text": "LoRAとは何ですか？"},
    {"label": "短_トークン", "text": "LLMにおけるトークンとは何ですか？"},
    # 長い質問（より詳しい回答を求める）
    {
        "label": "長_RAG詳説",
        "text": "RAG（Retrieval-Augmented Generation）の仕組みを、具体的なアーキテクチャと処理の流れを含めて詳しく説明してください。また、RAGを使う場合と使わない場合のメリット・デメリットも比較してください。",
    },
    {
        "label": "長_Transformer",
        "text": "Transformerアーキテクチャの自己注意機構（Self-Attention）について、数式を交えて詳しく説明してください。Multi-Head Attentionの意義や、位置エンコーディングの役割も含めてください。",
    },
]

RUNS_PER_COMBO = 3  # 各組み合わせの繰り返し回数


def run_benchmark():
    """全組み合わせを計測して結果を返す。"""
    api_key = get_api_key()
    client = genai.Client(api_key=api_key)

    results = []

    for model in MODELS:
        for q in QUESTIONS:
            for run_idx in range(RUNS_PER_COMBO):
                print(
                    f"[{model}] {q['label']} (run {run_idx + 1}/{RUNS_PER_COMBO})...",
                    flush=True,
                )

                try:
                    start = time.perf_counter()
                    response = client.models.generate_content(
                        model=model, contents=q["text"]
                    )
                    elapsed = time.perf_counter() - start

                    usage = response.usage_metadata
                    result = {
                        "model": model,
                        "question_label": q["label"],
                        "run": run_idx + 1,
                        "elapsed_sec": round(elapsed, 3),
                        "prompt_tokens": (
                            usage.prompt_token_count if usage else None
                        ),
                        "output_tokens": (
                            usage.candidates_token_count if usage else None
                        ),
                        "total_tokens": (
                            usage.total_token_count if usage else None
                        ),
                        "error": None,
                    }

                except errors.APIError as e:
                    elapsed = time.perf_counter() - start
                    result = {
                        "model": model,
                        "question_label": q["label"],
                        "run": run_idx + 1,
                        "elapsed_sec": round(elapsed, 3),
                        "prompt_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,
                        "error": api_error_message(e.code, e.message),
                    }

                results.append(result)
                print(
                    f"  → {result['elapsed_sec']}s, "
                    f"in={result['prompt_tokens']}, "
                    f"out={result['output_tokens']}, "
                    f"err={result['error']}"
                )

                # レート制限を避けるために待つ（503対策で長めに）
                time.sleep(10)

        # モデル切替時に余裕を持たせる
        print(f"--- {model} 完了。次のモデルへ ---")
        time.sleep(15)

    return results


def save_results(results, path):
    """結果をJSONに保存。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n結果を {path} に保存しました。")


def print_summary(results):
    """計測結果のサマリーを表示。"""
    from collections import defaultdict

    # モデル × 質問ラベルごとに集計
    grouped = defaultdict(list)
    for r in results:
        if r["error"] is None:
            key = (r["model"], r["question_label"])
            grouped[key].append(r)

    print("\n" + "=" * 90)
    print(f"{'モデル':<25} {'質問':<15} {'平均応答(s)':<12} {'入力tok':<10} {'平均出力tok':<12} {'出力tok範囲':<15}")
    print("=" * 90)

    for (model, qlabel), runs in sorted(grouped.items()):
        avg_elapsed = sum(r["elapsed_sec"] for r in runs) / len(runs)
        avg_output = sum(r["output_tokens"] for r in runs) / len(runs)
        input_tok = runs[0]["prompt_tokens"]  # 入力は同じ質問なので同一のはず
        min_out = min(r["output_tokens"] for r in runs)
        max_out = max(r["output_tokens"] for r in runs)
        print(
            f"{model:<25} {qlabel:<15} {avg_elapsed:<12.3f} {input_tok:<10} {avg_output:<12.1f} {min_out}-{max_out}"
        )

    # エラーがあれば表示
    errors_list = [r for r in results if r["error"] is not None]
    if errors_list:
        print(f"\n⚠ エラーが {len(errors_list)} 件発生:")
        for e in errors_list:
            print(f"  [{e['model']}] {e['question_label']} run{e['run']}: {e['error']}")


if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    results = run_benchmark()
    save_results(results, output_path)
    print_summary(results)

