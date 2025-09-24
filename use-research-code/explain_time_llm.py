#!/usr/bin/env python3
"""
特定の時刻(秒)またはフレームを指定して、その時点の視線データをLLMに説明させるスクリプト。

入力: frame_gaze_analysis.csv（列: frame, max_part, max_value, total_gaze, <part>_value, <part>_ratio ...）

使用例:
  - 秒で指定:  uv run use-research-code/explain_time_llm.py --csv results/User11/frame_gaze_analysis.csv --time 16.77 --frame_rate 30
  - フレームで指定:  uv run use-research-code/explain_time_llm.py --csv results/User11/frame_gaze_analysis.csv --frame 503
  - モデル/温度指定:  uv run use-research-code/explain_time_llm.py --csv output/frame_gaze_analysis.csv --time 9.6 --model qwen3:8b --temperature 0.2

ローカルOllamaにPOSTする。環境変数 OLLAMA_HOST があれば使用（デフォルト http://localhost:11434）。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


DEFAULT_MODEL = "qwen3:8b"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "600"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="指定時刻(秒)またはフレームの視線データをLLMに説明させる",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--csv", required=True, help="frame_gaze_analysis.csv のパス")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--time", type=float, help="説明したい時刻(秒)")
    group.add_argument("--frame", type=int, help="説明したいフレーム番号")
    parser.add_argument(
        "--frame_rate", type=float, default=30.0, help="フレームレート(Hz)"
    )
    parser.add_argument("--topk", type=int, default=8, help="上位k部位を抽出して提示")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollamaモデル名")
    parser.add_argument("--host", default=DEFAULT_HOST, help="OllamaホストURL")
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="サンプリング温度",
    )
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT, help="HTTPタイムアウト秒"
    )
    return parser.parse_args()


def sec_to_nearest_frame(second: float, frame_rate: float) -> int:
    return int(round(second * frame_rate))


def load_row_for_frame(csv_path: str, target_frame: int) -> Optional[pd.Series]:
    df = pd.read_csv(csv_path)
    row = df[df["frame"] == target_frame]
    if row.empty:
        # 近傍フレームを許容（最小の差）
        df["_diff"] = (df["frame"] - target_frame).abs()
        nearest = df.sort_values("_diff").head(1)
        if nearest.empty:
            return None
        return nearest.iloc[0]
    return row.iloc[0]


def extract_top_parts(row: pd.Series, topk: int) -> List[Tuple[str, float, float]]:
    part_values: List[Tuple[str, float, float]] = []
    for col in row.index:
        if col.endswith("_value") and col not in ("max_value",):
            part = col[:-6]  # remove _value
            value = float(row[col])
            ratio_col = f"{part}_ratio"
            ratio = float(row.get(ratio_col, 0.0))
            part_values.append((part, value, ratio))

    part_values.sort(key=lambda x: x[1], reverse=True)
    return part_values[:topk]


def build_messages_for_frame(
    frame: int,
    time_sec: float,
    row: pd.Series,
    top_parts: List[Tuple[str, float, float]],
) -> List[Dict[str, str]]:
    system_prompt = (
        "あなたは視線行動研究の専門家です。日本語で簡潔・論理的に回答してください。"
        "面接官が面接中の面接者を見ている時の視線データです。"
        "spine2は上胸部, spine1は胸部, spineは腹部を表しています。"
    )

    lines = []
    lines.append(f"対象フレーム: {frame}  時刻: {time_sec:.3f}秒")
    lines.append(f"最大注視部位: {row['max_part']}  値: {float(row['max_value']):.6g}")
    lines.append(f"総視線量(total_gaze): {float(row['total_gaze']):.6g}")
    lines.append("上位部位(値, 割合):")
    for part, val, ratio in top_parts:
        lines.append(f"- {part}: {val:.6g} ({ratio:.3%})")

    user_prompt = (
        "/no_think 特定の時刻（フレーム）での部位注目確率からどこを見ていたか簡潔に言語化してください。\n\n"
        + "\n".join(lines)
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_ollama_chat(
    host: str,
    model: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float,
    timeout: float,
) -> str:
    import urllib.request
    import urllib.error

    url = f"{host.rstrip('/')}/api/chat"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": float(temperature)},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_body = resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise SystemExit(f"HTTPError {e.code}: {body}")
    except urllib.error.URLError as e:
        if isinstance(e.reason, TimeoutError):
            raise SystemExit(
                "TimeoutError: Ollamaサーバ応答がタイムアウト。 --timeout を増やしてください。"
            )
        raise SystemExit(f"URLError: {e}")

    try:
        parsed = json.loads(resp_body.decode("utf-8"))
    except json.JSONDecodeError:
        raise SystemExit("Failed to parse Ollama response as JSON.")

    message = parsed.get("message", {})
    content = message.get("content")
    if not content:
        content = parsed.get("response") or json.dumps(parsed, ensure_ascii=False)
    return content


def main() -> int:
    args = parse_args()

    if args.time is not None:
        target_frame = sec_to_nearest_frame(args.time, args.frame_rate)
        time_sec = target_frame / args.frame_rate
    else:
        target_frame = int(args.frame)
        time_sec = target_frame / args.frame_rate

    row = load_row_for_frame(args.csv, target_frame)
    if row is None:
        print("エラー: 指定に近いフレームが見つかりませんでした。")
        return 1

    top_parts = extract_top_parts(row, args.topk)
    messages = build_messages_for_frame(int(row["frame"]), time_sec, row, top_parts)

    print("--- 送信プロンプト(要約) ---")
    print(messages[1]["content"])  # userプロンプト
    print("----------------------------\n")

    response = call_ollama_chat(
        args.host,
        args.model,
        messages,
        temperature=args.temperature,
        timeout=args.timeout,
    )
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
