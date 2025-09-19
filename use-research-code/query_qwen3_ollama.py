#!/usr/bin/env python3
"""
Query Ollama's qwen3:8b with a gaze transition analysis prompt.

This script sends the provided transitions data and analysis instructions
to a local Ollama server (default: http://localhost:11434) using the /api/chat endpoint.

Usage examples:
  - Use embedded sample data:
      python use-research-code/query_qwen3_ollama.py

  - Read data from a CSV file:
      python use-research-code/query_qwen3_ollama.py --file use-research-code/analysis_results/gaze_transitions.csv

  - Specify model and temperature:
      python use-research-code/query_qwen3_ollama.py --model qwen3:8b --temperature 0.4

Environment:
  - OLLAMA_HOST (optional): Override host, e.g. http://localhost:11434
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


DEFAULT_MODEL = "qwen3:8b"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


TRANSITIONS_CSV_TEXT = """transition_id,frame,time_seconds,from_part,to_part
1,78,2.6000,spine2,head
2,87,2.9000,head,neck
3,88,2.9333,neck,spine2
4,89,2.9667,spine2,nan
5,90,3.0000,nan,nan
6,91,3.0333,nan,nan
7,92,3.0667,nan,nan
8,93,3.1000,nan,hips
9,94,3.1333,hips,spine
10,95,3.1667,spine,spine1
11,99,3.3000,spine1,spine2
12,111,3.7000,spine2,head
13,117,3.9000,head,spine2
14,118,3.9333,spine2,head
15,119,3.9667,head,spine1
16,120,4.0000,spine1,spine
17,121,4.0333,spine,spine1
18,128,4.2667,spine1,leftUpLeg
19,129,4.3000,leftUpLeg,rightUpLeg
20,135,4.5000,rightUpLeg,leftUpLeg
21,138,4.6000,leftUpLeg,leftLeg
22,139,4.6333,leftLeg,leftUpLeg
23,147,4.9000,leftUpLeg,leftLeg
24,153,5.1000,leftLeg,leftUpLeg
25,154,5.1333,leftUpLeg,hips
26,155,5.1667,hips,spine2
27,201,6.7000,spine2,spine
28,205,6.8333,spine,hips
29,247,8.2333,hips,rightUpLeg
30,252,8.4000,rightUpLeg,rightHand
31,253,8.4333,rightHand,rightUpLeg
32,255,8.5000,rightUpLeg,rightHand
33,256,8.5333,rightHand,rightUpLeg
34,263,8.7667,rightUpLeg,hips
35,266,8.8667,hips,rightUpLeg
36,268,8.9333,rightUpLeg,hips
37,278,9.2667,hips,spine
38,279,9.3000,spine,spine2
39,280,9.3333,spine2,head
40,288,9.6000,head,nan
41,289,9.6333,nan,nan
42,290,9.6667,nan,nan
43,291,9.7000,nan,nan
44,292,9.7333,nan,spine2
45,331,11.0333,spine2,hips
46,332,11.0667,hips,rightUpLeg
47,333,11.1000,rightUpLeg,rightLeg
48,353,11.7667,rightLeg,rightUpLeg
49,361,12.0333,rightUpLeg,leftUpLeg
50,362,12.0667,leftUpLeg,hips
51,364,12.1333,hips,spine
52,365,12.1667,spine,spine1
53,366,12.2000,spine1,spine
54,368,12.2667,spine,spine1
55,369,12.3000,spine1,spine2
56,406,13.5333,spine2,hips
57,407,13.5667,hips,rightUpLeg
58,423,14.1000,rightUpLeg,leftLeg
59,426,14.2000,leftLeg,leftUpLeg
60,427,14.2333,leftUpLeg,leftLeg
61,428,14.2667,leftLeg,leftUpLeg
62,432,14.4000,leftUpLeg,leftLeg
63,439,14.6333,leftLeg,rightLeg
64,440,14.6667,rightLeg,rightFoot
65,446,14.8667,rightFoot,rightToeBase
66,461,15.3667,rightToeBase,rightFoot
67,462,15.4000,rightFoot,leftLeg
68,471,15.7000,leftLeg,rightUpLeg
69,472,15.7333,rightUpLeg,hips
70,473,15.7667,hips,spine
71,474,15.8000,spine,spine1
72,481,16.0333,spine1,spine2
73,496,16.5333,spine2,head
74,498,16.6000,head,neck
75,499,16.6333,neck,spine2
76,500,16.6667,spine2,head
77,503,16.7667,head,spine2
78,533,17.7667,spine2,neck
79,540,18.0000,neck,head
80,546,18.2000,head,spine2
81,547,18.2333,spine2,neck
82,548,18.2667,neck,head
83,549,18.3000,head,spine2
84,550,18.3333,spine2,nan
85,551,18.3667,nan,nan
86,552,18.4000,nan,leftToeBase
87,553,18.4333,leftToeBase,spine1
88,555,18.5000,spine1,spine2
89,574,19.1333,spine2,neck
90,575,19.1667,neck,head
91,577,19.2333,head,neck
92,580,19.3333,neck,head
93,583,19.4333,head,neck
94,587,19.5667,neck,spine2
95,588,19.6000,spine2,neck
96,589,19.6333,neck,spine2
97,596,19.8667,spine2,hips
98,598,19.9333,hips,nan
99,599,19.9667,nan,nan
100,600,20.0000,nan,nan
101,601,20.0333,nan,nan
102,602,20.0667,nan,leftLeg
103,603,20.1000,leftLeg,rightLeg
104,605,20.1667,rightLeg,rightUpLeg
105,606,20.2000,rightUpLeg,rightLeg
106,607,20.2333,rightLeg,leftUpLeg
107,612,20.4000,leftUpLeg,leftLeg
108,613,20.4333,leftLeg,rightLeg
109,625,20.8333,rightLeg,leftUpLeg
110,635,21.1667,leftUpLeg,rightUpLeg
111,640,21.3333,rightUpLeg,hips
112,641,21.3667,hips,spine1
113,647,21.5667,spine1,spine2
114,680,22.6667,spine2,head
115,682,22.7333,head,neck
116,683,22.7667,neck,head
117,689,22.9667,head,neck
118,694,23.1333,neck,spine2
119,695,23.1667,spine2,neck
120,697,23.2333,neck,spine2
121,704,23.4667,spine2,neck
122,705,23.5000,neck,spine2
123,711,23.7000,spine2,spine1
124,712,23.7333,spine1,hips
125,713,23.7667,hips,rightLeg
126,732,24.4000,rightLeg,leftFoot
127,733,24.4333,leftFoot,rightFoot
128,742,24.7333,rightFoot,leftFoot
129,743,24.7667,leftFoot,leftUpLeg
130,744,24.8000,leftUpLeg,hips
131,754,25.1333,hips,spine1
132,755,25.1667,spine1,head
133,764,25.4667,head,spine2
134,766,25.5333,spine2,head
135,769,25.6333,head,neck
136,779,25.9667,neck,head
137,780,26.0000,head,neck
138,781,26.0333,neck,spine2
139,785,26.1667,spine2,spine1
140,786,26.2000,spine1,spine
141,787,26.2333,spine,hips
142,788,26.2667,hips,spine
143,789,26.3000,spine,spine1
144,794,26.4667,spine1,spine2
145,815,27.1667,spine2,neck
146,816,27.2000,neck,spine2
147,817,27.2333,spine2,neck
148,818,27.2667,neck,spine2
149,833,27.7667,spine2,spine
150,835,27.8333,spine,hips
151,838,27.9333,hips,spine
152,842,28.0667,spine,spine1
153,844,28.1333,spine1,spine
154,845,28.1667,spine,spine1
155,856,28.5333,spine1,spine2
156,890,29.6667,spine2,neck
157,891,29.7000,neck,spine2
158,921,30.7000,spine2,spine1
159,922,30.7333,spine1,nan
160,923,30.7667,nan,leftToeBase
161,924,30.8000,leftToeBase,leftLeg
162,925,30.8333,leftLeg,leftForeArm
163,926,30.8667,leftForeArm,hips
164,927,30.9000,hips,spine1
165,937,31.2333,spine1,spine2
166,939,31.3000,spine2,spine1
167,943,31.4333,spine1,spine2
168,986,32.8667,spine2,head
169,997,33.2333,head,spine1
170,998,33.2667,spine1,spine2
"""


def build_messages(csv_text: str) -> List[Dict[str, str]]:
    system_prompt = (
        "あなたは視線行動研究の専門家です。日本語で簡潔・論理的に回答してください。"  # noqa: E501
        "出力は次の構成に厳密に従ってください: \n"
        "- 概要\n"
        "- 詳細分析\n"
        "- 行動解釈\n"
        "- 結論"
    )

    user_prompt = (
        "次の視線遷移データに基づき、以下の観点から分析してください。\n\n"
        "【データ（CSV）】\n"
        f"{csv_text.strip()}\n\n"
        "【分析タスク】\n"
        "1. 行動パターンの特徴\n"
        "2. 注意配分の傾向\n"
        "3. 認知的プロセスの推論\n"
        "4. 行動の意図や目的\n"
        "5. 専門的な知見（視線行動研究の観点）\n\n"
        "【出力フォーマット】\n"
        "- 概要: 全体的な視線行動の特徴\n"
        "- 詳細分析: 各指標の専門的解釈\n"
        "- 行動解釈: 認知的・心理的側面の考察\n"
        "- 結論: 主要な発見と示唆\n"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def chat_ollama(
    host: str, model: str, messages: List[Dict[str, str]], *, temperature: float
) -> str:
    url = f"{host.rstrip('/')}/api/chat"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": float(temperature),
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            resp_body = resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise SystemExit(f"HTTPError {e.code}: {body}")
    except urllib.error.URLError as e:
        raise SystemExit(f"URLError: {e}")

    try:
        parsed = json.loads(resp_body.decode("utf-8"))
    except json.JSONDecodeError:
        raise SystemExit("Failed to parse Ollama response as JSON.")

    # /api/chat returns { message: { content: ... }, done: true, ... }
    message = parsed.get("message", {})
    content = message.get("content")
    if not content:
        # Fallback for /api/generate-like responses or unexpected shapes
        content = parsed.get("response") or json.dumps(parsed, ensure_ascii=False)
    return content


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query Ollama qwen3:8b for gaze transition analysis"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Ollama model name (default: qwen3:8b)"
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Ollama host URL (default: env OLLAMA_HOST or http://localhost:11434)",
    )
    parser.add_argument(
        "--file",
        dest="file",
        default=None,
        help="Path to CSV file for transitions data",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature (default: 0.2)",
    )
    return parser.parse_args(argv)


def load_csv_text(file_path: Optional[str]) -> str:
    if not file_path:
        return TRANSITIONS_CSV_TEXT
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    csv_text = load_csv_text(args.file)
    messages = build_messages(csv_text)
    response = chat_ollama(
        args.host, args.model, messages, temperature=args.temperature
    )
    print(response)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
