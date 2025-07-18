"""
dedup_smpl_parts.py
CSV ＆ smpl_vert_segmentation.json から
  • 各部位の重複数
  • 重複を取り除いたユニーク頂点数
を出力します。
pip install pandas tabulate
"""

import json, pandas as pd, itertools
from pathlib import Path
from tabulate import tabulate

# ① あなたの CSV を読み込む ---------------------------------------------------
df = pd.read_csv("lib/vertex_counts.csv")  # part_name,vertex_count,percentage
keys = df.part_name.tolist()

# ② セグメンテーション JSON を読む -------------------------------------------
segm = json.loads(Path("smpl_vert_segmentation.json").read_text())
sets = {k: set(segm[k]) for k in keys}  # part → set(verts)

# ③ ペアごとの重複数を調べてみる --------------------------------------------
overlaps = []
for a, b in itertools.combinations(keys, 2):
    inter = sets[a] & sets[b]
    if inter:
        overlaps.append((a, b, len(inter)))
print("\n=== overlap (>0) 上位20 ===")
print(tabulate(sorted(overlaps, key=lambda x: -x[2])[:20], headers=["A", "B", "#dup"]))

# ④ 重複を除いた「排他的カウント」を計算 --------------------------------------
seen = set()
rows = []
for name, verts in sorted(sets.items(), key=lambda x: -len(x[1])):
    exclusive = len(verts - seen)  # まだ使われていない頂点だけ数える
    rows.append((name, exclusive))
    seen |= verts

print("\n=== exclusive counts ===")
print(tabulate(rows, headers=["part", "#unique"]))
print(f"\nTOTAL unique = {len(seen)} (should be 6 890)")
