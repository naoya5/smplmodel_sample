#!/usr/bin/env python3
"""
ヒートマップメッシュビューアー

生成されたヒートマップ付きT-poseメッシュを表示する
"""

import trimesh

# 灰色ベースのヒートマップメッシュを表示
mesh = trimesh.load("output/heatmap_gray/gaze_heatmap_tpose.obj")
print(f"メッシュ情報: {len(mesh.vertices)}頂点, {len(mesh.faces)}面")
print("3Dビューアーを起動中...")
mesh.show()