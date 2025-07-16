#!/usr/bin/env python3
"""
サンプルT-poseメッシュ作成

テスト用のサンプルT-poseメッシュを作成するスクリプトです。
"""

import numpy as np
import trimesh
from pathlib import Path


def create_sample_tpose_mesh(output_path: Path) -> Path:
    """
    テスト用のサンプルT-poseメッシュを作成
    
    Args:
        output_path: 出力ファイルパス
        
    Returns:
        保存されたファイルパス
    """
    print("サンプルT-poseメッシュを作成中...")
    
    # 簡単な人型メッシュの頂点を定義（6890頂点に合わせるため適当な点群を生成）
    np.random.seed(42)  # 再現性のため
    
    # 人型の大まかな形状を模擬
    vertices = []
    
    # 頭部 (0-588)
    head_vertices = np.random.normal([0, 1.6, 0], [0.1, 0.1, 0.1], (589, 3))
    vertices.extend(head_vertices)
    
    # 首 (589-717)
    neck_vertices = np.random.normal([0, 1.4, 0], [0.05, 0.05, 0.05], (129, 3))
    vertices.extend(neck_vertices)
    
    # 胴体上部 (718-1089)
    spine2_vertices = np.random.normal([0, 1.2, 0], [0.15, 0.1, 0.1], (372, 3))
    vertices.extend(spine2_vertices)
    
    # 胴体中部 (1090-1369)
    spine1_vertices = np.random.normal([0, 1.0, 0], [0.15, 0.1, 0.1], (280, 3))
    vertices.extend(spine1_vertices)
    
    # 胴体下部 (1370-1597)
    spine_vertices = np.random.normal([0, 0.8, 0], [0.15, 0.1, 0.1], (228, 3))
    vertices.extend(spine_vertices)
    
    # 左腕・肩 (1598-2007)
    left_arm_vertices = np.random.normal([-0.4, 1.2, 0], [0.05, 0.2, 0.05], (410, 3))
    vertices.extend(left_arm_vertices)
    
    # 右腕・肩 (2008-2417)
    right_arm_vertices = np.random.normal([0.4, 1.2, 0], [0.05, 0.2, 0.05], (410, 3))
    vertices.extend(right_arm_vertices)
    
    # 左手 (2418-2853)
    left_hand_vertices = np.random.normal([-0.6, 1.2, 0], [0.03, 0.05, 0.03], (436, 3))
    vertices.extend(left_hand_vertices)
    
    # 右手 (2854-3092)
    right_hand_vertices = np.random.normal([0.6, 1.2, 0], [0.03, 0.05, 0.03], (239, 3))
    vertices.extend(right_hand_vertices)
    
    # 左脚 (3093-3534)
    left_leg_vertices = np.random.normal([-0.1, 0.4, 0], [0.05, 0.3, 0.05], (442, 3))
    vertices.extend(left_leg_vertices)
    
    # 右脚 (3535-3976)
    right_leg_vertices = np.random.normal([0.1, 0.4, 0], [0.05, 0.3, 0.05], (442, 3))
    vertices.extend(right_leg_vertices)
    
    # 左足 (3977-4227)
    left_foot_vertices = np.random.normal([-0.1, 0.0, 0.1], [0.05, 0.02, 0.08], (251, 3))
    vertices.extend(left_foot_vertices)
    
    # 右足 (4228-4492)
    right_foot_vertices = np.random.normal([0.1, 0.0, 0.1], [0.05, 0.02, 0.08], (265, 3))
    vertices.extend(right_foot_vertices)
    
    # 残りの頂点を埋める (4493-6889)
    remaining_count = 6890 - len(vertices)
    remaining_vertices = np.random.normal([0, 1.0, 0], [0.2, 0.5, 0.1], (remaining_count, 3))
    vertices.extend(remaining_vertices)
    
    vertices = np.array(vertices)
    
    # 点群からメッシュを作成（各頂点を球として近似）
    # この方法で6890個の頂点を保持
    print(f"生成された頂点数: {len(vertices)}")
    
    # 簡単な三角面を生成（近隣点を接続）
    # 実際のSMPLモデルの面構造は複雑ですが、ここでは簡易版
    from scipy.spatial import Delaunay
    
    # 2D投影でDelaunay三角形分割を実行（XY平面）
    points_2d = vertices[:, :2]  # X, Y座標のみ使用
    tri = Delaunay(points_2d)
    faces = tri.simplices
    
    print(f"面数: {len(faces)}")
    
    # trimeshメッシュを作成
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # メッシュの修復とクリーニング
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    
    # メッシュとして保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)
    
    print(f"サンプルT-poseメッシュを保存: {output_path}")
    return output_path


def main():
    """メイン関数"""
    output_path = Path("sample_tpose.obj")
    create_sample_tpose_mesh(output_path)
    print("サンプルT-poseメッシュの作成が完了しました!")


if __name__ == "__main__":
    main()