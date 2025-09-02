import torch
from smplx import SMPL
import os
import json
import trimesh
import numpy as np
import zlib
import joblib
import io
import requests


def download_segmentation_file(json_path="smpl_vert_segmentation.json"):
    """SMPL 用頂点セグメンテーションファイルをダウンロードまたは読み込み"""
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    url = (
        "https://raw.githubusercontent.com/Meshcapade/wiki/main/assets/"
        "SMPL_body_segmentation/smpl/smpl_vert_segmentation.json"
    )
    response = requests.get(url)
    response.raise_for_status()
    seg = response.json()
    with open(json_path, "w") as f:
        json.dump(seg, f)
    return seg


def color_part_on_mesh(mesh, part_name="head", color=[1.0, 0.0, 0.0], seg=None):
    """既存の trimesh.Trimesh メッシュに部位別色付けを適用"""
    if seg is None:
        seg = download_segmentation_file()
    vertex_colors = np.ones((len(mesh.vertices), 4), dtype=float) * [0.7, 0.7, 0.7, 1.0]
    if part_name in seg:
        idxs = [i for i in seg[part_name] if i < len(mesh.vertices)]
        vertex_colors[idxs] = color + [1.0]
    mesh.visual.vertex_colors = vertex_colors
    return mesh


def load_4dhumans_pkl(pkl_path):
    """zlib 圧縮 + joblib 形式の 4D-Humans の pkl を読み込む"""
    with open(pkl_path, "rb") as f:
        compressed = f.read()
    decompressed = zlib.decompress(compressed)
    bio = io.BytesIO(decompressed)
    return joblib.load(bio)


def process_4d_humans_pkl_smpl(
    pkl_path,
    smpl_model_path,
    frame_idx=0,
    person_idx=0,
    part_name="head",
    color=[1.0, 0.0, 0.0],
):
    """
    4D-Humans の pkl から SMPL メッシュを生成し、部位色付け
    """
    data = load_4dhumans_pkl(pkl_path)
    keys = list(data.keys())
    if not (0 <= frame_idx < len(keys)):
        raise IndexError(f"frame_idx={frame_idx} is out of range")
    frame = data[keys[frame_idx]]
    if isinstance(frame, list):
        params = frame[person_idx] if 0 <= person_idx < len(frame) else None
    elif isinstance(frame, dict):
        params = frame
    else:
        raise TypeError("Unexpected frame data type")
    if params is None:
        raise IndexError(f"person_idx={person_idx} is out of range for frame")

    smpl_model = SMPL(model_path=smpl_model_path)
    with torch.no_grad():
        out = smpl_model(
            global_orient=torch.tensor(
                params.get("global_orient", np.zeros(3)), dtype=torch.float32
            ).unsqueeze(0),
            body_pose=torch.tensor(
                params.get("body_pose", np.zeros(69)), dtype=torch.float32
            ).unsqueeze(0),
            betas=torch.tensor(
                params.get("betas", np.zeros(10)), dtype=torch.float32
            ).unsqueeze(0),
        )
    verts = out.vertices[0].cpu().numpy()
    mesh = trimesh.Trimesh(vertices=verts, faces=smpl_model.faces)
    return color_part_on_mesh(mesh, part_name, color)


def color_obj_smpl(obj_path, part_name="head", color=[1.0, 0.0, 0.0]):
    """既存の .obj ファイルを読み込み、指定部位を色付けして返す"""
    try:
        # OBJファイルの読み込み
        mesh = trimesh.load(obj_path, process=False)

        # Scene オブジェクトの場合は最初のメッシュを取得
        if not isinstance(mesh, trimesh.Trimesh):
            if hasattr(mesh, "geometry") and len(mesh.geometry) > 0:
                mesh = list(mesh.geometry.values())[0]
            else:
                raise ValueError(f"有効なメッシュが見つかりません: {obj_path}")

        print(f"メッシュ情報:")
        print(f"  頂点数: {len(mesh.vertices)}")
        print(f"  面数: {len(mesh.faces)}")
        print(f"  境界: {mesh.bounds}")

        # 部位別着色を適用
        colored_mesh = color_part_on_mesh(mesh, part_name, color)
        return colored_mesh

    except Exception as e:
        print(f"エラー: OBJファイルの読み込みに失敗しました: {e}")
        print(f"ファイルパス: {obj_path}")
        import traceback

        traceback.print_exc()
        return None


# 使用例:
# pkl を処理
# mesh1 = process_4d_humans_pkl_smpl(
#     pkl_path="demo.pkl",
#     smpl_model_path="SMPL_NEUTRAL.npz",
#     frame_idx=5,
#     person_idx=0,
#     part_name="leftHand",
#     color=[1.0, 0.0, 0.0],
# )
# mesh1.show()

# obj ファイルに色付け
mesh2 = color_obj_smpl(
    obj_path="tpose_mesh.obj",
    part_name="hips",
    color=[0.0, 1.0, 0.0],
)

if mesh2 is not None:
    try:
        # 表示の試行
        print("メッシュの可視化を試行中...")
        mesh2.show()
    except Exception as e:
        print(f"可視化エラー: {e}")
        print("代替案: ファイルに保存します...")

        # 代替案: ファイル保存
        try:
            output_path = "colored_mesh_output.obj"
            mesh2.export(output_path)
            print(f"メッシュを保存しました: {output_path}")

            # 画像として保存も試行
            try:
                from lib.obj_visualizer import quick_visualize

                image_path = "colored_mesh_output.png"
                quick_visualize(output_path, image_path)
                print(f"可視化画像を保存しました: {image_path}")
            except Exception as viz_e:
                print(f"画像保存エラー: {viz_e}")

        except Exception as save_e:
            print(f"ファイル保存エラー: {save_e}")
else:
    print("メッシュの作成に失敗しました")
