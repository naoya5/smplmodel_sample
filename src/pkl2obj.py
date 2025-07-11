import torch
from smplx import SMPL  # SMPLX から SMPL クラスに変更
import json
import trimesh
import numpy as np
import zlib
import joblib
import io
import requests


def load_4dhumans_pkl(pkl_path):
    """zlib 圧縮 + joblib 形式の 4D-Humans の pkl を読み込む"""
    with open(pkl_path, "rb") as f:
        compressed = f.read()
    decompressed = zlib.decompress(compressed)
    bio = io.BytesIO(decompressed)
    data = joblib.load(bio)
    return data


def download_segmentation_file():
    """SMPL 用頂点セグメンテーションファイルをダウンロード"""
    url = (
        "https://raw.githubusercontent.com/Meshcapade/wiki/main/assets/"
        "SMPL_body_segmentation/smpl/smpl_vert_segmentation.json"
    )
    response = requests.get(url)
    with open("smpl_vert_segmentation.json", "w") as f:
        f.write(response.text)
    return json.loads(response.text)


def process_4d_humans_pkl_smpl(
    pkl_path,
    smpl_model_path,
    frame_idx=0,
    person_idx=0,
    part_name="head",
    color=[1.0, 0.0, 0.0],
):
    """
    SMPL 対応版: 4D-Humans pkl から指定フレーム・人物のメッシュを生成し、
    指定部位(part_name)を color で着色して返す
    """
    # 1) pkl 読み込み
    data = load_4dhumans_pkl(pkl_path)
    frame_keys = list(data.keys())
    # フレーム idx チェック
    if not (0 <= frame_idx < len(frame_keys)):
        raise IndexError(f"frame_idx={frame_idx} は範囲外 (0-{len(frame_keys) - 1})")
    frame_data = data[frame_keys[frame_idx]]

    # 2) 人物データ取得 (複数 or 単一)
    if isinstance(frame_data, list):
        if not (0 <= person_idx < len(frame_data)):
            raise IndexError(
                f"person_idx={person_idx} は範囲外 (0-{len(frame_data) - 1})"
            )
        params = frame_data[person_idx]
    elif isinstance(frame_data, dict):
        params = frame_data
    else:
        raise TypeError(f"予期しない frame_data の型: {type(frame_data)}")

    # 3) SMPL モデルでメッシュ生成 (SMPLX→SMPL)
    smpl_model = SMPL(model_path=smpl_model_path)
    with torch.no_grad():
        output = smpl_model(
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

    # 4) trimesh でメッシュ作成
    vertices = output.vertices[0].cpu().numpy()
    faces = smpl_model.faces
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # 5) 部位着色
    part_seg = download_segmentation_file()
    vertex_colors = np.ones((len(mesh.vertices), 4)) * [0.7, 0.7, 0.7, 1.0]
    if part_name in part_seg:
        # モデル頂点数に合わせて有効な頂点のみ抽出
        idxs = [i for i in part_seg[part_name] if i < len(mesh.vertices)]
        vertex_colors[idxs] = color + [1.0]
    mesh.visual.vertex_colors = vertex_colors

    return mesh


# 使用例:
mesh = process_4d_humans_pkl_smpl(
    pkl_path="/home/pattern/atomi/vlm/test/demo_gymnasts.pkl",
    smpl_model_path="/home/pattern/projects/4D-Humans/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
    frame_idx=5,
    person_idx=0,
    part_name="rightShoulder",
    color=[1.0, 0.0, 0.0],
)
mesh.show()  # trimesh ビューアで表示
