# チュートリアル

このチュートリアルでは、SMPL Model Sample ライブラリの基本的な使用方法を学びます。

## 目次

1. [環境設定](#環境設定)
2. [基本的な使い方](#基本的な使い方)
3. [4D-Humans PKL ファイルの処理](#4d-humans-pklファイルの処理)
4. [OBJ ファイルの処理](#objファイルの処理)
5. [カスタム着色](#カスタム着色)
6. [トラブルシューティング](#トラブルシューティング)

## 環境設定

### 必要なライブラリのインストール

```bash
pip install torch torchvision
pip install smplx
pip install trimesh
pip install numpy
pip install requests
pip install joblib
```

### SMPL モデルファイルの準備

SMPL モデルファイル（`.pkl`または`.npz`形式）を準備してください。
[SMPL 公式サイト](https://smpl.is.tue.mpg.de/)からダウンロードできます。

## 基本的な使い方

### 1. ライブラリのインポート

```python
import sys
sys.path.append('src')  # srcディレクトリをパスに追加

from pkl2obj import process_4d_humans_pkl_smpl, load_4dhumans_pkl
from obj2obj import color_obj_smpl, color_part_on_mesh
```

### 2. 簡単な例

```python
# OBJファイルに着色
mesh = color_obj_smpl(
    obj_path="your_model.obj",
    part_name="head",
    color=[1.0, 0.0, 0.0]  # 赤色
)
mesh.show()  # 3Dビューアで表示
```

## 4D-Humans PKL ファイルの処理

### PKL ファイルの内容確認

```python
# PKLファイルを読み込んで内容を確認
data = load_4dhumans_pkl("demo.pkl")

print("利用可能なフレーム数:", len(data))
print("フレームキー:", list(data.keys())[:5])  # 最初の5フレーム

# 最初のフレームの内容を確認
first_frame = data[list(data.keys())[0]]
if isinstance(first_frame, list):
    print("人物数:", len(first_frame))
    print("パラメータキー:", first_frame[0].keys())
else:
    print("パラメータキー:", first_frame.keys())
```

### メッシュ生成

```python
# 特定のフレームと人物からメッシュを生成
mesh = process_4d_humans_pkl_smpl(
    pkl_path="demo.pkl",
    smpl_model_path="SMPL_NEUTRAL.npz",
    frame_idx=10,      # 10番目のフレーム
    person_idx=0,      # 最初の人物
    part_name="head",  # 頭部を着色
    color=[1.0, 0.0, 0.0]  # 赤色
)

# メッシュを表示
mesh.show()

# OBJファイルとして保存
mesh.export("output.obj")
```

### 複数フレームの処理

```python
import os

# 出力ディレクトリを作成
os.makedirs("output", exist_ok=True)

# 複数フレームを連続処理
for frame_idx in range(0, 10):
    try:
        mesh = process_4d_humans_pkl_smpl(
            pkl_path="demo.pkl",
            smpl_model_path="SMPL_NEUTRAL.npz",
            frame_idx=frame_idx,
            person_idx=0,
            part_name="rightHand",
            color=[0.0, 1.0, 0.0]  # 緑色
        )
        mesh.export(f"output/frame_{frame_idx:03d}.obj")
        print(f"フレーム {frame_idx} 処理完了")
    except IndexError as e:
        print(f"フレーム {frame_idx} スキップ: {e}")
```

## OBJ ファイルの処理

### 既存 OBJ ファイルの着色

```python
# 既存のOBJファイルを読み込んで着色
mesh = color_obj_smpl(
    obj_path="input_model.obj",
    part_name="leftFoot",
    color=[0.0, 0.0, 1.0]  # 青色
)

# 結果を保存
mesh.export("colored_model.obj")
```

### 部位の組み合わせ着色

```python
import trimesh

# 元のメッシュを読み込み
mesh = trimesh.load("model.obj", process=False)

# 複数部位を異なる色で着色
from obj2obj import color_part_on_mesh

# 頭部を赤に
mesh = color_part_on_mesh(mesh, "head", [1.0, 0.0, 0.0])

# 手を緑に
mesh = color_part_on_mesh(mesh, "leftHand", [0.0, 1.0, 0.0])
mesh = color_part_on_mesh(mesh, "rightHand", [0.0, 1.0, 0.0])

# 足を青に
mesh = color_part_on_mesh(mesh, "leftFoot", [0.0, 0.0, 1.0])
mesh = color_part_on_mesh(mesh, "rightFoot", [0.0, 0.0, 1.0])

mesh.show()
```

## カスタム着色

### 利用可能な部位名の確認

```python
from obj2obj import download_segmentation_file

# セグメンテーション情報を取得
seg = download_segmentation_file()

# 利用可能な部位名を表示
print("利用可能な部位:")
for part_name in sorted(seg.keys()):
    vertex_count = len(seg[part_name])
    print(f"  {part_name}: {vertex_count}頂点")
```

### カスタム色の定義

```python
# カラーパレットの定義
COLORS = {
    "red": [1.0, 0.0, 0.0],
    "green": [0.0, 1.0, 0.0],
    "blue": [0.0, 0.0, 1.0],
    "yellow": [1.0, 1.0, 0.0],
    "cyan": [0.0, 1.0, 1.0],
    "magenta": [1.0, 0.0, 1.0],
    "orange": [1.0, 0.5, 0.0],
    "purple": [0.5, 0.0, 1.0],
}

# カスタム色で着色
mesh = process_4d_humans_pkl_smpl(
    pkl_path="demo.pkl",
    smpl_model_path="SMPL_NEUTRAL.npz",
    frame_idx=0,
    person_idx=0,
    part_name="head",
    color=COLORS["orange"]  # オレンジ色
)
```

### グラデーション効果

```python
import numpy as np

def create_gradient_color(base_color, intensity=1.0):
    """ベース色から強度を調整した色を生成"""
    return [c * intensity for c in base_color]

# グラデーション例
mesh = trimesh.load("model.obj", process=False)

# 異なる強度で同系色着色
mesh = color_part_on_mesh(mesh, "head", create_gradient_color([1.0, 0.0, 0.0], 1.0))
mesh = color_part_on_mesh(mesh, "neck", create_gradient_color([1.0, 0.0, 0.0], 0.7))
mesh = color_part_on_mesh(mesh, "leftShoulder", create_gradient_color([1.0, 0.0, 0.0], 0.5))
```

## トラブルシューティング

### よくあるエラーと解決方法

#### 1. `IndexError: frame_idx=X is out of range`

```python
# 解決方法: フレーム数を事前に確認
data = load_4dhumans_pkl("demo.pkl")
max_frames = len(data)
print(f"最大フレーム数: {max_frames}")

# 有効範囲内でアクセス
frame_idx = min(frame_idx, max_frames - 1)
```

#### 2. `FileNotFoundError: [Errno 2] No such file or directory`

```python
import os

# 解決方法: ファイル存在確認
if not os.path.exists("demo.pkl"):
    print("PKLファイルが見つかりません")

if not os.path.exists("SMPL_NEUTRAL.npz"):
    print("SMPLモデルファイルが見つかりません")
```

#### 3. メッシュが正しく表示されない

```python
# 解決方法: メッシュの基本情報を確認
mesh = trimesh.load("model.obj")
print(f"頂点数: {len(mesh.vertices)}")
print(f"面数: {len(mesh.faces)}")
print(f"バウンディングボックス: {mesh.bounds}")

# メッシュを修復
mesh.fix_normals()
mesh.remove_degenerate_faces()
```

### デバッグのヒント

```python
# パラメータの詳細確認
def debug_params(params):
    """SMPLパラメータの詳細を表示"""
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: shape={value.shape}, range=[{value.min():.3f}, {value.max():.3f}]")
        else:
            print(f"{key}: {type(value)} = {value}")

# 使用例
data = load_4dhumans_pkl("demo.pkl")
frame_data = data[list(data.keys())[0]]
if isinstance(frame_data, list):
    debug_params(frame_data[0])
else:
    debug_params(frame_data)
```

### パフォーマンス最適化

```python
# バッチ処理の例
def batch_process_frames(pkl_path, smpl_model_path, frame_range, output_dir):
    """複数フレームを効率的に処理"""
    os.makedirs(output_dir, exist_ok=True)

    # SMPLモデルを一度だけ読み込み
    from smplx import SMPL
    smpl_model = SMPL(model_path=smpl_model_path)

    # PKLデータを一度だけ読み込み
    data = load_4dhumans_pkl(pkl_path)

    for frame_idx in frame_range:
        try:
            # 個別処理ロジック
            # ...
            print(f"フレーム {frame_idx} 完了")
        except Exception as e:
            print(f"フレーム {frame_idx} エラー: {e}")

# 使用例
batch_process_frames("demo.pkl", "SMPL_NEUTRAL.npz", range(0, 50), "output")
```

このチュートリアルを参考に、あなたのプロジェクトに適した処理を実装してください。
