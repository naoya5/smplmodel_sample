# API リファレンス

## pkl2obj.py

### 関数一覧

#### `load_4dhumans_pkl(pkl_path)`

4D-Humans 形式の PKL ファイルを読み込みます。

**パラメータ:**

- `pkl_path` (str): PKL ファイルのパス

**戻り値:**

- `dict`: 解凍されたデータ辞書

**説明:**
zlib 圧縮 + joblib 形式でシリアライズされた 4D-Humans の PKL ファイルを読み込み、Python の辞書として返します。

**例:**

```python
data = load_4dhumans_pkl("demo.pkl")
print(list(data.keys()))  # フレーム一覧
```

---

#### `download_segmentation_file()`

SMPL 頂点セグメンテーションファイルをダウンロードします。

**戻り値:**

- `dict`: 部位名と頂点インデックスのマッピング

**説明:**
Meshcapade の GitHub リポジトリから SMPL 用の頂点セグメンテーション情報をダウンロードし、ローカルに保存します。

---

#### `process_4d_humans_pkl_smpl(pkl_path, smpl_model_path, frame_idx=0, person_idx=0, part_name="head", color=[1.0, 0.0, 0.0])`

4D-Humans の PKL ファイルから SMPL メッシュを生成し、指定部位を着色します。

**パラメータ:**

- `pkl_path` (str): 4D-Humans の PKL ファイルパス
- `smpl_model_path` (str): SMPL モデルファイルパス
- `frame_idx` (int, optional): フレームインデックス（デフォルト: 0）
- `person_idx` (int, optional): 人物インデックス（デフォルト: 0）
- `part_name` (str, optional): 着色する部位名（デフォルト: "head"）
- `color` (list, optional): RGB 色値（デフォルト: [1.0, 0.0, 0.0]）

**戻り値:**

- `trimesh.Trimesh`: 着色されたメッシュオブジェクト

**例外:**

- `IndexError`: フレーム or 人物インデックスが範囲外
- `TypeError`: 予期しないデータ型

**例:**

```python
mesh = process_4d_humans_pkl_smpl(
    pkl_path="demo.pkl",
    smpl_model_path="SMPL_NEUTRAL.npz",
    frame_idx=5,
    person_idx=0,
    part_name="rightShoulder",
    color=[1.0, 0.0, 0.0]
)
mesh.show()
```

---

## obj2obj.py

### 関数一覧

#### `download_segmentation_file(json_path="smpl_vert_segmentation.json")`

SMPL 頂点セグメンテーションファイルをダウンロードまたは読み込みます。

**パラメータ:**

- `json_path` (str, optional): 保存先ファイルパス（デフォルト: "smpl_vert_segmentation.json"）

**戻り値:**

- `dict`: 部位名と頂点インデックスのマッピング

**説明:**
既存ファイルがあれば読み込み、なければダウンロードして保存します。

---

#### `color_part_on_mesh(mesh, part_name="head", color=[1.0, 0.0, 0.0], seg=None)`

既存の trimesh メッシュに部位別着色を適用します。

**パラメータ:**

- `mesh` (trimesh.Trimesh): 着色対象メッシュ
- `part_name` (str, optional): 着色部位名（デフォルト: "head"）
- `color` (list, optional): RGB 色値（デフォルト: [1.0, 0.0, 0.0]）
- `seg` (dict, optional): セグメンテーション辞書（未指定時は自動ダウンロード）

**戻り値:**

- `trimesh.Trimesh`: 着色されたメッシュ

**例:**

```python
mesh = trimesh.load("model.obj")
colored_mesh = color_part_on_mesh(mesh, "leftHand", [0.0, 1.0, 0.0])
```

---

#### `load_4dhumans_pkl(pkl_path)`

zlib 圧縮された joblib 形式の 4D-Humans PKL ファイルを読み込みます。

**パラメータ:**

- `pkl_path` (str): PKL ファイルのパス

**戻り値:**

- `dict`: 解凍されたデータ

---

#### `process_4d_humans_pkl_smpl(pkl_path, smpl_model_path, frame_idx=0, person_idx=0, part_name="head", color=[1.0, 0.0, 0.0])`

4D-Humans の PKL から SMPL メッシュを生成し、部位着色を行います。

**パラメータ:**

- `pkl_path` (str): PKL ファイルパス
- `smpl_model_path` (str): SMPL モデルパス
- `frame_idx` (int, optional): フレーム番号（デフォルト: 0）
- `person_idx` (int, optional): 人物番号（デフォルト: 0）
- `part_name` (str, optional): 着色部位（デフォルト: "head"）
- `color` (list, optional): RGB 色値（デフォルト: [1.0, 0.0, 0.0]）

**戻り値:**

- `trimesh.Trimesh`: 着色済みメッシュ

---

#### `color_obj_smpl(obj_path, part_name="head", color=[1.0, 0.0, 0.0])`

既存の OBJ ファイルを読み込み、指定部位を着色します。

**パラメータ:**

- `obj_path` (str): OBJ ファイルのパス
- `part_name` (str, optional): 着色部位名（デフォルト: "head"）
- `color` (list, optional): RGB 色値（デフォルト: [1.0, 0.0, 0.0]）

**戻り値:**

- `trimesh.Trimesh`: 着色されたメッシュ

**例:**

```python
mesh = color_obj_smpl("model.obj", "neck", [0.0, 1.0, 0.0])
mesh.show()
```

## サポートされる部位名

セグメンテーションファイルで定義された部位名：

- `head` - 頭部
- `neck` - 首
- `leftShoulder` / `rightShoulder` - 肩
- `leftHand` / `rightHand` - 手
- `leftFoot` / `rightFoot` - 足
- その他多数の解剖学的部位

## 色値の指定

色値は 0.0〜1.0 の範囲で RGB 形式で指定：

- `[1.0, 0.0, 0.0]` - 赤
- `[0.0, 1.0, 0.0]` - 緑
- `[0.0, 0.0, 1.0]` - 青
- `[1.0, 1.0, 0.0]` - 黄
