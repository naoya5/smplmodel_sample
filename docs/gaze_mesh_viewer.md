gaze_mesh_viewer.py は T-pose メッシュに視線デー
タを適用してカラーマップで可視化するツールです。

基本的な使い方

uv run utils/gaze_mesh_viewer.py tpose.obj data/

必須引数

- obj_path: T-pose の OBJ ファイルパス
- data_folder: 視線データ (.npy ファイル)
  が格納されたフォルダパス

主要オプション

| オプション | 説明
| デフォルト |
|---------------------|-------------------------
--------|--------------------|
| --output, -o | 出力ディレクトリ
| output/gaze_meshes |
| --colormap, -c | カラーマップ名
| hot |
| --normalization, -n | 正規化方法
(global/frame/percentile) | global |
| --frame-range | 処理フレーム範囲
| 全フレーム |
| --single-frame | 単一フレーム処理
| なし |
| --format, -f | 出力形式 (obj/ply/stl)
| obj |
| --no-images | 画像出力をスキップ
| false |
| --view-angle | 画像の視点角度
| 45 45 |
| --resolution | 画像解像度
| 800 600 |

使用例

# 基本実行

python utils/gaze_mesh_viewer.py tpose.obj data/

# カラーマップとフレーム範囲指定

python utils/gaze_mesh_viewer.py tpose.obj data/
--colormap viridis --frame-range 0 100

# 単一フレーム処理

python utils/gaze_mesh_viewer.py tpose.obj data/
--single-frame 50

# メッシュのみ出力（画像なし）

python utils/gaze_mesh_viewer.py tpose.obj data/
--no-images
