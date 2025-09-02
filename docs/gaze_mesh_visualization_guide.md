# T-pose メッシュ視線可視化システム使用ガイド

## 概要

T-pose メッシュ視線可視化システムは、SMPL モデルの T-pose（標準ポーズ）メッシュに視線データを適用し、視線の強度をカラーマップで可視化するツールです。フレーム別の視線パターンを 3D メッシュ上で直感的に理解することができます。

## システム構成

### 主要コンポーネント

1. **`src/gaze_mesh_visualizer.py`**: 視線メッシュ可視化の核となるクラス
2. **`utils/gaze_mesh_viewer.py`**: コマンドライン実行インターフェース
3. **`utils/create_tpose_mesh.py`**: SMPL モデルから T-pose メッシュ生成
4. **`utils/create_sample_tpose.py`**: テスト用サンプルメッシュ生成

### 処理フロー

```
T-poseメッシュ (.obj) + 視線データ (.npy) → 視線付きメッシュ (.obj) + 可視化画像 (.png)
```

## 前提条件

### 必要なライブラリ

```bash
uv add numpy trimesh matplotlib pandas pillow scipy smplx
```

### 入力データ要件

1. **T-pose メッシュファイル**:

   - 形式: `.obj`
   - 頂点数: 6890 個（SMPL モデル標準）
   - 形状: T-pose（腕を水平に伸ばした標準ポーズ）

2. **視線データファイル**:
   - 形式: `.npy`
   - データ形状: `(6890,)` または `(6890, 1)`
   - 値: 各頂点への視線強度（0 以上の実数）

## 基本的な使い方

### 1. T-pose メッシュの準備

#### 方法 A: SMPL モデルから T-pose メッシュを生成

```bash
# SMPLモデルファイルが必要
uv run utils/create_tpose_mesh.py SMPL_NEUTRAL.npz --output tpose_mesh.obj

# 性別指定
uv run utils/create_tpose_mesh.py SMPL_MALE.npz --gender male --output tpose_male.obj
```

#### 方法 B: サンプル T-pose メッシュの生成（テスト用）

```bash
uv run utils/create_sample_tpose.py
# → sample_tpose.obj が生成される
```

### 2. 視線データの可視化

#### 基本コマンド

```bash
# 最も基本的な実行
uv run utils/gaze_mesh_viewer.py tpose_mesh.obj data/

# 詳細ログ付き実行
uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/ --verbose
```

#### 単一フレーム処理

```bash
# フレーム76のみ処理
uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/ --single-frame 76

# 出力先指定
uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/ --single-frame 76 --output results/
```

#### バッチ処理

```bash
# 全フレーム処理
uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/

# フレーム範囲指定
uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/ --frame-range 0 100

# 画像出力なし（メッシュファイルのみ）
uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/ --no-images
```

## 詳細オプション

### カラーマップの選択

```bash
# 利用可能なカラーマップ
uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/ --colormap viridis
uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/ --colormap plasma
uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/ --colormap hot      # デフォルト
uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/ --colormap cool
uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/ --colormap inferno
```

#### カラーマップの特徴

| カラーマップ | 特徴              | 用途               |
| ------------ | ----------------- | ------------------ |
| `hot`        | 黒 → 赤 → 黄 → 白 | 一般的な熱分布表示 |
| `viridis`    | 紫 → 青 → 緑 → 黄 | 科学的可視化に適用 |
| `plasma`     | 紫 → ピンク → 黄  | 高コントラスト表示 |
| `cool`       | 青 → 水色 → 緑    | 冷たい印象の可視化 |
| `inferno`    | 黒 → 紫 → 赤 → 黄 | 暗い背景に適用     |

### 正規化方式

```bash
# グローバル正規化（デフォルト）- 全フレーム通しての最小・最大値で正規化
uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/ --normalization global

# フレーム正規化 - 各フレーム内での最小・最大値で正規化
uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/ --normalization frame

# パーセンタイル正規化 - 5%〜95%パーセンタイルで正規化
uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/ --normalization percentile
```

### ファイル形式

```bash
# 出力メッシュファイル形式
uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/ --format obj  # デフォルト
uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/ --format ply
uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/ --format stl
```

### 画像設定

```bash
# 画像解像度設定
uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/ --resolution 1920 1080

# 視点角度設定
uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/ --view-angle 30 60

# 画像出力スキップ
uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/ --no-images

# 凡例生成スキップ
uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/ --no-legend
```

## 出力ファイル

### ファイル構造

```
output/gaze_meshes/
├── meshes/                         # メッシュファイル
│   ├── gaze_frame_000076.obj       # フレーム別視線メッシュ
│   ├── gaze_frame_000077.obj
│   └── ...
├── images/                         # 可視化画像
│   ├── gaze_frame_000076.png       # フレーム別可視化画像
│   ├── gaze_frame_000077.png
│   └── ...
└── colorbar_legend.png             # カラーバー凡例
```

### 出力ファイルの詳細

#### 1. 視線メッシュファイル (`gaze_frame_XXXXXX.obj`)

- **形式**: Wavefront OBJ 形式
- **内容**: T-pose メッシュの各頂点に視線強度に応じた色が適用
- **用途**: 3D ソフトウェアでの詳細分析、アニメーション作成

#### 2. 可視化画像 (`gaze_frame_XXXXXX.png`)

- **形式**: PNG 画像
- **内容**: 3D 視点からの視線メッシュレンダリング
- **用途**: プレゼンテーション、レポート作成

#### 3. カラーバー凡例 (`colorbar_legend.png`)

- **形式**: PNG 画像
- **内容**: 視線強度とカラーマップの対応関係
- **用途**: 可視化結果の解釈、論文・レポートでの参照

## 実践的な使用例

### 例 1: 単一フレームの詳細分析

```bash
# フレーム76の詳細分析
uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/ \
    --single-frame 76 \
    --colormap viridis \
    --normalization frame \
    --resolution 1920 1080 \
    --output analysis/frame_76/
```

**用途**: 特定フレームの視線パターンの詳細分析

### 例 2: 時系列比較分析

```bash
# 連続フレームの比較
uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/ \
    --frame-range 70 80 \
    --colormap plasma \
    --normalization global \
    --output comparison/frames_70_80/
```

**用途**: 時間経過による視線パターンの変化分析

### 例 3: プレゼンテーション用画像作成

```bash
# 高解像度画像の作成
uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/ \
    --single-frame 76 \
    --colormap hot \
    --resolution 2560 1440 \
    --view-angle 45 45 \
    --output presentation/
```

**用途**: 学会発表、論文掲載用の高品質画像

### 例 4: 大量フレームの一括処理

```bash
# 全フレームのメッシュ生成（画像なし）
uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/ \
    --no-images \
    --format ply \
    --normalization global \
    --output batch_processing/
```

**用途**: 3D ソフトウェアでのアニメーション作成

## プログラム内での使用

### 基本的な使い方

```python
from pathlib import Path
from src.gaze_mesh_visualizer import GazeMeshVisualizer

# 可視化器の初期化
visualizer = GazeMeshVisualizer(
    obj_path="sample_tpose.obj",
    segmentation_file="smpl_vert_segmentation.json"
)

# 完全な可視化処理
result = visualizer.run_visualization(
    data_folder=Path("data/"),
    output_dir=Path("output/"),
    normalization="global",
    colormap="viridis"
)

print(f"生成されたメッシュファイル: {len(result['mesh_files'])}個")
```

### 詳細制御

```python
# ベースメッシュの読み込み
visualizer.load_base_mesh()

# 視線データの読み込み
visualizer.load_gaze_data(Path("data/"))

# 単一フレームの視線メッシュ作成
mesh = visualizer.create_gaze_mesh(
    frame_num=76,
    normalization="frame",
    colormap="plasma"
)

# 画像の生成
visualizer.create_gaze_visualization(
    frame_num=76,
    save_path=Path("output/frame_76.png"),
    resolution=(1920, 1080)
)

# 統計情報の取得
stats = visualizer.get_gaze_statistics()
print(f"視線値範囲: {stats['gaze_range']['min']:.3f} - {stats['gaze_range']['max']:.3f}")
```

## トラブルシューティング

### よくある問題と解決方法

#### 1. モジュールエラー

```bash
ModuleNotFoundError: No module named 'trimesh'
```

**解決方法**:

```bash
uv add trimesh matplotlib pandas pillow scipy
```

#### 2. 頂点数不一致エラー

```
警告: 頂点数不一致: データ=6890, メッシュ=3000
```

**解決方法**:

- T-pose メッシュが 6890 頂点を持つことを確認
- 適切な SMPL モデルから生成されたメッシュを使用

#### 3. 画像生成エラー

```
画像レンダリングエラー: No module named 'pyglet'
```

**解決方法**:

- 自動的に matplotlib フォールバックが実行されます
- より高品質な 3D 描画が必要な場合: `uv add pyglet`

#### 4. メモリ不足エラー

```
MemoryError: Unable to allocate array
```

**解決方法**:

- フレーム範囲を制限: `--frame-range 0 100`
- 画像出力を無効化: `--no-images`
- より小さな解像度を使用: `--resolution 800 600`

### デバッグ方法

```bash
# 詳細ログの有効化
uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/ --verbose

# Python内でのデバッグ
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 性能最適化

### 処理速度の改善

1. **画像生成のスキップ**:

   ```bash
   uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/ --no-images
   ```

2. **フレーム範囲の制限**:

   ```bash
   uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/ --frame-range 0 50
   ```

3. **低解像度での確認**:
   ```bash
   uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/ --resolution 400 300
   ```

### メモリ使用量の削減

1. **バッチサイズの調整**:

   - 1000 フレーム以上の場合は分割処理を推奨

2. **ファイル形式の選択**:
   - STL 形式は最も軽量: `--format stl`

## カスタマイズ

### 独自のカラーマップ作成

```python
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# カスタムカラーマップの定義
colors = ['blue', 'cyan', 'yellow', 'red']
custom_cmap = LinearSegmentedColormap.from_list('custom', colors)

# 可視化器での使用
visualizer = GazeMeshVisualizer("sample_tpose.obj")
visualizer.colormap = custom_cmap
```

### 視線データの前処理

```python
import numpy as np

# 視線データの読み込み
data = np.load("data/000076.npy")

# 閾値処理
data[data < 0.1] = 0  # 0.1未満を0に設定

# 対数スケール変換
data = np.log1p(data)  # log(1 + x)変換

# 平滑化
from scipy.ndimage import gaussian_filter1d
data = gaussian_filter1d(data, sigma=1.0)
```

## 応用例

### 1. 時系列アニメーション作成

```bash
# 全フレームのメッシュ生成
uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/ --format obj

# Blenderでアニメーション作成
# 1. 生成されたOBJファイルを順次読み込み
# 2. キーフレームアニメーションを作成
# 3. 動画として出力
```

### 2. 統計的分析

```python
# 視線統計の取得
stats = visualizer.get_gaze_statistics()

# 部位別分析との組み合わせ
from src.gaze_part_analyzer import GazePartAnalyzer
part_analyzer = GazePartAnalyzer()
part_results = part_analyzer.run_analysis(Path("data/"))

# 3D可視化と統計分析の統合
combined_analysis = {
    'mesh_visualization': result,
    'part_analysis': part_results,
    'statistics': stats
}
```

### 3. 複数被験者の比較

```bash
# 被験者Aの視線可視化
uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/subject_A/ \
    --output results/subject_A/ --colormap viridis

# 被験者Bの視線可視化
uv run utils/gaze_mesh_viewer.py sample_tpose.obj data/subject_B/ \
    --output results/subject_B/ --colormap viridis

# 同一正規化で比較分析
```

## まとめ

T-pose メッシュ視線可視化システムは、視線データの 3D 可視化を通じて、人間の視線パターンを直感的に理解するための強力なツールです。コマンドライン操作とプログラム内使用の両方に対応し、研究から実用的な分析まで幅広く活用できます。

詳細な機能や技術仕様については、以下のドキュメントも参照してください：

- [`gaze_part_analysis_guide.md`](./gaze_part_analysis_guide.md): 部位別視線分析システム
- [`api_reference.md`](./api_reference.md): API 詳細仕様
- [`tutorial.md`](./tutorial.md): 基本的なチュートリアル

---

本システムは研究・教育目的での使用を想定しています。商用利用の場合は適切なライセンス確認を行ってください。
