# 視線データ分布解析システム ガイド

## 概要

視線データ分布解析システムは、視線追跡データと3Dオブジェクトメッシュを組み合わせて、時系列ごとの各頂点への視線の分布確率を計算するPythonライブラリです。

### 主な機能

- **視線データの読み込み**: CSV、JSON、NumPy形式の視線データに対応
- **3Dメッシュ処理**: OBJファイルやtrimeshオブジェクトの処理
- **空間関係計算**: 視線と頂点の距離・角度関係の計算
- **確率分布モデリング**: ガウス混合モデルによる分布の学習
- **時系列解析**: 時間軸での視線パターンの追跡
- **結果可視化**: ヒートマップとグラフによる結果表示

## インストール

必要なライブラリをuvでインストールします：

```bash
uv add numpy scipy scikit-learn matplotlib trimesh
```

## クイックスタート

### 1. 基本的な使用例

```python
from src.gaze_distribution_analyzer import (
    GazeDataLoader, MeshProcessor, GazeDistributionAnalyzer, ResultExporter
)

# 視線データの読み込み
gaze_data = GazeDataLoader.load_from_csv("gaze_data.csv")

# メッシュの読み込み
mesh_processor = MeshProcessor.load_from_obj("model.obj")

# 解析器の初期化
analyzer = GazeDistributionAnalyzer(mesh_processor)

# 時系列解析の実行
temporal_distribution = analyzer.analyze_temporal_distribution(gaze_data)

# 結果の出力
ResultExporter.export_to_csv(temporal_distribution, "results.csv")
ResultExporter.visualize_temporal_heatmap(temporal_distribution, mesh_processor)
```

### 2. 既存SMPLモデルとの統合

```python
from src.pkl2obj import process_4d_humans_pkl_smpl
from src.gaze_distribution_analyzer import MeshProcessor, GazeDistributionAnalyzer

# SMPLメッシュの生成
smpl_mesh = process_4d_humans_pkl_smpl(
    pkl_path="demo.pkl",
    smpl_model_path="SMPL_NEUTRAL.npz",
    frame_idx=0
)

# 視線解析
mesh_processor = MeshProcessor(smpl_mesh)
analyzer = GazeDistributionAnalyzer(mesh_processor)

# 解析の実行
gaze_data = GazeDataLoader.load_from_csv("eye_tracking.csv")
results = analyzer.analyze_temporal_distribution(gaze_data)
```

## データフォーマット

### 視線データCSVフォーマット

```csv
timestamp,gaze_x,gaze_y,gaze_z,origin_x,origin_y,origin_z,confidence
0.0,0.1,0.2,0.9,0.0,0.0,2.0,0.95
0.1,0.15,0.18,0.92,0.0,0.0,2.0,0.87
```

| カラム名 | 説明 | 型 |
|---------|------|---|
| timestamp | 時刻（秒） | float |
| gaze_x, gaze_y, gaze_z | 視線方向ベクトル | float |
| origin_x, origin_y, origin_z | 視線原点（カメラ/目の位置） | float |
| confidence | 信頼度（0.0-1.0） | float |

### 視線データJSONフォーマット

```json
{
  "gaze_data": [
    {
      "timestamp": 0.0,
      "direction": [0.1, 0.2, 0.9],
      "origin": [0.0, 0.0, 2.0],
      "confidence": 0.95
    }
  ]
}
```

## APIリファレンス

### GazePoint クラス

視線データポイントを表すデータクラス

```python
@dataclass
class GazePoint:
    timestamp: float        # タイムスタンプ（秒）
    direction: np.ndarray   # 視線方向ベクトル (x, y, z)
    origin: np.ndarray      # 視線原点 (x, y, z)
    confidence: float = 1.0 # 信頼度 (0.0-1.0)
```

### VertexGazeProbability クラス

頂点ごとの視線確率データ

```python
@dataclass
class VertexGazeProbability:
    vertex_index: int       # 頂点インデックス
    probability: float      # 視線確率
    distance_score: float   # 距離スコア
    angle_score: float      # 角度スコア
    temporal_weight: float  # 時間的重み
```

### GazeDataLoader クラス

#### メソッド

##### `load_from_csv(csv_path: str) -> List[GazePoint]`

CSVファイルから視線データを読み込みます。

**パラメータ:**
- `csv_path`: CSVファイルのパス

**戻り値:**
- 視線データポイントのリスト

**例外:**
- `FileNotFoundError`: ファイルが見つからない場合
- `ValueError`: データフォーマットが不正な場合

##### `load_from_json(json_path: str) -> List[GazePoint]`

JSONファイルから視線データを読み込みます。

##### `normalize_gaze_directions(gaze_points: List[GazePoint]) -> List[GazePoint]`

視線方向ベクトルを正規化します。

### MeshProcessor クラス

#### 初期化

```python
MeshProcessor(mesh: trimesh.Trimesh)
```

#### クラスメソッド

##### `load_from_obj(obj_path: str) -> MeshProcessor`

OBJファイルからメッシュを読み込みます。

#### メソッド

##### `get_vertex_normals() -> np.ndarray`

各頂点の法線ベクトルを取得します。

##### `get_face_centers() -> np.ndarray`

各面の中心点を取得します。

##### `calculate_surface_area() -> float`

メッシュの表面積を計算します。

### SpatialRelationCalculator クラス

#### 静的メソッド

##### `point_to_line_distance(point: np.ndarray, line_origin: np.ndarray, line_direction: np.ndarray) -> float`

点から直線（視線）への最短距離を計算します。

**パラメータ:**
- `point`: 3D点座標
- `line_origin`: 直線の原点
- `line_direction`: 直線の方向ベクトル（正規化済み）

**戻り値:**
- 最短距離

##### `calculate_gaze_angle(vertex: np.ndarray, gaze_origin: np.ndarray, gaze_direction: np.ndarray, vertex_normal: np.ndarray) -> float`

視線と頂点法線の角度を計算します。

### GazeDistributionAnalyzer クラス

#### 初期化

```python
GazeDistributionAnalyzer(
    mesh_processor: MeshProcessor,
    distance_threshold: float = 0.1,      # 視線距離の閾値
    angle_threshold: float = π/4,         # 視線角度の閾値（ラジアン）
    temporal_window: float = 1.0          # 時間窓のサイズ（秒）
)
```

#### メソッド

##### `calculate_vertex_probabilities(gaze_points: List[GazePoint], timestamp: float) -> List[VertexGazeProbability]`

指定時刻における各頂点への視線確率を計算します。

**パラメータ:**
- `gaze_points`: 視線データポイントのリスト
- `timestamp`: 対象時刻

**戻り値:**
- 頂点ごとの視線確率リスト

##### `analyze_temporal_distribution(gaze_points: List[GazePoint], time_step: float = 0.1) -> Dict[float, List[VertexGazeProbability]]`

時系列にわたる視線分布を解析します。

**パラメータ:**
- `gaze_points`: 視線データポイントのリスト
- `time_step`: 時間ステップ（秒）

**戻り値:**
- 時刻ごとの頂点確率辞書

##### `fit_gaussian_mixture_model(temporal_distribution: Dict, n_components: int = 3) -> GaussianMixture`

視線分布データにガウス混合モデルを適用します。

### ResultExporter クラス

#### 静的メソッド

##### `export_to_csv(temporal_distribution: Dict, output_path: str) -> None`

解析結果をCSVファイルにエクスポートします。

##### `export_to_json(temporal_distribution: Dict, output_path: str) -> None`

解析結果をJSONファイルにエクスポートします。

##### `visualize_temporal_heatmap(temporal_distribution: Dict, mesh_processor: MeshProcessor, output_path: str = None) -> None`

時系列視線分布のヒートマップを可視化します。

## 詳細な使用例

### 例1: リアルタイム視線解析

```python
import numpy as np
from src.gaze_distribution_analyzer import *

# リアルタイムデータストリームの設定
def process_realtime_gaze(gaze_stream, mesh_file):
    # メッシュの読み込み
    mesh_processor = MeshProcessor.load_from_obj(mesh_file)
    analyzer = GazeDistributionAnalyzer(
        mesh_processor,
        distance_threshold=0.05,  # より厳密な距離閾値
        temporal_window=0.5       # 短い時間窓
    )
    
    gaze_buffer = []
    current_time = 0.0
    
    for gaze_point in gaze_stream:
        gaze_buffer.append(gaze_point)
        
        # バッファサイズの制限
        if len(gaze_buffer) > 100:
            gaze_buffer.pop(0)
        
        # 現在時刻の視線確率を計算
        vertex_probs = analyzer.calculate_vertex_probabilities(
            gaze_buffer, current_time
        )
        
        # 最も注目されている頂点を特定
        max_prob_vertex = max(vertex_probs, key=lambda x: x.probability)
        print(f"時刻 {current_time:.1f}s: 頂点 {max_prob_vertex.vertex_index} "
              f"(確率: {max_prob_vertex.probability:.3f})")
        
        current_time += 0.1
```

### 例2: 複数オブジェクトの比較解析

```python
def compare_gaze_patterns(objects, gaze_data_files):
    results = {}
    
    for obj_name, obj_file in objects.items():
        # オブジェクトごとの解析
        mesh_processor = MeshProcessor.load_from_obj(obj_file)
        analyzer = GazeDistributionAnalyzer(mesh_processor)
        
        gaze_data = GazeDataLoader.load_from_csv(gaze_data_files[obj_name])
        temporal_dist = analyzer.analyze_temporal_distribution(gaze_data)
        
        # 注目度の統計計算
        total_attention = 0
        for timestamp, vertex_probs in temporal_dist.items():
            total_attention += sum(vp.probability for vp in vertex_probs)
        
        results[obj_name] = {
            'total_attention': total_attention,
            'distribution': temporal_dist
        }
    
    # 結果の比較
    for obj_name, result in results.items():
        print(f"{obj_name}: 総注目度 = {result['total_attention']:.2f}")
```

### 例3: ガウス混合モデルによる視線パターン分類

```python
def classify_gaze_patterns(temporal_distribution):
    analyzer = GazeDistributionAnalyzer(mesh_processor)
    
    # ガウス混合モデルの学習
    gmm = analyzer.fit_gaussian_mixture_model(temporal_distribution, n_components=5)
    
    # パターンの分類
    features = []
    timestamps = []
    
    for timestamp, vertex_probs in temporal_distribution.items():
        for vp in vertex_probs:
            if vp.probability > 0.1:
                feature = [
                    timestamp, vp.vertex_index, vp.probability,
                    vp.distance_score, vp.angle_score
                ]
                features.append(feature)
                timestamps.append(timestamp)
    
    # クラスタリング結果
    features_array = np.array(features)
    cluster_labels = gmm.predict(features_array)
    
    # 時系列パターンの分析
    pattern_sequences = {}
    for timestamp, label in zip(timestamps, cluster_labels):
        if timestamp not in pattern_sequences:
            pattern_sequences[timestamp] = []
        pattern_sequences[timestamp].append(label)
    
    return pattern_sequences
```

## トラブルシューティング

### よくある問題と解決法

#### 1. メモリ不足エラー

**問題**: 大きなメッシュや長時間の視線データでメモリが不足する

**解決法**:
- `time_step`を大きくして解析点数を減らす
- `temporal_window`を小さくしてメモリ使用量を削減
- チャンク処理で分割して実行

```python
# チャンク処理の例
def process_large_dataset(gaze_data, mesh_processor, chunk_size=1000):
    results = {}
    
    for i in range(0, len(gaze_data), chunk_size):
        chunk = gaze_data[i:i+chunk_size]
        analyzer = GazeDistributionAnalyzer(mesh_processor)
        
        chunk_results = analyzer.analyze_temporal_distribution(
            chunk, time_step=0.5
        )
        results.update(chunk_results)
    
    return results
```

#### 2. 精度が低い場合

**問題**: 視線確率の計算精度が期待より低い

**解決法**:
- `distance_threshold`と`angle_threshold`を調整
- 視線データの品質を確認
- 視線方向の正規化を実行

```python
# パラメータ調整の例
analyzer = GazeDistributionAnalyzer(
    mesh_processor,
    distance_threshold=0.02,  # より厳密
    angle_threshold=np.pi/6,  # より狭い角度
    temporal_window=0.3       # より短い時間窓
)
```

#### 3. 可視化エラー

**問題**: ヒートマップの生成でエラーが発生

**解決法**:
- データの形状を確認
- Matplotlibのバックエンドを設定

```python
import matplotlib
matplotlib.use('Agg')  # GUI不要のバックエンド
```

## パフォーマンス最適化

### 計算高速化のヒント

1. **NumPy vectorization**: ループの代わりにNumPy配列操作を使用
2. **並列処理**: `multiprocessing`で時刻ごとの計算を並列化
3. **メモリ効率**: 不要なデータは即座に削除
4. **事前計算**: 頂点法線などは事前に計算して保存

### 推奨システム要件

- **RAM**: 8GB以上（大規模データセットの場合は16GB以上）
- **CPU**: マルチコア推奨
- **ストレージ**: 結果出力用に十分な空き容量

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 更新履歴

- **v1.0.0** (2025-07-10): 初回リリース
  - 基本的な視線分布解析機能
  - CSV/JSON形式のデータ読み込み
  - ヒートマップ可視化
  - ガウス混合モデル対応

## 貢献

バグ報告や機能改善の提案は、GitHubのIssuesページでお願いします。

## サポート

技術的な質問やサポートが必要な場合は、プロジェクトのドキュメントを参照するか、開発チームにお問い合わせください。