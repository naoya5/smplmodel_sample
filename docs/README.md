# SMPL Model Sample - 視線データ分布解析システム

SMPL（Skinned Multi-Person Linear Model）を使用した 3D ヒューマンメッシュ処理と視線データ分布解析を行うPythonライブラリです。

## 主な機能

### 従来機能
- **4D-Humans PKL ファイルの読み込み**: zlib 圧縮された joblib 形式のファイルを解析
- **SMPL メッシュ生成**: パラメータからの 3D メッシュ作成
- **部位別着色**: 解剖学的部位に基づいた色付け
- **OBJ ファイル処理**: 既存の OBJ ファイルへの着色適用

### 新機能（視線データ分布解析）
- **視線データの読み込み**: CSV、JSON形式の視線データに対応
- **3D空間での視線-メッシュ関係計算**: 距離・角度ベースの確率計算
- **時系列解析**: 時間軸での視線パターン追跡
- **確率分布モデリング**: ガウス混合モデルによる分布学習
- **可視化**: ヒートマップとグラフによる結果表示
- **多様な出力形式**: CSV、JSON、画像ファイル対応

## プロジェクト構成

```
smplmodel_sample/
├── src/
│   ├── obj2obj.py                      # OBJファイル間での変換・着色処理
│   ├── pkl2obj.py                      # PKLファイルからOBJファイルへの変換
│   └── gaze_distribution_analyzer.py   # 視線データ分布解析（新機能）
├── examples/
│   ├── example_usage.py                # 使用例とデモ
│   └── sample_data/                    # サンプルデータ
└── docs/
    ├── README.md                       # プロジェクト概要（このファイル）
    ├── gaze_analysis_guide.md          # 視線解析システムの詳細ガイド
    ├── api_reference.md                # API リファレンス
    └── tutorial.md                     # チュートリアル
```

## 必要な依存関係

- `torch` - PyTorch フレームワーク
- `smplx` - SMPL モデル実装
- `trimesh` - 3D メッシュ処理
- `numpy` - 数値計算
- `scipy` - 科学計算
- `scikit-learn` - 機械学習
- `matplotlib` - 可視化
- `requests` - HTTP リクエスト
- `joblib` - シリアライゼーション

## インストール

```bash
# プロジェクトの初期化
uv init

# 必要なライブラリのインストール
uv add numpy scipy scikit-learn matplotlib trimesh smplx torch requests joblib
```

## クイックスタート

### 従来機能（SMPLメッシュ処理）

```python
# PKLファイルからSMPLメッシュを生成
from src.pkl2obj import process_4d_humans_pkl_smpl

mesh = process_4d_humans_pkl_smpl(
    pkl_path="your_data.pkl",
    smpl_model_path="SMPL_NEUTRAL.npz",
    frame_idx=0,
    person_idx=0,
    part_name="head",
    color=[1.0, 0.0, 0.0]
)
mesh.show()
```

### 新機能（視線データ分布解析）

```python
from src.gaze_distribution_analyzer import (
    GazeDataLoader, MeshProcessor, GazeDistributionAnalyzer, ResultExporter
)

# 視線データの読み込み
gaze_data = GazeDataLoader.load_from_csv("gaze_data.csv")

# メッシュの処理
mesh_processor = MeshProcessor.load_from_obj("model.obj")

# 解析の実行
analyzer = GazeDistributionAnalyzer(mesh_processor)
temporal_distribution = analyzer.analyze_temporal_distribution(gaze_data)

# 結果の出力
ResultExporter.export_to_csv(temporal_distribution, "results.csv")
ResultExporter.visualize_temporal_heatmap(temporal_distribution, mesh_processor)
```

### 使用例の実行

```bash
# 詳細な使用例とデモの実行
uv run --frozen python examples/example_usage.py
```

## 視線データフォーマット

### CSVフォーマット
```csv
timestamp,gaze_x,gaze_y,gaze_z,origin_x,origin_y,origin_z,confidence
0.0,0.1,0.2,0.9,0.0,0.0,2.0,0.95
0.1,0.15,0.18,0.92,0.0,0.0,2.0,0.87
```

### JSONフォーマット
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

## 主要コンポーネント

### 視線データ分布解析システム

1. **GazeDataLoader**: 視線データの読み込みと前処理
2. **MeshProcessor**: 3Dメッシュの処理と頂点データ管理
3. **SpatialRelationCalculator**: 視線と頂点の空間関係計算
4. **GazeDistributionAnalyzer**: 分布確率の計算と時系列解析
5. **ResultExporter**: 結果の可視化とエクスポート

## 応用例

1. **VR/AR研究**: 仮想環境での視線行動分析
2. **人間工学**: 製品デザインの視線誘導効果測定
3. **認知科学**: 人物認識時の視線パターン解析
4. **医療**: 視線追跡による診断支援

## ドキュメント

- [視線解析システム詳細ガイド](gaze_analysis_guide.md)
- [APIリファレンス](api_reference.md)
- [チュートリアル](tutorial.md)

詳細な使用方法については、各ドキュメントをご参照ください。
