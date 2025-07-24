# 視線データ部位別分析システム

## 概要

本システムは、SMPL モデルの 3D ヒューマンメッシュに対する視線データを部位別に分析するツールです。`.npy`形式の視線データファイルを読み込み、SMPL モデルの各解剖学的部位（頭部、首、胴体、手足など）への視線集中度を定量的に計算・分析します。

## システム構成

### メインコンポーネント

1. **`src/gaze_part_analyzer.py`**: 視線分析の核となるクラス
2. **`utils/gaze_part_viewer.py`**: コマンドライン実行インターフェース
3. **SMPL セグメンテーションファイル**: 頂点と部位の対応データ

### データフロー

```
.npy視線データ → 部位別集計 → 統計計算 → 結果出力・可視化
    ↓               ↓           ↓            ↓
  6890頂点値    → 24部位別値 → 確率分布  → CSV/JSON/PNG
```

## データ仕様

### 入力データ形式

- **ファイル形式**: `.npy` (NumPy 配列)
- **データ形状**: `(6890, 1)` または `(6890,)`
- **データ型**: `float64`
- **値の範囲**: `0.0` 以上の実数値
- **意味**: 各頂点への視線注視度・密度

### ファイル命名規則

```
000076.npy  # フレーム番号76
001234.npy  # フレーム番号1234
frame_N.npy # フレーム番号N
```

プログラムはファイル名から数値部分を抽出してフレーム番号として使用します。

## SMPL 部位分類

本システムは 24 の解剖学的部位を認識します：

### 頭部・首

- **head** (589 頂点): 頭部全体
- **neck** (129 頂点): 首

### 胴体

- **spine** (228 頂点): 腰椎部
- **spine1** (280 頂点): 中胸椎部
- **spine2** (372 頂点): 上胸椎部

### 上肢

- **leftShoulder** / **rightShoulder** (130 頂点): 肩
- **leftArm** / **rightArm** (280 頂点): 上腕
- **leftForeArm** / **rightForeArm** (184/280 頂点): 前腕

### 手

- **leftHand** / **rightHand** (436/239 頂点): 手のひら・指
- **leftHandIndex1** / **rightHandIndex1** (436 頂点): 人差し指

### 下肢

- **leftUpLeg** / **rightUpLeg** (228 頂点): 大腿部
- **leftLeg** / **rightLeg** (214/280 頂点): 下腿部

### 足

- **leftFoot** / **rightFoot** (129/143 頂点): 足部
- **leftToeBase** / **rightToeBase** (122 頂点): つま先

## 使用方法

### 基本実行

```bash
# 指定フォルダの.npyファイルを分析
python utils/gaze_part_viewer.py data/

# 詳細ログ付き実行
uv run utils/gaze_part_viewer.py data/ --verbose

# 出力先ディレクトリ指定
uv run utils/gaze_part_viewer.py data/ --output results/

uv run utils/gaze_part_viewer.py data/ --output results/ --verbos
```

### 高度なオプション

```bash
# 視線値の閾値設定（0.1未満の値を除外）
python utils/gaze_part_viewer.py data/ --threshold 0.1

# 可視化グラフの生成をスキップ
python utils/gaze_part_viewer.py data/ --no-visualization

# カスタムセグメンテーションファイル使用
python utils/gaze_part_viewer.py data/ --segmentation my_segmentation.json
```

### プログラム内での使用

```python
from src.gaze_part_analyzer import GazePartAnalyzer
from pathlib import Path

# 分析器の初期化（90Hzサンプリングレート）
analyzer = GazePartAnalyzer(frame_rate=90.0)

# 他のフレームレートも指定可能
analyzer_30fps = GazePartAnalyzer(frame_rate=30.0)  # 30fps
analyzer_60fps = GazePartAnalyzer(frame_rate=60.0)  # 60fps

# データ読み込みと分析実行
analyzer.run_analysis(
    data_folder=Path("data/"),
    output_dir=Path("output/")
)

# 結果取得
summary = analyzer.get_summary_statistics()
part_results = analyzer.part_results
frame_results = analyzer.frame_results

# 時間変換機能の使用
timestamp = analyzer.get_timestamp_from_frame(100)  # フレーム100の秒数
time_str = analyzer.get_time_formatted(100)  # "0:01.111"形式

# 視線変化検出
gaze_changes = analyzer.detect_gaze_changes(min_change_threshold=0.1)
```

## 出力ファイル

### 1. 部位別分析結果 (`part_gaze_analysis.csv`)

各部位の統計データを含む CSV ファイル：

| 列名               | 説明                     |
| ------------------ | ------------------------ |
| part_name          | 部位名                   |
| total_gaze         | 全フレームでの視線値合計 |
| probability        | 視線確率（0-1）          |
| average_per_frame  | フレーム当たり平均視線値 |
| average_per_vertex | 頂点当たり平均視線値     |
| vertex_count       | 部位の頂点数             |

### 2. フレーム別分析結果 (`frame_gaze_analysis.csv`)

各フレームの詳細データを含む CSV ファイル：

| 列名            | 説明                     |
| --------------- | ------------------------ |
| frame           | フレーム番号             |
| timestamp       | 時間（秒）               |
| time_formatted  | 時:分:秒形式の時間       |
| max_part        | 最大注視部位             |
| max_value       | 最大視線値               |
| total_gaze      | フレーム内視線値合計     |
| {部位名}\_value | 各部位の視線値           |
| {部位名}\_ratio | 各部位の視線割合         |

**時間情報の例**:
- `timestamp`: `3.222` (3.222秒)
- `time_formatted`: `0:03.222` (0分3.222秒)

### 3. 統計サマリー (`gaze_analysis_summary.json`)

```json
{
  "total_frames": 150,
  "total_parts": 24,
  "top_parts_by_probability": [
    {
      "part": "head",
      "probability": 0.519,
      "total_gaze": 3120.26,
      "avg_per_frame": 20.8
    }
  ],
  "most_attended_frames": [
    {
      "part": "head",
      "frame_count": 95,
      "percentage": 63.3
    }
  ]
}
```

### 4. 視線変化分析結果 (`gaze_changes.json`)

視線の部位間移動を時系列で記録した JSON ファイル：

```json
{
  "frame_rate": 90.0,
  "total_changes": 15,
  "changes": [
    {
      "timestamp": 3.200,
      "time_formatted": "0:03.200",
      "frame": 288,
      "from_part": "head",
      "to_part": "rightHand",
      "confidence": 0.456,
      "description": "3.2秒: headからrightHandに移動"
    },
    {
      "timestamp": 5.678,
      "time_formatted": "0:05.678",
      "frame": 511,
      "from_part": "rightHand",
      "to_part": "leftArm",
      "confidence": 0.321,
      "description": "5.7秒: rightHandからleftArmに移動"
    }
  ]
}
```

**フィールド説明**:
- `timestamp`: 変化が発生した時間（秒）
- `time_formatted`: 時:分:秒形式の時間
- `frame`: 該当フレーム番号
- `from_part`: 変化前の注視部位
- `to_part`: 変化後の注視部位
- `confidence`: 新しい部位への視線集中度（0.0-1.0）
- `description`: 変化の日本語説明

### 5. 可視化グラフ (`gaze_part_analysis.png`)

2 つのグラフを含む PNG 画像：

1. **部位別視線分布**: 各部位への視線確率の棒グラフ（上位 15 位）
2. **フレーム別最大注視部位推移**: 時間軸での注視部位変化（最初の 50 フレーム）

## 分析指標

### 視線確率 (Gaze Probability)

各部位への視線の集中度を示す基本指標：

```
部位iの視線確率 = 部位iの総視線値 / 全部位の総視線値
```

- **範囲**: 0.0 - 1.0
- **解釈**: 値が大きいほど注視されている部位

### フレーム別最大注視部位

各フレームで最も視線値が高い部位：

```python
max_part = argmax(各部位の視線値)
```

### 頂点正規化視線値

部位の大きさ（頂点数）を考慮した指標：

```
頂点正規化値 = 部位の総視線値 / 部位の頂点数
```

## 分析例と解釈

### 例 1: 顔中心の視線パターン

```
head: 52.0%          # 頭部が最大の注視対象
neck: 25.6%          # 首も強く注視
spine2: 22.2%        # 上胸部に一定の注視
leftShoulder: 0.2%   # 肩への軽微な注視
```

**解釈**: 典型的な対人注視パターン。顔・首領域に集中した視線分布。

### 例 2: 動作注視パターン

```
leftHand: 35.0%      # 左手への強い注視
rightHand: 28.0%     # 右手への注視
head: 20.0%          # 顔への基本注視
leftArm: 12.0%       # 腕の動きへの注視
```

**解釈**: 手の動作や操作に注目した視線パターン。

## パフォーマンスと制限事項

### パフォーマンス

- **処理速度**: 100 フレーム/秒程度（標準的な PC）
- **メモリ使用量**: フレーム数 × 6890 × 8 バイト + 分析結果
- **推奨フレーム数**: 1000 フレーム以下（メモリ効率のため）

### 制限事項

1. **頂点数固定**: SMPL モデルの 6890 頂点に固定
2. **部位分類固定**: 24 部位の事前定義分類のみ
3. **フレーム順序**: ファイル名の数値順でのみ処理
4. **視線値非負**: 負の視線値は想定外

### エラーハンドリング

```python
try:
    analyzer.run_analysis(data_folder, output_dir)
except FileNotFoundError:
    print("データフォルダが見つかりません")
except ValueError as e:
    print(f"データ形式エラー: {e}")
except Exception as e:
    print(f"予期しないエラー: {e}")
```

## 技術詳細

### アルゴリズム

1. **データ読み込み**

   ```python
   # .npyファイルの読み込み
   data = np.load(file_path)
   # 形状チェック: (6890,) または (6890,1)
   if data.shape[0] != 6890:
       raise ValueError("Invalid vertex count")
   ```

2. **部位マッピング**

   ```python
   # セグメンテーションデータから頂点インデックス取得
   vertex_indices = segmentation_data[part_name]
   # 該当頂点の視線値を合計
   part_gaze = np.sum(gaze_data[vertex_indices])
   ```

3. **確率計算**
   ```python
   # 全部位の視線値合計
   total_gaze = sum(part_totals.values())
   # 各部位の確率
   probabilities = {part: total/total_gaze for part, total in part_totals.items()}
   ```

### 依存ライブラリ

```python
numpy>=1.21.0      # 数値計算
pandas>=1.3.0      # データ処理・CSV出力
matplotlib>=3.4.0  # グラフ可視化
requests>=2.25.0   # セグメンテーションファイルダウンロード
```

### ファイル構造

```
smplmodel_sample/
├── src/
│   └── gaze_part_analyzer.py      # メイン分析クラス
├── utils/
│   └── gaze_part_viewer.py        # CLI実行スクリプト
├── data/
│   └── *.npy                      # 視線データファイル
├── output/
│   ├── part_gaze_analysis.csv     # 部位別結果
│   ├── frame_gaze_analysis.csv    # フレーム別結果（時間情報付き）
│   ├── gaze_analysis_summary.json # 統計サマリー
│   ├── gaze_changes.json          # 視線変化分析結果
│   └── gaze_part_analysis.png     # 可視化グラフ
└── smpl_vert_segmentation.json    # 部位セグメンテーション
```

## トラブルシューティング

### よくある問題

1. **ModuleNotFoundError: No module named 'pandas'**

   ```bash
   uv add pandas matplotlib
   ```

2. **ValueError: Invalid vertex count**

   - 入力データの形状を確認
   - SMPL モデル以外のメッシュデータの可能性

3. **日本語フォントの警告**

   - 可視化での日本語表示警告（機能には影響なし）
   - `--no-visualization`オプションで回避可能

4. **Memory Error (大量フレーム)**
   - フレーム数を分割して処理
   - `--threshold`オプションで低視線値除外

### デバッグオプション

```bash
# 詳細ログでの実行
python utils/gaze_part_viewer.py data/ --verbose

# Python内でのデバッグ
import logging
logging.basicConfig(level=logging.DEBUG)
analyzer = GazePartAnalyzer()
```

## 新機能: 時間軸分析と視線変化検出 (v2.0)

### 時間変換機能

90Hzサンプリングレートに対応した高精度な時間変換機能を提供：

#### 主要機能

1. **フレーム→秒数変換**
   ```python
   timestamp = analyzer.get_timestamp_from_frame(270)  # 3.0秒
   ```

2. **時:分:秒形式での表示**
   ```python
   time_str = analyzer.get_time_formatted(270)  # "0:03.000"
   ```

3. **フレームレート設定**
   ```python
   # 90Hz（デフォルト）
   analyzer = GazePartAnalyzer(frame_rate=90.0)
   
   # その他のフレームレート
   analyzer = GazePartAnalyzer(frame_rate=30.0)   # 30fps
   analyzer = GazePartAnalyzer(frame_rate=60.0)   # 60fps
   analyzer = GazePartAnalyzer(frame_rate=120.0)  # 120fps
   ```

### 視線変化検出機能

部位間の視線移動を自動検出し、時系列で記録：

#### 検出アルゴリズム

1. **変化の定義**: フレーム間で最大注視部位が変化
2. **閾値設定**: 新しい部位への視線集中度が設定値以上
3. **時間記録**: 変化発生時刻を秒数と時:分:秒で記録

#### 使用例

```python
# 視線変化検出（閾値10%）
changes = analyzer.detect_gaze_changes(min_change_threshold=0.1)

# 結果の例
for change in changes:
    print(f"{change['description']}")
    # 出力: "3.2秒: headからrightHandに移動"
```

#### 出力例

```json
{
  "timestamp": 3.200,
  "time_formatted": "0:03.200",
  "frame": 288,
  "from_part": "head",
  "to_part": "rightHand", 
  "confidence": 0.456,
  "description": "3.2秒: headからrightHandに移動"
}
```

### 拡張された出力ファイル

#### 1. frame_gaze_analysis.csv の拡張

新しく追加された列：
- `timestamp`: 秒数での時間
- `time_formatted`: 時:分:秒形式の時間

#### 2. gaze_changes.json の新規追加

視線変化の詳細ログ：
- 変化発生時刻
- 変化前後の部位
- 視線集中度
- 日本語での変化説明

### 分析結果の表示拡張

コンソール出力に以下の情報が追加：

```
総フレーム数: 150
分析対象部位数: 24
総再生時間: 0:01.667 (1.667秒)
フレームレート: 90Hz

【フレーム別詳細 (最初の10フレーム)】
フレーム 000076 (0:00.844): head           (52.3%)
フレーム 000077 (0:00.856): head           (48.1%)

【視線変化検出 (最初の10件)】
 1. 3.2秒: headからrightHandに移動
 2. 5.7秒: rightHandからleftArmに移動
総変化回数: 15回
```

### パフォーマンス最適化

- **計算効率**: 時間変換はリアルタイム計算（キャッシュ不要）
- **メモリ効率**: 既存データ構造に時間情報を統合
- **処理速度**: 変化検出は線形時間O(n)で実行

## 今後の拡張可能性

### 機能拡張案

1. ~~**時系列分析**: 視線の時間的変化パターンの分析~~ ✅ **実装済み**
2. **クラスタリング**: 類似の視線パターンのグループ化
3. **ヒートマップ**: 3D メッシュ上での視線密度可視化
4. **統計検定**: 部位間視線差の有意性検定
5. **機械学習**: 視線パターンの自動分類
6. **変化パターン分析**: 典型的な視線移動シーケンスの抽出

### カスタマイズ

```python
# カスタム部位定義
custom_segmentation = {
    "upper_body": head_vertices + neck_vertices + spine2_vertices,
    "lower_body": leg_vertices + foot_vertices
}

# カスタム分析指標
def custom_metric(gaze_data, part_vertices):
    return np.std(gaze_data[part_vertices])  # 視線のばらつき
```

## 変更履歴

### v2.0 (2025-07-24)

**新機能**:
- ✅ 90Hzサンプリングレート対応の時間変換機能
- ✅ フレーム番号から秒数・時:分:秒形式への変換
- ✅ 視線変化検出機能（部位間移動の自動検出）
- ✅ JSON形式での視線変化ログ出力
- ✅ CSV出力への時間情報列追加
- ✅ コンソール表示の時間情報拡張

**技術仕様**:
- `GazePartAnalyzer`クラスに`frame_rate`パラメータ追加
- `get_timestamp_from_frame()`メソッド追加
- `get_time_formatted()`メソッド追加
- `detect_gaze_changes()`メソッド追加
- `gaze_changes.json`ファイル出力追加

**出力ファイルの変更**:
- `frame_gaze_analysis.csv`: `timestamp`, `time_formatted`列を追加
- `gaze_changes.json`: 新規ファイル追加

### v1.0 (初期リリース)

**基本機能**:
- 視線データの部位別分析
- CSV/JSON形式での結果出力
- 可視化グラフ生成
- SMPL 24部位対応
- フレーム別最大注視部位分析

## 参考資料

- [SMPL 公式サイト](https://smpl.is.tue.mpg.de/)
- [Meshcapade SMPL セグメンテーション](https://github.com/Meshcapade/wiki)
- [NumPy 公式ドキュメント](https://numpy.org/doc/)
- [pandas 公式ドキュメント](https://pandas.pydata.org/docs/)

---

本システムは研究・教育目的での使用を想定しています。商用利用の場合は適切なライセンス確認を行ってください。
