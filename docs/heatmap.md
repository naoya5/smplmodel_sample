作成したファイル

1. src/gaze_heatmap_visualizer.py -
   コア可視化クラス
2. create_gaze_heatmap.py -
   メイン実行スクリプト

主要機能

- T-pose メッシュ生成:
  SMPL モデルから T-pose を作成
- 視線データ累積:
  全フレームの視線データを頂点レベルで合計
- ヒートマップ適用:
  色の強度で視線集中度を表現
- 複数視点出力:
  正面・背面・左右側面の画像を生成
- 3D メッシュ保存:
  OBJ 形式でヒートマップ付きメッシュを出力

使用方法

# 基本的な使用

uv run create_gaze_heatmap.py results/User11/
models/SMPL_NEUTRAL.pkl

# カスタム設定

uv run create_gaze_heatmap.py results/User11/
smpl/SMPL_NEUTRAL.pkl \
 -o output/heatmap --colormap cool
--views front back left right

出力ファイル

- gaze_heatmap_front.png - 正面視点画像
- gaze_heatmap_back.png - 背面視点画像
- gaze_heatmap_left.png - 左側面画像
- gaze_heatmap_right.png - 右側面画像
- gaze_heatmap_tpose.obj - 3D メッシュファイル
- heatmap_colorbar.png - カラースケール凡例
- heatmap_statistics.json - 統計情報

これで T-pose に全フレームの視線を重ねたヒート
マップ可視化が可能になります。
