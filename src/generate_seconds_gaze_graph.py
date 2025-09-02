#!/usr/bin/env python3
"""
視線データ部位別分析の秒数版グラフを生成するスクリプト
"""

from pathlib import Path
from gaze_part_analyzer import GazePartAnalyzer


def main():
    """メイン関数"""
    # データフォルダの設定
    data_folder = Path("/Users/naoya/dev/research-university/smplmodel_sample/data/User11")
    output_dir = Path("/Users/naoya/dev/research-university/smplmodel_sample/results/User11")
    
    # 分析器の初期化（フレームレートを90Hzに設定）
    analyzer = GazePartAnalyzer(frame_rate=90.0)
    
    print("視線データ部位別分析（秒数版）を開始...")
    print(f"データフォルダ: {data_folder}")
    print(f"出力ディレクトリ: {output_dir}")
    print("-" * 50)
    
    try:
        # 1. データ読み込み
        print("1. 視線データを読み込み中...")
        analyzer.load_gaze_data(data_folder)
        
        # 2. 部位別集計
        print("2. 部位別視線データを集計中...")
        analyzer.map_gaze_to_parts()
        
        # 3. フレーム別分析
        print("3. フレーム別分析を実行中...")
        analyzer.analyze_frame_attention()
        
        # 4. 秒数版グラフを生成
        print("4. 秒数版グラフを生成中...")
        graph_path = analyzer.visualize_results_with_seconds(output_dir)
        
        print(f"\n✓ 秒数版グラフを生成しました: {graph_path}")
        print("\n分析が正常に完了しました!")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    main()