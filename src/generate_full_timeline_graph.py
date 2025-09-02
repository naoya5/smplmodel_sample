#!/usr/bin/env python3
"""
全フレーム版の視線データ部位別分析グラフを生成するスクリプト
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
    
    print("全フレーム版視線データ部位別分析を開始...")
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
        
        # 4. 全フレーム版グラフを生成
        print("4. 全フレーム版時系列グラフを生成中...")
        graph_path = analyzer.visualize_full_timeline(output_dir)
        
        # 総フレーム数と時間を表示
        total_frames = len(analyzer.frame_results)
        total_time = total_frames / analyzer.frame_rate
        
        print(f"\n✓ 全フレーム版グラフを生成しました: {graph_path}")
        print(f"総フレーム数: {total_frames}")
        print(f"総時間: {total_time:.1f}秒")
        print(f"フレームレート: {analyzer.frame_rate}Hz")
        print("\n分析が正常に完了しました!")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    main()