#!/usr/bin/env python3
"""
視線データ部位別分析ツール

指定フォルダ内の.npyファイル（視線データ）を読み込み、SMPLモデルの各部位への
視線集中度を計算・分析するコマンドラインツールです。
"""

import argparse
import sys
from pathlib import Path

# プロジェクトのsrcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gaze_part_analyzer import GazePartAnalyzer


def main() -> None:
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="視線データの部位別分析を実行します",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    # 基本実行
    python utils/gaze_part_viewer.py data/
    
    # 出力先指定
    python utils/gaze_part_viewer.py data/ --output results/
    
    # 閾値指定（低い視線値を除外）
    python utils/gaze_part_viewer.py data/ --threshold 0.1
    
    # セグメンテーションファイル指定
    python utils/gaze_part_viewer.py data/ --segmentation my_segmentation.json
        """
    )
    
    parser.add_argument(
        "data_folder",
        type=str,
        help="視線データ(.npyファイル)が格納されたフォルダパス"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output",
        help="出力ディレクトリパス (デフォルト: output)"
    )
    
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.0,
        help="視線値の閾値（この値未満の視線データは除外）(デフォルト: 0.0)"
    )
    
    parser.add_argument(
        "--segmentation", "-s",
        type=str,
        default="smpl_vert_segmentation.json",
        help="SMPLセグメンテーションファイルパス (デフォルト: smpl_vert_segmentation.json)"
    )
    
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="可視化グラフの生成をスキップ"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="詳細な実行ログを表示"
    )
    
    args = parser.parse_args()
    
    # パスの検証
    data_folder = Path(args.data_folder)
    if not data_folder.exists():
        print(f"エラー: データフォルダが存在しません: {data_folder}")
        sys.exit(1)
    
    if not data_folder.is_dir():
        print(f"エラー: 指定されたパスはディレクトリではありません: {data_folder}")
        sys.exit(1)
    
    output_dir = Path(args.output)
    
    try:
        # 分析器の初期化
        analyzer = GazePartAnalyzer(segmentation_file=args.segmentation)
        
        if args.verbose:
            print(f"分析設定:")
            print(f"  データフォルダ: {data_folder.absolute()}")
            print(f"  出力ディレクトリ: {output_dir.absolute()}")
            print(f"  視線値閾値: {args.threshold}")
            print(f"  セグメンテーションファイル: {args.segmentation}")
            print(f"  可視化: {'無効' if args.no_visualization else '有効'}")
            print()
        
        # データ読み込み
        print("視線データを読み込み中...")
        gaze_data = analyzer.load_gaze_data(data_folder)
        
        # 閾値適用（オプション）
        if args.threshold > 0.0:
            print(f"閾値 {args.threshold} 未満の視線値を除外中...")
            filtered_count = 0
            for frame_num, data in gaze_data.items():
                original_nonzero = np.count_nonzero(data)
                data[data < args.threshold] = 0.0
                filtered_nonzero = np.count_nonzero(data)
                filtered_count += original_nonzero - filtered_nonzero
                analyzer.gaze_data[frame_num] = data
            
            if args.verbose:
                print(f"  {filtered_count} 個の視線値を除外しました")
        
        # 部位別分析実行
        print("部位別視線分析を実行中...")
        analyzer.map_gaze_to_parts()
        
        print("フレーム別分析を実行中...")
        analyzer.analyze_frame_attention()
        
        # 結果表示
        analyzer.print_results()
        
        # ファイル出力
        print("\n結果をファイルに出力中...")
        output_files = analyzer.export_results(output_dir)
        
        # 可視化（オプション）
        if not args.no_visualization:
            print("可視化グラフを生成中...")
            try:
                graph_path = analyzer.visualize_results(output_dir)
                print(f"可視化グラフ: {graph_path}")
            except Exception as e:
                print(f"警告: 可視化でエラーが発生しました: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
        
        print(f"\n分析が正常に完了しました!")
        print(f"結果は {output_dir.absolute()} に保存されました。")
        
    except KeyboardInterrupt:
        print("\n\n処理が中断されました。")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()