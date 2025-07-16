#!/usr/bin/env python3
"""
T-pose視線メッシュ可視化ツール

T-poseのOBJファイルに視線データを適用して、カラーマップで可視化するコマンドラインツールです。
"""

import argparse
import sys
from pathlib import Path

# プロジェクトのsrcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gaze_mesh_visualizer import GazeMeshVisualizer


def main() -> None:
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="T-poseメッシュに視線データを適用して可視化します",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    # 基本実行
    python utils/gaze_mesh_viewer.py tpose.obj data/
    
    # 出力先とカラーマップ指定
    python utils/gaze_mesh_viewer.py tpose.obj data/ --output results/ --colormap viridis
    
    # 特定フレームのみ処理
    python utils/gaze_mesh_viewer.py tpose.obj data/ --frame-range 0 100
    
    # 画像出力なし（メッシュファイルのみ）
    python utils/gaze_mesh_viewer.py tpose.obj data/ --no-images
    
    # フレーム正規化（フレーム内での最小・最大値で正規化）
    python utils/gaze_mesh_viewer.py tpose.obj data/ --normalization frame

利用可能なカラーマップ:
    hot, viridis, plasma, inferno, magma, cool, spring, summer, autumn, winter
        """
    )
    
    parser.add_argument(
        "obj_path",
        type=str,
        help="T-poseのOBJファイルパス"
    )
    
    parser.add_argument(
        "data_folder",
        type=str,
        help="視線データ(.npyファイル)が格納されたフォルダパス"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output/gaze_meshes",
        help="出力ディレクトリパス (デフォルト: output/gaze_meshes)"
    )
    
    parser.add_argument(
        "--colormap", "-c",
        type=str,
        default="hot",
        choices=["hot", "viridis", "plasma", "inferno", "magma", "cool", 
                "spring", "summer", "autumn", "winter", "jet", "rainbow"],
        help="カラーマップ名 (デフォルト: hot)"
    )
    
    parser.add_argument(
        "--normalization", "-n",
        type=str,
        default="global",
        choices=["global", "frame", "percentile"],
        help="正規化方法 (デフォルト: global)"
    )
    
    parser.add_argument(
        "--frame-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="処理するフレーム範囲 (例: --frame-range 0 100)"
    )
    
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="obj",
        choices=["obj", "ply", "stl"],
        help="出力メッシュファイル形式 (デフォルト: obj)"
    )
    
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="画像出力をスキップ（メッシュファイルのみ生成）"
    )
    
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="カラーバー凡例の生成をスキップ"
    )
    
    parser.add_argument(
        "--segmentation", "-s",
        type=str,
        default="smpl_vert_segmentation.json",
        help="SMPLセグメンテーションファイルパス (デフォルト: smpl_vert_segmentation.json)"
    )
    
    parser.add_argument(
        "--single-frame",
        type=int,
        help="単一フレームのみ処理（フレーム番号指定）"
    )
    
    parser.add_argument(
        "--view-angle",
        type=float,
        nargs=2,
        default=[45, 45],
        metavar=("ELEVATION", "AZIMUTH"),
        help="画像の視点角度 (デフォルト: 45 45)"
    )
    
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[800, 600],
        metavar=("WIDTH", "HEIGHT"),
        help="画像解像度 (デフォルト: 800 600)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="詳細な実行ログを表示"
    )
    
    args = parser.parse_args()
    
    # パスの検証
    obj_path = Path(args.obj_path)
    if not obj_path.exists():
        print(f"エラー: OBJファイルが存在しません: {obj_path}")
        sys.exit(1)
    
    data_folder = Path(args.data_folder)
    if not data_folder.exists():
        print(f"エラー: データフォルダが存在しません: {data_folder}")
        sys.exit(1)
    
    if not data_folder.is_dir():
        print(f"エラー: 指定されたパスはディレクトリではありません: {data_folder}")
        sys.exit(1)
    
    output_dir = Path(args.output)
    
    try:
        # 可視化器の初期化
        visualizer = GazeMeshVisualizer(
            obj_path=obj_path,
            segmentation_file=args.segmentation
        )
        
        if args.verbose:
            print(f"可視化設定:")
            print(f"  T-poseメッシュ: {obj_path.absolute()}")
            print(f"  視線データフォルダ: {data_folder.absolute()}")
            print(f"  出力ディレクトリ: {output_dir.absolute()}")
            print(f"  カラーマップ: {args.colormap}")
            print(f"  正規化方法: {args.normalization}")
            print(f"  出力形式: {args.format}")
            print(f"  フレーム範囲: {args.frame_range}")
            print(f"  画像出力: {'無効' if args.no_images else '有効'}")
            print(f"  凡例作成: {'無効' if args.no_legend else '有効'}")
            print()
        
        # ベースメッシュの読み込み
        print("T-poseメッシュを読み込み中...")
        visualizer.load_base_mesh()
        
        # 視線データの読み込み
        print("視線データを読み込み中...")
        visualizer.load_gaze_data(data_folder)
        
        # 統計情報の表示
        stats = visualizer.get_gaze_statistics()
        print(f"\n視線データ統計:")
        print(f"  総フレーム数: {stats['total_frames']}")
        print(f"  フレーム範囲: {stats['frame_range']['start']} - {stats['frame_range']['end']}")
        print(f"  視線値範囲: {stats['gaze_range']['min']:.3f} - {stats['gaze_range']['max']:.3f}")
        print(f"  平均視線値: {stats['gaze_range']['mean']:.3f}")
        print(f"  非ゼロ視線比率: {stats['non_zero_ratio']:.1%}")
        
        # 単一フレーム処理の場合
        if args.single_frame is not None:
            frame_num = args.single_frame
            print(f"\nフレーム {frame_num} の視線メッシュを生成中...")
            
            # メッシュファイルの保存
            mesh_path = output_dir / f"gaze_frame_{frame_num:06d}.{args.format}"
            visualizer.save_gaze_mesh(
                frame_num=frame_num,
                output_path=mesh_path,
                normalization=args.normalization,
                colormap=args.colormap
            )
            
            # 画像の保存（オプション）
            if not args.no_images:
                image_path = output_dir / f"gaze_frame_{frame_num:06d}.png"
                visualizer.create_gaze_visualization(
                    frame_num=frame_num,
                    save_path=image_path,
                    view_angle=tuple(args.view_angle),
                    resolution=tuple(args.resolution),
                    colormap=args.colormap
                )
            
            print(f"フレーム {frame_num} の処理が完了しました!")
            
        else:
            # 複数フレーム処理
            print("\n視線メッシュシーケンスを生成中...")
            
            # フレーム範囲の設定
            if args.frame_range:
                frame_range = tuple(args.frame_range)
                print(f"フレーム範囲: {frame_range[0]} - {frame_range[1]}")
            else:
                frame_range = None
                print("全フレームを処理")
            
            # メッシュファイルの一括生成
            mesh_files = visualizer.create_frame_sequence(
                output_dir=output_dir / "meshes",
                frame_range=frame_range,
                normalization=args.normalization,
                colormap=args.colormap,
                file_format=args.format
            )
            
            # 画像の生成（オプション）
            if not args.no_images:
                print("\n視線可視化画像を生成中...")
                image_dir = output_dir / "images"
                
                # 処理するフレームを決定
                if frame_range:
                    start, end = frame_range
                    frames_to_process = [f for f in sorted(visualizer.gaze_data.keys()) 
                                       if start <= f <= end]
                else:
                    frames_to_process = sorted(visualizer.gaze_data.keys())
                
                # 最初の10フレーム or 全フレーム（少ない方）
                frames_to_process = frames_to_process[:min(10, len(frames_to_process))]
                
                for frame_num in frames_to_process:
                    image_path = image_dir / f"gaze_frame_{frame_num:06d}.png"
                    try:
                        visualizer.create_gaze_visualization(
                            frame_num=frame_num,
                            save_path=image_path,
                            view_angle=tuple(args.view_angle),
                            resolution=tuple(args.resolution),
                            colormap=args.colormap
                        )
                    except Exception as e:
                        print(f"画像生成エラー (フレーム {frame_num}): {e}")
                        if args.verbose:
                            import traceback
                            traceback.print_exc()
            
            # カラーバー凡例の生成（オプション）
            if not args.no_legend:
                print("\nカラーバー凡例を生成中...")
                legend_path = output_dir / "colorbar_legend.png"
                visualizer.create_colorbar_legend(
                    save_path=legend_path,
                    colormap=args.colormap,
                    title="視線強度"
                )
            
            print(f"\n処理が正常に完了しました!")
            print(f"生成されたメッシュファイル: {len(mesh_files)}個")
            print(f"出力ディレクトリ: {output_dir.absolute()}")
        
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