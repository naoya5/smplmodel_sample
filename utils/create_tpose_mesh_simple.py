#!/usr/bin/env python3
"""
T-poseメッシュ生成ツール（簡易版）

chumpy依存を回避してSMPLモデルからT-poseメッシュを生成するツールです。
"""

import argparse
import sys
from pathlib import Path
import pickle
import numpy as np
import trimesh
import torch

# プロジェクトのsrcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_smpl_model_simple(model_path: Path) -> dict:
    """
    SMPLモデルファイルを読み込み（chumpy依存なし）
    
    Args:
        model_path: SMPLモデルファイルパス
        
    Returns:
        SMPLモデルデータ
    """
    print(f"SMPLモデルを読み込み中: {model_path}")
    
    # モデル情報を保存する辞書
    model_data = {}
    
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        # chumpy配列を numpy配列に変換
        for key, value in data.items():
            if hasattr(value, 'r'):  # chumpy配列の場合
                model_data[key] = np.array(value.r)
            else:
                model_data[key] = value
                
        print(f"モデル読み込み完了")
        print(f"利用可能なキー: {list(model_data.keys())}")
        
        return model_data
        
    except Exception as e:
        print(f"モデル読み込みエラー: {e}")
        raise


def create_tpose_mesh_simple(smpl_model_path: Path, 
                           output_path: Path) -> Path:
    """
    SMPLモデルからT-poseメッシュを生成（簡易版）
    
    Args:
        smpl_model_path: SMPLモデルファイルパス
        output_path: 出力OBJファイルパス
        
    Returns:
        保存されたファイルパス
    """
    print(f"T-poseメッシュを生成中...")
    print(f"SMPLモデル: {smpl_model_path}")
    
    # SMPLモデルの読み込み
    model_data = load_smpl_model_simple(smpl_model_path)
    
    # T-poseの頂点位置を取得（template vertices）
    if 'v_template' in model_data:
        vertices = model_data['v_template']
    elif 'vertices' in model_data:
        vertices = model_data['vertices']
    else:
        raise ValueError("テンプレート頂点が見つかりません")
    
    # 面の情報を取得
    if 'f' in model_data:
        faces = model_data['f']
    elif 'faces' in model_data:
        faces = model_data['faces']
    else:
        raise ValueError("面の情報が見つかりません")
    
    print(f"メッシュ情報: 頂点数={len(vertices)}, 面数={len(faces)}")
    
    # trimeshメッシュの作成
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # OBJファイルとして保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)
    
    print(f"T-poseメッシュを保存: {output_path}")
    print(f"頂点数: {len(vertices)}")
    print(f"面数: {len(faces)}")
    
    return output_path


def main() -> None:
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="SMPLモデルからT-poseメッシュを生成します（簡易版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    # 基本実行
    python utils/create_tpose_mesh_simple.py smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
    
    # 出力先指定
    python utils/create_tpose_mesh_simple.py smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl --output tpose_mesh.obj
        """
    )
    
    parser.add_argument(
        "smpl_model_path",
        type=str,
        help="SMPLモデルファイルパス (.pkl)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="tpose_mesh.obj",
        help="出力OBJファイルパス (デフォルト: tpose_mesh.obj)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="詳細な実行ログを表示"
    )
    
    args = parser.parse_args()
    
    # パスの検証
    smpl_model_path = Path(args.smpl_model_path)
    if not smpl_model_path.exists():
        print(f"エラー: SMPLモデルファイルが存在しません: {smpl_model_path}")
        sys.exit(1)
    
    output_path = Path(args.output)
    
    try:
        if args.verbose:
            print(f"設定:")
            print(f"  SMPLモデル: {smpl_model_path.absolute()}")
            print(f"  出力ファイル: {output_path.absolute()}")
            print()
        
        # T-poseメッシュの生成
        result_path = create_tpose_mesh_simple(
            smpl_model_path=smpl_model_path,
            output_path=output_path
        )
        
        print(f"\nT-poseメッシュの生成が完了しました!")
        print(f"保存先: {result_path.absolute()}")
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()