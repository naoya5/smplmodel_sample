#!/usr/bin/env python3
"""
T-poseメッシュ生成ツール（chumpy回避版）

chumpy依存を完全に回避してSMPLモデルからT-poseメッシュを生成するツールです。
"""

import argparse
import sys
from pathlib import Path
import pickle
import numpy as np
import trimesh

# プロジェクトのsrcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# chumpy スタブを sys.modules に追加
sys.path.insert(0, str(Path(__file__).parent))
import chumpy_stub
sys.modules['chumpy'] = chumpy_stub


def convert_chumpy_to_numpy(obj):
    """
    chumpy オブジェクトを numpy 配列に再帰的に変換
    
    Args:
        obj: 変換対象のオブジェクト
        
    Returns:
        numpy配列またはその他のオブジェクト
    """
    if hasattr(obj, 'r'):  # chumpy配列
        return np.array(obj.r)
    elif hasattr(obj, '__dict__'):  # オブジェクト
        result = {}
        for key, value in obj.__dict__.items():
            result[key] = convert_chumpy_to_numpy(value)
        return result
    elif isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            result[key] = convert_chumpy_to_numpy(value)
        return result
    elif isinstance(obj, (list, tuple)):
        return [convert_chumpy_to_numpy(item) for item in obj]
    else:
        return obj


def load_smpl_model_nochumpy(model_path: Path) -> dict:
    """
    SMPLモデルファイルを読み込み（chumpy回避）
    
    Args:
        model_path: SMPLモデルファイルパス
        
    Returns:
        SMPLモデルデータ
    """
    print(f"SMPLモデルを読み込み中: {model_path}")
    
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        # chumpy配列を numpy配列に変換
        model_data = convert_chumpy_to_numpy(data)
                
        print(f"モデル読み込み完了")
        print(f"利用可能なキー: {list(model_data.keys())}")
        
        return model_data
        
    except Exception as e:
        print(f"モデル読み込みエラー: {e}")
        raise


def create_tpose_mesh_nochumpy(smpl_model_path: Path, 
                             output_path: Path) -> Path:
    """
    SMPLモデルからT-poseメッシュを生成（chumpy回避版）
    
    Args:
        smpl_model_path: SMPLモデルファイルパス
        output_path: 出力OBJファイルパス
        
    Returns:
        保存されたファイルパス
    """
    print(f"T-poseメッシュを生成中...")
    print(f"SMPLモデル: {smpl_model_path}")
    
    # SMPLモデルの読み込み
    model_data = load_smpl_model_nochumpy(smpl_model_path)
    
    # T-poseの頂点位置を取得（template vertices）
    vertices = None
    for key in ['v_template', 'vertices', 'v_posed']:
        if key in model_data:
            vertices = model_data[key]
            print(f"頂点データを取得: {key}")
            break
    
    if vertices is None:
        print("利用可能なキー:", list(model_data.keys()))
        raise ValueError("テンプレート頂点が見つかりません")
    
    # 面の情報を取得
    faces = None
    for key in ['f', 'faces']:
        if key in model_data:
            faces = model_data[key]
            print(f"面データを取得: {key}")
            break
    
    if faces is None:
        print("利用可能なキー:", list(model_data.keys()))
        raise ValueError("面の情報が見つかりません")
    
    # numpy配列に変換
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    print(f"メッシュ情報: 頂点数={len(vertices)}, 面数={len(faces)}")
    print(f"頂点の形状: {vertices.shape}")
    print(f"面の形状: {faces.shape}")
    
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
        description="SMPLモデルからT-poseメッシュを生成します（chumpy回避版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    # 基本実行
    python utils/create_tpose_mesh_nochumpy.py smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
    
    # 出力先指定
    python utils/create_tpose_mesh_nochumpy.py smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl --output tpose_mesh.obj
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
        result_path = create_tpose_mesh_nochumpy(
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