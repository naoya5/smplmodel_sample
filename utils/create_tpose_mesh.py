#!/usr/bin/env python3
"""
T-poseメッシュ生成ツール

SMPLモデルからT-pose（デフォルトポーズ）のメッシュを生成するツールです。
"""

import argparse
import sys
from pathlib import Path

# プロジェクトのsrcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
import trimesh

try:
    from smplx import SMPL
except ImportError:
    print("エラー: smplxライブラリがインストールされていません")
    print("インストール: uv add smplx")
    sys.exit(1)


def create_tpose_mesh(smpl_model_path: Path, 
                     output_path: Path,
                     gender: str = "neutral") -> Path:
    """
    SMPLモデルからT-poseメッシュを生成
    
    Args:
        smpl_model_path: SMPLモデルファイルパス
        output_path: 出力OBJファイルパス
        gender: 性別 ("neutral", "male", "female")
        
    Returns:
        保存されたファイルパス
    """
    print(f"T-poseメッシュを生成中...")
    print(f"SMPLモデル: {smpl_model_path}")
    print(f"性別: {gender}")
    
    # SMPLモデルの読み込み
    # smplxライブラリ用に適切な形式でモデルを読み込み
    try:
        # 既存のファイルを直接使用
        smpl_model = SMPL(model_path=str(smpl_model_path.parent), gender=gender)
    except Exception as e:
        # ファイル名を変更して再試行
        print(f"標準的な読み込みに失敗: {e}")
        print("代替手段を試行中...")
        
        # ファイルを期待される名前でコピー
        expected_name = f"SMPL_{gender.upper()}.pkl"
        expected_path = smpl_model_path.parent / expected_name
        
        if not expected_path.exists():
            import shutil
            shutil.copy2(smpl_model_path, expected_path)
            print(f"ファイルをコピー: {smpl_model_path} -> {expected_path}")
        
        smpl_model = SMPL(model_path=str(smpl_model_path.parent), gender=gender)
    
    # T-pose（すべてのパラメータをゼロに設定）
    batch_size = 1
    global_orient = torch.zeros(batch_size, 3, dtype=torch.float32)  # ルート回転
    body_pose = torch.zeros(batch_size, 69, dtype=torch.float32)     # 関節回転 (23関節 × 3)
    betas = torch.zeros(batch_size, 10, dtype=torch.float32)         # 体型パラメータ
    
    # メッシュ生成
    with torch.no_grad():
        output = smpl_model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas
        )
    
    # 頂点と面の取得
    vertices = output.vertices[0].cpu().numpy()
    faces = smpl_model.faces
    
    print(f"メッシュ情報: 頂点数={len(vertices)}, 面数={len(faces)}")
    
    # trimeshメッシュの作成
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # メッシュの正規化（オプション）
    # mesh.apply_translation(-mesh.centroid)  # 重心を原点に
    # mesh.apply_scale(1.0 / mesh.scale)      # スケール正規化
    
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
        description="SMPLモデルからT-poseメッシュを生成します",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    # 基本実行
    python utils/create_tpose_mesh.py SMPL_NEUTRAL.npz
    
    # 出力先指定
    python utils/create_tpose_mesh.py SMPL_NEUTRAL.npz --output tpose_mesh.obj
    
    # 性別指定
    python utils/create_tpose_mesh.py SMPL_MALE.npz --gender male
        """
    )
    
    parser.add_argument(
        "smpl_model_path",
        type=str,
        help="SMPLモデルファイルパス (.npz または .pkl)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="tpose_mesh.obj",
        help="出力OBJファイルパス (デフォルト: tpose_mesh.obj)"
    )
    
    parser.add_argument(
        "--gender", "-g",
        type=str,
        default="neutral",
        choices=["neutral", "male", "female"],
        help="性別 (デフォルト: neutral)"
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
            print(f"  性別: {args.gender}")
            print()
        
        # T-poseメッシュの生成
        result_path = create_tpose_mesh(
            smpl_model_path=smpl_model_path,
            output_path=output_path,
            gender=args.gender
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