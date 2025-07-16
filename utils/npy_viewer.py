#!/usr/bin/env python3
"""
NPYファイル読み込み・表示ユーティリティ

npyファイルを読み込んで内容を表示するためのスクリプトです。
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


def load_and_display_npy(file_path: Path) -> None:
    """
    npyファイルを読み込んで内容を表示します。
    
    Args:
        file_path: npyファイルのパス
    """
    try:
        # npyファイルを読み込み
        data = np.load(file_path)
        
        print(f"ファイル: {file_path}")
        print(f"データ型: {type(data)}")
        print(f"形状: {data.shape if hasattr(data, 'shape') else 'N/A'}")
        print(f"dtype: {data.dtype if hasattr(data, 'dtype') else 'N/A'}")
        print(f"サイズ: {data.size if hasattr(data, 'size') else 'N/A'}")
        
        if hasattr(data, 'ndim'):
            print(f"次元数: {data.ndim}")
        
        print("\n--- データ内容 ---")
        
        # すべてのデータを表示
        print(data)
            
        # 統計情報（数値データの場合）
        if hasattr(data, 'dtype') and np.issubdtype(data.dtype, np.number):
            print("\n--- 統計情報 ---")
            print(f"最小値: {np.min(data)}")
            print(f"最大値: {np.max(data)}")
            print(f"平均値: {np.mean(data):.6f}")
            print(f"標準偏差: {np.std(data):.6f}")
            
    except FileNotFoundError:
        print(f"エラー: ファイル '{file_path}' が見つかりません。")
        sys.exit(1)
    except Exception as e:
        print(f"エラー: ファイルの読み込み中にエラーが発生しました: {e}")
        sys.exit(1)


def main() -> None:
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="npyファイルを読み込んで内容を表示します",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    python npy_viewer.py data.npy
    python npy_viewer.py /path/to/array.npy
        """
    )
    
    parser.add_argument(
        "file_path",
        type=str,
        help="読み込むnpyファイルのパス"
    )
    
    args = parser.parse_args()
    
    file_path = Path(args.file_path)
    
    if not file_path.exists():
        print(f"エラー: ファイル '{file_path}' が存在しません。")
        sys.exit(1)
    
    if not file_path.suffix.lower() == '.npy':
        print(f"警告: ファイル '{file_path}' はnpyファイルではない可能性があります。")
    
    load_and_display_npy(file_path)


if __name__ == "__main__":
    main()