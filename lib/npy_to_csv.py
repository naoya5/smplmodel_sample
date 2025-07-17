"""
.npyファイルをCSVに変換するユーティリティライブラリ

このモジュールは.npyファイルを読み込み、CSV形式で出力する機能を提供します。
"""

from typing import Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path


def npy_to_csv(
    npy_path: Union[str, Path],
    csv_path: Optional[Union[str, Path]] = None,
    header: Optional[bool] = True,
    index: Optional[bool] = False,
    column_names: Optional[list[str]] = None
) -> Path:
    """
    .npyファイルをCSVファイルに変換します。

    Args:
        npy_path (Union[str, Path]): 変換する.npyファイルのパス
        csv_path (Optional[Union[str, Path]]): 出力するCSVファイルのパス。
                                              指定されない場合は.npyファイルと同じディレクトリに
                                              .csvの拡張子で保存されます。
        header (Optional[bool]): CSVにヘッダーを含めるかどうか（デフォルト: True）
        index (Optional[bool]): インデックスを含めるかどうか（デフォルト: False）
        column_names (Optional[list[str]]): カラム名のリスト。指定されない場合は
                                           自動的に生成されます。

    Returns:
        Path: 作成されたCSVファイルのパス

    Raises:
        FileNotFoundError: .npyファイルが見つからない場合
        ValueError: .npyファイルの読み込みに失敗した場合
    """
    npy_path = Path(npy_path)
    
    if not npy_path.exists():
        raise FileNotFoundError(f".npyファイルが見つかりません: {npy_path}")
    
    if not npy_path.suffix == '.npy':
        raise ValueError(f"拡張子が.npyではありません: {npy_path}")
    
    # CSVファイルのパスを決定
    if csv_path is None:
        csv_path = npy_path.with_suffix('.csv')
    else:
        csv_path = Path(csv_path)
    
    try:
        # .npyファイルを読み込み
        data = np.load(npy_path)
        
        # データをDataFrameに変換
        if data.ndim == 1:
            # 1次元の場合、縦ベクトルとして扱う
            df = pd.DataFrame(data, columns=column_names or ['value'])
        elif data.ndim == 2:
            # 2次元の場合
            if column_names is None:
                if data.shape[1] == 1:
                    column_names = ['value']
                else:
                    column_names = [f'col_{i}' for i in range(data.shape[1])]
            df = pd.DataFrame(data, columns=column_names)
        else:
            # 3次元以上の場合、リシェイプして2次元にする
            reshaped_data = data.reshape(data.shape[0], -1)
            if column_names is None:
                column_names = [f'col_{i}' for i in range(reshaped_data.shape[1])]
            df = pd.DataFrame(reshaped_data, columns=column_names)
        
        # CSVファイルに保存
        df.to_csv(csv_path, header=header, index=index)
        
        return csv_path
        
    except Exception as e:
        raise ValueError(f".npyファイルの変換に失敗しました: {e}")


def batch_convert_npy_to_csv(
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    pattern: str = "*.npy",
    **kwargs
) -> list[Path]:
    """
    指定されたディレクトリ内の複数の.npyファイルを一括でCSVに変換します。

    Args:
        input_dir (Union[str, Path]): .npyファイルが含まれるディレクトリ
        output_dir (Optional[Union[str, Path]]): 出力ディレクトリ。指定されない場合は
                                                入力ディレクトリと同じ場所に保存されます。
        pattern (str): ファイルパターン（デフォルト: "*.npy"）
        **kwargs: npy_to_csv関数に渡される追加のキーワード引数

    Returns:
        list[Path]: 作成されたCSVファイルのパスのリスト

    Raises:
        FileNotFoundError: 入力ディレクトリが見つからない場合
    """
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"入力ディレクトリが見つかりません: {input_dir}")
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    npy_files = list(input_dir.glob(pattern))
    
    if not npy_files:
        print(f"指定されたパターン '{pattern}' に一致する.npyファイルが見つかりません")
        return []
    
    converted_files = []
    
    for npy_file in npy_files:
        try:
            if output_dir is not None:
                csv_path = output_dir / f"{npy_file.stem}.csv"
            else:
                csv_path = None
            
            converted_path = npy_to_csv(npy_file, csv_path, **kwargs)
            converted_files.append(converted_path)
            print(f"変換完了: {npy_file} -> {converted_path}")
            
        except Exception as e:
            print(f"変換エラー: {npy_file} - {e}")
    
    return converted_files


if __name__ == "__main__":
    # 使用例
    import argparse
    
    parser = argparse.ArgumentParser(description='.npyファイルをCSVに変換します')
    parser.add_argument('input', help='変換する.npyファイルまたはディレクトリのパス')
    parser.add_argument('-o', '--output', help='出力ファイルまたはディレクトリのパス')
    parser.add_argument('--no-header', action='store_true', help='ヘッダーを含めない')
    parser.add_argument('--with-index', action='store_true', help='インデックスを含める')
    parser.add_argument('--batch', action='store_true', help='ディレクトリ内の複数ファイルを一括変換')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    try:
        if args.batch or input_path.is_dir():
            # バッチ変換
            converted_files = batch_convert_npy_to_csv(
                input_path,
                args.output,
                header=not args.no_header,
                index=args.with_index
            )
            print(f"\n変換完了: {len(converted_files)} ファイル")
        else:
            # 単一ファイル変換
            converted_path = npy_to_csv(
                input_path,
                args.output,
                header=not args.no_header,
                index=args.with_index
            )
            print(f"変換完了: {input_path} -> {converted_path}")
            
    except Exception as e:
        print(f"エラー: {e}")