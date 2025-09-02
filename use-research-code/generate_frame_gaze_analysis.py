#!/usr/bin/env python3
"""
視線データフレーム別分析スクリプト

指定フォルダ内の.npyファイル（視線データ）を読み込み、SMPLモデルの各部位への
視線集中度をフレーム別に計算・分析して、frame_gaze_analysis.csvを生成します。

使用例:
    uv run generate_frame_gaze_analysis.py --data_folder data/User11 --output results/User11/frame_gaze_analysis.csv

引数:
    --data_folder: 視線データ(.npyファイル)が格納されたフォルダパス
    --output: 出力CSVファイルパス（デフォルト: frame_gaze_analysis.csv）
    --frame_rate: フレームレート（Hz、デフォルト: 90.0）
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import requests


class FrameGazeAnalyzer:
    """視線データのフレーム別分析を行うクラス"""
    
    def __init__(self, segmentation_file: str = "smpl_vert_segmentation.json", frame_rate: float = 90.0):
        """
        初期化
        
        Args:
            segmentation_file: SMPLモデルの頂点セグメンテーションファイルパス
            frame_rate: フレームレート（Hz）
        """
        self.segmentation_file = segmentation_file
        self.frame_rate = frame_rate
        self.segmentation_data: Optional[Dict[str, List[int]]] = None
        self.gaze_data: Dict[int, np.ndarray] = {}
        self.part_results: Dict[str, Dict[str, Any]] = {}
        self.frame_results: List[Dict[str, Any]] = []
        
    def download_segmentation_file(self) -> Dict[str, List[int]]:
        """
        SMPL用頂点セグメンテーションファイルをダウンロードまたは読み込み
        
        Returns:
            セグメンテーションデータ
        """
        if os.path.exists(self.segmentation_file):
            with open(self.segmentation_file, "r") as f:
                return json.load(f)
        
        url = (
            "https://raw.githubusercontent.com/Meshcapade/wiki/main/assets/"
            "SMPL_body_segmentation/smpl/smpl_vert_segmentation.json"
        )
        
        print(f"セグメンテーションファイルをダウンロード中: {url}")
        response = requests.get(url)
        response.raise_for_status()
        
        seg_data = response.json()
        with open(self.segmentation_file, "w") as f:
            json.dump(seg_data, f, indent=2)
        
        print(f"セグメンテーションファイルを保存: {self.segmentation_file}")
        return seg_data
    
    def load_gaze_data(self, data_folder: Path) -> Dict[int, np.ndarray]:
        """
        指定フォルダから.npyファイルを読み込み
        
        Args:
            data_folder: 視線データが格納されたフォルダパス
            
        Returns:
            フレーム番号をキーとした視線データ辞書
        """
        if not data_folder.exists():
            raise FileNotFoundError(f"データフォルダが見つかりません: {data_folder}")
        
        npy_files = list(data_folder.glob("*.npy"))
        if not npy_files:
            raise ValueError(f"npyファイルが見つかりません: {data_folder}")
        
        print(f"{len(npy_files)}個の.npyファイルを発見")
        
        gaze_data = {}
        for npy_file in sorted(npy_files):
            # ファイル名からフレーム番号を抽出
            frame_match = re.search(r'(\d+)', npy_file.stem)
            if frame_match:
                frame_num = int(frame_match.group(1))
            else:
                print(f"警告: フレーム番号を抽出できません: {npy_file.name}")
                continue
                
            try:
                data = np.load(npy_file)
                if data.shape[0] != 6890:
                    print(f"警告: 予期しないデータ形状 {data.shape}: {npy_file.name}")
                    continue
                
                # (6890, 1) -> (6890,) に変換
                if len(data.shape) == 2 and data.shape[1] == 1:
                    data = data.flatten()
                
                gaze_data[frame_num] = data
                
            except Exception as e:
                print(f"エラー: ファイル読み込み失敗 {npy_file.name}: {e}")
                continue
        
        if not gaze_data:
            raise ValueError("有効な視線データが見つかりませんでした")
        
        print(f"{len(gaze_data)}フレームの視線データを読み込み完了")
        self.gaze_data = gaze_data
        return gaze_data
    
    def map_gaze_to_parts(self) -> Dict[str, Dict[str, Any]]:
        """
        視線データを部位別に集計
        
        Returns:
            部位別の視線データ
        """
        if self.segmentation_data is None:
            self.segmentation_data = self.download_segmentation_file()
        
        if not self.gaze_data:
            raise ValueError("視線データが読み込まれていません")
        
        print("部位別視線データの集計を開始")
        
        # 部位別結果の初期化
        part_results = {}
        for part_name in self.segmentation_data.keys():
            part_results[part_name] = {
                "total_gaze": 0.0,
                "frame_values": {},
                "vertex_indices": self.segmentation_data[part_name],
                "vertex_count": len(self.segmentation_data[part_name])
            }
        
        # フレームごとに部位別集計
        for frame_num, gaze_values in self.gaze_data.items():
            for part_name, part_info in part_results.items():
                vertex_indices = part_info["vertex_indices"]
                
                # 頂点インデックスの範囲チェック
                valid_indices = [idx for idx in vertex_indices if 0 <= idx < len(gaze_values)]
                
                if valid_indices:
                    part_gaze_sum = np.sum(gaze_values[valid_indices])
                    part_results[part_name]["frame_values"][frame_num] = part_gaze_sum
                    part_results[part_name]["total_gaze"] += part_gaze_sum
                else:
                    part_results[part_name]["frame_values"][frame_num] = 0.0
        
        print(f"{len(part_results)}部位の集計が完了")
        self.part_results = part_results
        return part_results
    
    def analyze_frame_attention(self) -> List[Dict[str, Any]]:
        """
        フレームごとの最大注視部位分析
        
        Returns:
            フレーム別分析結果
        """
        if not self.part_results:
            raise ValueError("部位別データが計算されていません")
        
        print("フレーム別注視分析を開始")
        
        frame_results = []
        
        for frame_num in sorted(self.gaze_data.keys()):
            frame_data = {
                "frame": frame_num,
                "part_values": {},
                "max_part": None,
                "max_value": 0.0,
                "total_gaze": 0.0
            }
            
            # 各部位の視線値を取得
            for part_name, part_info in self.part_results.items():
                gaze_value = part_info["frame_values"].get(frame_num, 0.0)
                frame_data["part_values"][part_name] = gaze_value
                frame_data["total_gaze"] += gaze_value
                
                # 最大値の更新
                if gaze_value > frame_data["max_value"]:
                    frame_data["max_value"] = gaze_value
                    frame_data["max_part"] = part_name
            
            # 各部位の割合を計算
            if frame_data["total_gaze"] > 0:
                frame_data["part_ratios"] = {
                    part: value / frame_data["total_gaze"] 
                    for part, value in frame_data["part_values"].items()
                }
            else:
                frame_data["part_ratios"] = {
                    part: 0.0 for part in frame_data["part_values"].keys()
                }
            
            frame_results.append(frame_data)
        
        print(f"{len(frame_results)}フレームの分析が完了")
        self.frame_results = frame_results
        return frame_results
    
    def export_frame_analysis_csv(self, output_path: Path) -> None:
        """
        フレーム別分析結果をCSVファイルに出力
        
        Args:
            output_path: 出力CSVファイルパス
        """
        if not self.frame_results:
            raise ValueError("フレーム別分析結果がありません")
        
        # 出力ディレクトリを作成
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        frame_data = []
        for frame_result in self.frame_results:
            row = {
                "frame": frame_result["frame"],
                "max_part": frame_result["max_part"],
                "max_value": frame_result["max_value"],
                "total_gaze": frame_result["total_gaze"]
            }
            
            # 各部位の値と割合を追加
            for part_name in sorted(self.part_results.keys()):
                row[f"{part_name}_value"] = frame_result["part_values"].get(part_name, 0.0)
                row[f"{part_name}_ratio"] = frame_result["part_ratios"].get(part_name, 0.0)
            
            frame_data.append(row)
        
        # CSVに出力
        df = pd.DataFrame(frame_data)
        df.to_csv(output_path, index=False)
        
        print(f"フレーム別分析結果をCSVに出力: {output_path}")
        print(f"総フレーム数: {len(frame_data)}")
        print(f"分析した部位数: {len(self.part_results)}")
    
    def run_analysis(self, data_folder: Path, output_path: Path) -> None:
        """
        フレーム別視線分析を実行してCSVファイルを生成
        
        Args:
            data_folder: 視線データフォルダ
            output_path: 出力CSVファイルパス
        """
        print("フレーム別視線分析を開始")
        print(f"データフォルダ: {data_folder}")
        print(f"出力ファイル: {output_path}")
        print("-" * 60)
        
        try:
            # 1. データ読み込み
            self.load_gaze_data(data_folder)
            
            # 2. 部位別集計
            self.map_gaze_to_parts()
            
            # 3. フレーム別分析
            self.analyze_frame_attention()
            
            # 4. CSV出力
            self.export_frame_analysis_csv(output_path)
            
            print("\nframe_gaze_analysis.csvの生成が正常に完了しました!")
            
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            raise


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="視線データフレーム別分析スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    uv run generate_frame_gaze_analysis.py --data_folder data/User11 --output results/User11/frame_gaze_analysis.csv
    uv run generate_frame_gaze_analysis.py --data_folder data/User11 --output frame_gaze_analysis.csv --frame_rate 60.0
        """
    )
    
    parser.add_argument(
        "--data_folder",
        type=str,
        required=True,
        help="視線データ(.npyファイル)が格納されたフォルダパス"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="frame_gaze_analysis.csv",
        help="出力CSVファイルパス（デフォルト: frame_gaze_analysis.csv）"
    )
    
    parser.add_argument(
        "--frame_rate",
        type=float,
        default=90.0,
        help="フレームレート（Hz、デフォルト: 90.0）"
    )
    
    args = parser.parse_args()
    
    # パスの変換
    data_folder = Path(args.data_folder)
    output_path = Path(args.output)
    
    # データフォルダの存在確認
    if not data_folder.exists():
        print(f"エラー: データフォルダが存在しません: {data_folder}")
        return
    
    if not data_folder.is_dir():
        print(f"エラー: 指定されたパスはディレクトリではありません: {data_folder}")
        return
    
    # 分析器の作成と実行
    analyzer = FrameGazeAnalyzer(frame_rate=args.frame_rate)
    analyzer.run_analysis(data_folder, output_path)


if __name__ == "__main__":
    main()