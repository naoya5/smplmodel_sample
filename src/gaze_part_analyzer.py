#!/usr/bin/env python3
"""
視線データ部位別分析システム

指定フォルダ内の.npyファイル（視線データ）を読み込み、SMPLモデルの各部位への
視線集中度を計算・分析するプログラムです。
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests


class GazePartAnalyzer:
    """視線データの部位別分析を行うクラス"""
    
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
            frame_total = np.sum(gaze_values)
            
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
        
        # 確率の計算
        total_all_gaze = sum(part_info["total_gaze"] for part_info in part_results.values())
        
        for part_name, part_info in part_results.items():
            if total_all_gaze > 0:
                part_info["probability"] = part_info["total_gaze"] / total_all_gaze
                part_info["average_per_frame"] = part_info["total_gaze"] / len(self.gaze_data)
                part_info["average_per_vertex"] = (
                    part_info["total_gaze"] / part_info["vertex_count"] if part_info["vertex_count"] > 0 else 0.0
                )
            else:
                part_info["probability"] = 0.0
                part_info["average_per_frame"] = 0.0
                part_info["average_per_vertex"] = 0.0
        
        print(f"{len(part_results)}部位の集計が完了")
        self.part_results = part_results
        return part_results
    
    def get_timestamp_from_frame(self, frame_num: int) -> float:
        """
        フレーム番号から時間（秒）を計算
        
        Args:
            frame_num: フレーム番号
            
        Returns:
            時間（秒）
        """
        return frame_num / self.frame_rate
    
    def get_time_formatted(self, frame_num: int) -> str:
        """
        フレーム番号から時:分:秒形式の文字列を生成
        
        Args:
            frame_num: フレーム番号
            
        Returns:
            時:分:秒形式の文字列
        """
        total_seconds = self.get_timestamp_from_frame(frame_num)
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:06.3f}"
        else:
            return f"{minutes}:{seconds:06.3f}"
    
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
                "timestamp": frame_num / self.frame_rate,
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
    
    def detect_gaze_changes(self, min_change_threshold: float = 0.1) -> List[Dict[str, Any]]:
        """
        視線変化を検出してJSON形式で記録
        
        Args:
            min_change_threshold: 変化とみなす最小閾値（0.0-1.0）
            
        Returns:
            視線変化のリスト
        """
        if not self.frame_results:
            raise ValueError("フレーム別分析結果が準備されていません")
        
        changes = []
        previous_max_part = None
        
        for frame_result in self.frame_results:
            current_max_part = frame_result["max_part"]
            current_ratio = frame_result["part_ratios"].get(current_max_part, 0) if current_max_part else 0
            
            # 変化があり、かつ閾値を超えている場合
            if (previous_max_part != current_max_part and 
                current_max_part is not None and 
                current_ratio >= min_change_threshold):
                
                timestamp = frame_result.get("timestamp", self.get_timestamp_from_frame(frame_result["frame"]))
                time_str = self.get_time_formatted(frame_result["frame"])
                
                change_data = {
                    "timestamp": round(timestamp, 3),
                    "time_formatted": time_str,
                    "frame": frame_result["frame"],
                    "from_part": previous_max_part,
                    "to_part": current_max_part,
                    "confidence": round(current_ratio, 3),
                    "description": f"{timestamp:.1f}秒: {previous_max_part or '無し'}から{current_max_part}に移動"
                }
                changes.append(change_data)
            
            previous_max_part = current_max_part
        
        return changes
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        全体の統計サマリーを取得
        
        Returns:
            統計サマリー
        """
        if not self.part_results or not self.frame_results:
            raise ValueError("分析結果が準備されていません")
        
        # 部位別統計
        sorted_parts = sorted(
            self.part_results.items(),
            key=lambda x: x[1]["probability"],
            reverse=True
        )
        
        # フレーム別最大注視部位の統計
        max_parts_count = {}
        for frame_result in self.frame_results:
            max_part = frame_result["max_part"]
            if max_part:
                max_parts_count[max_part] = max_parts_count.get(max_part, 0) + 1
        
        # 全部位のフレーム集計を計算
        all_parts_frame_analysis = []
        for part_name, part_info in sorted_parts:
            frame_count_with_gaze = sum(1 for frame in self.frame_results 
                                       if frame["part_values"].get(part_name, 0) > 0)
            frame_count_without_gaze = len(self.frame_results) - frame_count_with_gaze
            all_parts_frame_analysis.append({
                "part": part_name,
                "frame_count_with_gaze": frame_count_with_gaze,
                "frame_count_without_gaze": frame_count_without_gaze,
                "percentage_with_gaze": frame_count_with_gaze / len(self.frame_results) * 100,
                "percentage_without_gaze": frame_count_without_gaze / len(self.frame_results) * 100,
                "probability": part_info["probability"],
                "total_gaze": part_info["total_gaze"]
            })
        
        summary = {
            "total_frames": len(self.frame_results),
            "total_parts": len(self.part_results),
            "top_parts_by_probability": [
                {
                    "part": part,
                    "probability": info["probability"],
                    "total_gaze": info["total_gaze"],
                    "avg_per_frame": info["average_per_frame"]
                }
                for part, info in sorted_parts[:10]
            ],
            "most_attended_frames": [
                {
                    "part": part,
                    "frame_count": count,
                    "percentage": count / len(self.frame_results) * 100
                }
                for part, count in sorted(max_parts_count.items(), key=lambda x: x[1], reverse=True)
            ],
            "all_parts_frame_analysis": all_parts_frame_analysis
        }
        
        return summary
    
    def print_results(self) -> None:
        """結果をコンソールに表示"""
        if not self.part_results or not self.frame_results:
            print("分析結果がありません")
            return
        
        summary = self.get_summary_statistics()
        
        print("\n" + "="*60)
        print("視線データ部位別分析結果")
        print("="*60)
        
        print(f"\n総フレーム数: {summary['total_frames']}")
        print(f"分析対象部位数: {summary['total_parts']}")
        if self.frame_results:
            total_duration = self.get_timestamp_from_frame(max(f["frame"] for f in self.frame_results))
            print(f"総再生時間: {self.get_time_formatted(max(f['frame'] for f in self.frame_results))} ({total_duration:.3f}秒)")
            print(f"フレームレート: {self.frame_rate}Hz")
        
        print("\n【部位別視線確率 (上位10位)】")
        print("-" * 50)
        for i, part_data in enumerate(summary["top_parts_by_probability"], 1):
            print(f"{i:2d}. {part_data['part']:15s} "
                  f"{part_data['probability']*100:6.2f}% "
                  f"(総視線値: {part_data['total_gaze']:8.1f})")
        
        print("\n【フレーム別最大注視部位 (上位5位)】")
        print("-" * 50)
        for i, frame_data in enumerate(summary["most_attended_frames"], 1):
            print(f"{i:2d}. {frame_data['part']:15s} "
                  f"{frame_data['frame_count']:3d}フレーム "
                  f"({frame_data['percentage']:5.1f}%)")
        
        print("\n【フレーム別詳細 (最初の10フレーム)】")
        print("-" * 50)
        for frame_result in self.frame_results[:10]:
            max_part = frame_result["max_part"]
            max_ratio = frame_result["part_ratios"].get(max_part, 0) * 100 if max_part else 0
            timestamp = frame_result.get("timestamp", self.get_timestamp_from_frame(frame_result["frame"]))
            time_str = self.get_time_formatted(frame_result["frame"])
            print(f"フレーム {frame_result['frame']:06d} ({time_str}): "
                  f"{max_part:15s} ({max_ratio:5.1f}%)")
        
        if len(self.frame_results) > 10:
            print(f"... (残り{len(self.frame_results) - 10}フレーム)")
        
        # 視線変化情報を表示
        gaze_changes = self.detect_gaze_changes()
        print(f"\n【視線変化検出 (最初の10件)】")
        print("-" * 50)
        for i, change in enumerate(gaze_changes[:10], 1):
            print(f"{i:2d}. {change['description']}")
        
        if len(gaze_changes) > 10:
            print(f"... (残り{len(gaze_changes) - 10}件の変化)")
        print(f"総変化回数: {len(gaze_changes)}回")
    
    def export_results(self, output_dir: Path = Path("output")) -> Dict[str, Path]:
        """
        結果をファイルに出力
        
        Args:
            output_dir: 出力ディレクトリ
            
        Returns:
            出力ファイルパス辞書
        """
        output_dir.mkdir(exist_ok=True)
        output_files = {}
        
        # CSV: 部位別詳細データ
        part_csv_path = output_dir / "part_gaze_analysis.csv"
        part_data = []
        for part_name, part_info in self.part_results.items():
            part_data.append({
                "part_name": part_name,
                "total_gaze": part_info["total_gaze"],
                "probability": part_info["probability"],
                "average_per_frame": part_info["average_per_frame"],
                "average_per_vertex": part_info["average_per_vertex"],
                "vertex_count": part_info["vertex_count"]
            })
        
        pd.DataFrame(part_data).to_csv(part_csv_path, index=False)
        output_files["part_analysis"] = part_csv_path
        
        # CSV: フレーム別データ
        frame_csv_path = output_dir / "frame_gaze_analysis.csv"
        frame_data = []
        for frame_result in self.frame_results:
            row = {
                "frame": frame_result["frame"],
                "timestamp": frame_result.get("timestamp", self.get_timestamp_from_frame(frame_result["frame"])),
                "time_formatted": self.get_time_formatted(frame_result["frame"]),
                "max_part": frame_result["max_part"],
                "max_value": frame_result["max_value"],
                "total_gaze": frame_result["total_gaze"]
            }
            # 各部位の値を追加
            for part_name in self.part_results.keys():
                row[f"{part_name}_value"] = frame_result["part_values"].get(part_name, 0.0)
                row[f"{part_name}_ratio"] = frame_result["part_ratios"].get(part_name, 0.0)
            
            frame_data.append(row)
        
        pd.DataFrame(frame_data).to_csv(frame_csv_path, index=False)
        output_files["frame_analysis"] = frame_csv_path
        
        # JSON: サマリー統計
        summary_json_path = output_dir / "gaze_analysis_summary.json"
        summary = self.get_summary_statistics()
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        output_files["summary"] = summary_json_path
        
        # JSON: 視線変化データ
        changes_json_path = output_dir / "gaze_changes.json"
        gaze_changes = self.detect_gaze_changes()
        change_data = {
            "frame_rate": self.frame_rate,
            "total_changes": len(gaze_changes),
            "changes": gaze_changes
        }
        with open(changes_json_path, "w", encoding="utf-8") as f:
            json.dump(change_data, f, indent=2, ensure_ascii=False)
        output_files["gaze_changes"] = changes_json_path
        
        print(f"\n結果を出力しました:")
        for key, path in output_files.items():
            print(f"  {key}: {path}")
        
        return output_files
    
    def visualize_results(self, output_dir: Path = Path("output")) -> Path:
        """
        結果の可視化
        
        Args:
            output_dir: 出力ディレクトリ
            
        Returns:
            グラフファイルパス
        """
        if not self.part_results:
            raise ValueError("分析結果がありません")
        
        output_dir.mkdir(exist_ok=True)
        
        # フォント設定を改善
        plt.rcParams['font.family'] = ['DejaVu Sans']
        plt.rcParams['font.size'] = 10
        
        # 部位別確率の上位15位をプロット
        sorted_parts = sorted(
            self.part_results.items(),
            key=lambda x: x[1]["probability"],
            reverse=True
        )[:15]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 1. 部位別確率の棒グラフ
        parts = [item[0] for item in sorted_parts]
        probs = [item[1]["probability"] * 100 for item in sorted_parts]
        
        ax1.bar(range(len(parts)), probs, color='skyblue', alpha=0.7)
        ax1.set_xticks(range(len(parts)))
        ax1.set_xticklabels(parts, rotation=45, ha='right', fontsize=9)
        ax1.set_ylabel('Gaze Probability (%)', fontsize=11)
        ax1.set_title('Body Part Gaze Distribution (Top 15)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. フレーム別最大注視部位の推移（最初の50フレーム）
        frame_subset = self.frame_results[:50]
        frames = [f["frame"] for f in frame_subset]
        max_parts = [f["max_part"] for f in frame_subset]
        
        # Noneを除外して部位を数値にマッピング
        valid_max_parts = [part for part in max_parts if part is not None]
        if not valid_max_parts:
            print("警告: 有効な最大注視部位データがありません")
            unique_parts = ["unknown"]
            max_part_nums = [0] * len(max_parts)
        else:
            unique_parts = sorted(set(valid_max_parts))
            part_to_num = {part: i for i, part in enumerate(unique_parts)}
            max_part_nums = [part_to_num.get(part, -1) if part is not None else -1 for part in max_parts]
        
        ax2.plot(frames, max_part_nums, 'o-', markersize=4, linewidth=1)
        ax2.set_xlabel('Frame Number', fontsize=11)
        ax2.set_ylabel('Most Attended Body Part', fontsize=11)
        ax2.set_title('Frame-by-Frame Maximum Attention Transition (First 50 Frames)', fontsize=12, fontweight='bold')
        ax2.set_yticks(range(len(unique_parts)))
        ax2.set_yticklabels(unique_parts, fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        graph_path = output_dir / "gaze_part_analysis.png"
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可視化グラフを保存: {graph_path}")
        return graph_path
    
    def visualize_results_with_seconds(self, output_dir: Path = Path("output")) -> Path:
        """
        結果の可視化（秒数バージョン）
        
        Args:
            output_dir: 出力ディレクトリ
            
        Returns:
            グラフファイルパス
        """
        if not self.part_results:
            raise ValueError("分析結果がありません")
        
        output_dir.mkdir(exist_ok=True)
        
        # フォント設定を改善
        plt.rcParams['font.family'] = ['DejaVu Sans']
        plt.rcParams['font.size'] = 10
        
        # 部位別確率の上位15位をプロット
        sorted_parts = sorted(
            self.part_results.items(),
            key=lambda x: x[1]["probability"],
            reverse=True
        )[:15]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 1. 部位別確率の棒グラフ
        parts = [item[0] for item in sorted_parts]
        probs = [item[1]["probability"] * 100 for item in sorted_parts]
        
        ax1.bar(range(len(parts)), probs, color='skyblue', alpha=0.7)
        ax1.set_xticks(range(len(parts)))
        ax1.set_xticklabels(parts, rotation=45, ha='right', fontsize=9)
        ax1.set_ylabel('Gaze Probability (%)', fontsize=11)
        ax1.set_title('Body Part Gaze Distribution (Top 15)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. 時間別最大注視部位の推移（最初の50フレーム）
        frame_subset = self.frame_results[:50]
        timestamps = [self.get_timestamp_from_frame(f["frame"]) for f in frame_subset]
        max_parts = [f["max_part"] for f in frame_subset]
        
        # Noneを除外して部位を数値にマッピング
        valid_max_parts = [part for part in max_parts if part is not None]
        if not valid_max_parts:
            print("警告: 有効な最大注視部位データがありません")
            unique_parts = ["unknown"]
            max_part_nums = [0] * len(max_parts)
        else:
            unique_parts = sorted(set(valid_max_parts))
            part_to_num = {part: i for i, part in enumerate(unique_parts)}
            max_part_nums = [part_to_num.get(part, -1) if part is not None else -1 for part in max_parts]
        
        ax2.plot(timestamps, max_part_nums, 'o-', markersize=4, linewidth=1, color='#2E86AB')
        ax2.set_xlabel('Time (seconds)', fontsize=11)
        ax2.set_ylabel('Most Attended Body Part', fontsize=11)
        ax2.set_title('Time-based Maximum Attention Transition (First 50 Frames)', fontsize=12, fontweight='bold')
        ax2.set_yticks(range(len(unique_parts)))
        ax2.set_yticklabels(unique_parts, fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # X軸の時間表示を改善
        ax2.tick_params(axis='x', labelsize=9)
        
        plt.tight_layout()
        
        graph_path = output_dir / "gaze_part_analysis_seconds.png"
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"秒数版可視化グラフを保存: {graph_path}")
        return graph_path
    
    def visualize_full_timeline(self, output_dir: Path = Path("output")) -> Path:
        """
        全フレームの時系列可視化
        
        Args:
            output_dir: 出力ディレクトリ
            
        Returns:
            グラフファイルパス
        """
        if not self.part_results:
            raise ValueError("分析結果がありません")
        
        output_dir.mkdir(exist_ok=True)
        
        # フォント設定を改善
        plt.rcParams['font.family'] = ['DejaVu Sans']
        plt.rcParams['font.size'] = 10
        
        # 部位別確率の上位15位をプロット
        sorted_parts = sorted(
            self.part_results.items(),
            key=lambda x: x[1]["probability"],
            reverse=True
        )[:15]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 1. 部位別確率の棒グラフ
        parts = [item[0] for item in sorted_parts]
        probs = [item[1]["probability"] * 100 for item in sorted_parts]
        
        ax1.bar(range(len(parts)), probs, color='skyblue', alpha=0.7)
        ax1.set_xticks(range(len(parts)))
        ax1.set_xticklabels(parts, rotation=45, ha='right', fontsize=9)
        ax1.set_ylabel('Gaze Probability (%)', fontsize=11)
        ax1.set_title('Body Part Gaze Distribution (Top 15)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. 全フレームの時間別最大注視部位の推移
        timestamps = [self.get_timestamp_from_frame(f["frame"]) for f in self.frame_results]
        max_parts = [f["max_part"] for f in self.frame_results]
        
        # Noneを除外して部位を数値にマッピング
        valid_max_parts = [part for part in max_parts if part is not None]
        if not valid_max_parts:
            print("警告: 有効な最大注視部位データがありません")
            unique_parts = ["unknown"]
            max_part_nums = [0] * len(max_parts)
        else:
            unique_parts = sorted(set(valid_max_parts))
            part_to_num = {part: i for i, part in enumerate(unique_parts)}
            max_part_nums = [part_to_num.get(part, -1) if part is not None else -1 for part in max_parts]
        
        # 線グラフではなく散布図として表示（全フレームなので）
        ax2.scatter(timestamps, max_part_nums, s=8, alpha=0.7, color='#2E86AB')
        ax2.set_xlabel('Time (seconds)', fontsize=11)
        ax2.set_ylabel('Most Attended Body Part', fontsize=11)
        ax2.set_title(f'Time-based Maximum Attention Transition (All {len(self.frame_results)} Frames)', fontsize=12, fontweight='bold')
        ax2.set_yticks(range(len(unique_parts)))
        ax2.set_yticklabels(unique_parts, fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # X軸の時間表示を改善
        ax2.tick_params(axis='x', labelsize=9)
        
        # 総時間を表示
        total_time = max(timestamps)
        ax2.text(0.02, 0.98, f'Total duration: {total_time:.1f}s', 
                transform=ax2.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        graph_path = output_dir / "gaze_part_analysis_full_timeline.png"
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"全フレーム時系列グラフを保存: {graph_path}")
        return graph_path
    
    def run_analysis(self, data_folder: Path, output_dir: Path = Path("output")) -> None:
        """
        完全な分析を実行
        
        Args:
            data_folder: 視線データフォルダ
            output_dir: 出力ディレクトリ
        """
        print("視線データ部位別分析を開始")
        print(f"データフォルダ: {data_folder}")
        print(f"出力ディレクトリ: {output_dir}")
        print("-" * 50)
        
        try:
            # 1. データ読み込み
            self.load_gaze_data(data_folder)
            
            # 2. 部位別集計
            self.map_gaze_to_parts()
            
            # 3. フレーム別分析
            self.analyze_frame_attention()
            
            # 4. 結果表示
            self.print_results()
            
            # 5. ファイル出力
            self.export_results(output_dir)
            
            # 6. 可視化
            self.visualize_results(output_dir)
            
            print("\n分析が正常に完了しました!")
            
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            raise