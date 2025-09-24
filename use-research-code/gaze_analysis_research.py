#!/usr/bin/env python3
"""
研究用視線データ分析システム

このプログラムは、フレーム単位の視線分析結果（frame_gaze_analysis.csv）を入力として、
各部位の注視時間・注視変化（トランジション）・基本統計・タイムライン情報を算出し、
研究用途のファイル群として保存します。

- 生成物
  - part_gaze_durations.csv: 部位別の累積注視時間（秒）
  - gaze_transitions.csv: 注視部位の変化イベント（frame, time_seconds, from_part, to_part）
  - part_statistics.json: 各部位の視線値の基本統計（合計/平均/最大/最小、支配フレーム等）
  - timeline_analysis.json: 支配部位シーケンスや時刻列などのタイムライン情報
  - analysis_summary.md: 上記の概要を日本語でまとめたサマリーレポート

- 入力
  - output/frame_gaze_analysis.csv（既定）または引数で指定したパス

- 出力先
  - use-research-code/analysis_results（既定）または引数で指定したディレクトリ

- 使い方
  - デフォルトパスで実行:
    python use-research-code/gaze_analysis_research.py
  - 入力と出力先を指定:
    python use-research-code/gaze_analysis_research.py <csv_path> <output_dir>

主な処理の流れ:
1) データ読み込み
2) 部位別注視時間の算出
3) 注視変化（トランジション）の抽出
4) 統計量の計算
5) タイムライン分析の生成
6) 各種ファイルの保存とサマリー生成
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class GazeResearchAnalyzer:
    """視線データの研究用分析クラス"""

    def __init__(self, frame_rate: float = 30.0):
        """
        初期化

        Args:
            frame_rate: フレームレート（Hz）
        """
        self.frame_rate = frame_rate
        self.frame_duration = 1.0 / frame_rate  # 1フレームの時間（秒）

    def load_frame_gaze_data(self, csv_path: str) -> pd.DataFrame:
        """
        フレーム単位の視線データを読み込み

        Args:
            csv_path: frame_gaze_analysis.csvのパス

        Returns:
            データフレーム
        """
        df = pd.read_csv(csv_path)
        return df

    def calculate_part_gaze_duration(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        各部位の注視時間を計算（秒単位）

        Args:
            df: フレーム単位の視線データ

        Returns:
            各部位の注視時間辞書
        """
        part_duration = {}

        # max_partカラムから各部位が最も注視された回数を計算
        part_counts = df["max_part"].value_counts()

        # フレーム数を時間に変換
        for part, count in part_counts.items():
            part_duration[part] = count * self.frame_duration

        return part_duration

    def find_gaze_transitions(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        注視変化のタイミングを特定

        Args:
            df: フレーム単位の視線データ

        Returns:
            注視変化情報のリスト
        """
        transitions = []
        prev_part = None

        for idx, row in df.iterrows():
            current_part = row["max_part"]
            frame_num = row["frame"]
            time_sec = frame_num * self.frame_duration

            if prev_part is not None and prev_part != current_part:
                transitions.append(
                    {
                        "frame": frame_num,
                        "time_seconds": time_sec,
                        "from_part": prev_part,
                        "to_part": current_part,
                        "transition_id": len(transitions) + 1,
                    }
                )

            prev_part = current_part

        return transitions

    def calculate_part_statistics(
        self, df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        各部位の統計情報を計算

        Args:
            df: フレーム単位の視線データ

        Returns:
            各部位の統計情報
        """
        stats = {}

        # 全部位の名前を取得（max_value以外の_valueで終わるカラム）
        value_columns = [
            col for col in df.columns if col.endswith("_value") and col != "max_value"
        ]
        part_names = [col.replace("_value", "") for col in value_columns]

        total_frames = len(df)
        total_time = total_frames * self.frame_duration

        for part in part_names:
            value_col = f"{part}_value"
            ratio_col = f"{part}_ratio"

            if value_col in df.columns:
                # 基本統計
                values = df[value_col]
                stats[part] = {
                    "total_gaze_value": float(values.sum()),
                    "average_gaze_value": float(values.mean()),
                    "max_gaze_value": float(values.max()),
                    "min_gaze_value": float(values.min()),
                    "frames_with_gaze": int((values > 0).sum()),
                    "time_with_gaze_seconds": float(
                        (values > 0).sum() * self.frame_duration
                    ),
                    "gaze_percentage": float((values > 0).sum() / total_frames * 100),
                }

                # 最も注視された回数
                max_part_count = (df["max_part"] == part).sum()
                stats[part]["dominant_frames"] = int(max_part_count)
                stats[part]["dominant_time_seconds"] = float(
                    max_part_count * self.frame_duration
                )
                stats[part]["dominant_percentage"] = float(
                    max_part_count / total_frames * 100
                )

        return stats

    def generate_timeline_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        タイムライン分析を生成

        Args:
            df: フレーム単位の視線データ

        Returns:
            タイムライン分析結果
        """
        timeline = {
            "total_frames": len(df),
            "total_time_seconds": len(df) * self.frame_duration,
            "frame_rate": self.frame_rate,
            "dominant_parts_sequence": df["max_part"].tolist(),
            "time_sequence": (df["frame"] * self.frame_duration).tolist(),
        }

        return timeline

    def save_results(
        self,
        output_dir: str,
        part_durations: Dict[str, float],
        transitions: List[Dict[str, Any]],
        stats: Dict[str, Dict[str, float]],
        timeline: Dict[str, Any],
    ) -> None:
        """
        分析結果を保存

        Args:
            output_dir: 出力ディレクトリ
            part_durations: 各部位の注視時間
            transitions: 注視変化情報
            stats: 各部位の統計情報
            timeline: タイムライン分析結果
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1. 部位別注視時間を保存
        duration_path = os.path.join(output_dir, "part_gaze_durations.csv")
        with open(duration_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["part_name", "duration_seconds"])
            for part, duration in sorted(
                part_durations.items(), key=lambda x: x[1], reverse=True
            ):
                writer.writerow([part, f"{duration:.4f}"])

        # 2. 注視変化タイミングを保存
        transitions_path = os.path.join(output_dir, "gaze_transitions.csv")
        with open(transitions_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["transition_id", "frame", "time_seconds", "from_part", "to_part"]
            )
            for trans in transitions:
                writer.writerow(
                    [
                        trans["transition_id"],
                        trans["frame"],
                        f"{trans['time_seconds']:.4f}",
                        trans["from_part"],
                        trans["to_part"],
                    ]
                )

        # 3. 詳細統計情報を保存
        stats_path = os.path.join(output_dir, "part_statistics.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        # 4. タイムライン情報を保存
        timeline_path = os.path.join(output_dir, "timeline_analysis.json")
        with open(timeline_path, "w", encoding="utf-8") as f:
            json.dump(timeline, f, indent=2, ensure_ascii=False)

        # 5. サマリーレポートを生成
        self._generate_summary_report(
            output_dir, part_durations, transitions, stats, timeline
        )

    def _generate_summary_report(
        self,
        output_dir: str,
        part_durations: Dict[str, float],
        transitions: List[Dict[str, Any]],
        stats: Dict[str, Dict[str, float]],
        timeline: Dict[str, Any],
    ) -> None:
        """
        サマリーレポートを生成
        """
        report_path = os.path.join(output_dir, "analysis_summary.md")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# 視線データ分析レポート\n\n")

            f.write("## 基本情報\n")
            f.write(f"- 総フレーム数: {timeline['total_frames']}\n")
            f.write(f"- 総時間: {timeline['total_time_seconds']:.2f}秒\n")
            f.write(f"- フレームレート: {timeline['frame_rate']}Hz\n\n")

            f.write("## 部位別注視時間（上位10位）\n")
            sorted_durations = sorted(
                part_durations.items(), key=lambda x: x[1], reverse=True
            )[:10]
            for i, (part, duration) in enumerate(sorted_durations, 1):
                percentage = (duration / timeline["total_time_seconds"]) * 100
                f.write(f"{i}. {part}: {duration:.4f}秒 ({percentage:.2f}%)\n")
            f.write("\n")

            f.write("## 注視変化情報\n")
            f.write(f"- 総変化回数: {len(transitions)}回\n")
            if transitions:
                avg_duration = timeline["total_time_seconds"] / len(transitions)
                f.write(f"- 平均注視持続時間: {avg_duration:.4f}秒\n")
            f.write("\n")

            f.write("## 最初の10回の注視変化\n")
            for trans in transitions[:10]:
                f.write(
                    f"- {trans['time_seconds']:.3f}秒: {trans['from_part']} → {trans['to_part']}\n"
                )

        print(f"分析結果を {output_dir} に保存しました")

    def run_analysis(self, csv_path: str, output_dir: str) -> None:
        """
        分析を実行

        Args:
            csv_path: frame_gaze_analysis.csvのパス
            output_dir: 出力ディレクトリ
        """
        print("視線データ分析を開始します...")

        # データ読み込み
        df = self.load_frame_gaze_data(csv_path)
        print(f"データ読み込み完了: {len(df)}フレーム")

        # 部位別注視時間計算
        part_durations = self.calculate_part_gaze_duration(df)
        print(f"部位別注視時間計算完了: {len(part_durations)}部位")

        # 注視変化タイミング特定
        transitions = self.find_gaze_transitions(df)
        print(f"注視変化タイミング特定完了: {len(transitions)}回の変化")

        # 統計情報計算
        stats = self.calculate_part_statistics(df)
        print("統計情報計算完了")

        # タイムライン分析
        timeline = self.generate_timeline_analysis(df)
        print("タイムライン分析完了")

        # 結果保存
        self.save_results(output_dir, part_durations, transitions, stats, timeline)
        print("分析完了")


def main():
    """メイン実行関数"""
    # デフォルトのパス設定
    csv_path = "output/frame_gaze_analysis.csv"
    output_dir = "use-research-code/analysis_results"

    # 引数からパスを取得する場合
    import sys

    if len(sys.argv) >= 2:
        csv_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_dir = sys.argv[2]

    # 分析実行
    analyzer = GazeResearchAnalyzer(frame_rate=30.0)
    analyzer.run_analysis(csv_path, output_dir)


if __name__ == "__main__":
    main()
