#!/usr/bin/env python3
"""
SMPL頂点セグメンテーションファイルから各部位の頂点数をカウントするプログラム
"""

import json
import os
from typing import Dict, List
from pathlib import Path


class VertexCounter:
    """SMPL頂点の部位別カウンター"""

    def __init__(self, segmentation_file: str = None):
        """
        Args:
            segmentation_file: セグメンテーションファイルのパス
                              Noneの場合はデフォルト位置を使用
        """
        if segmentation_file is None:
            # デフォルトのセグメンテーションファイルパス
            current_dir = Path(__file__).parent
            self.segmentation_file = current_dir.parent / "smpl_vert_segmentation.json"
        else:
            self.segmentation_file = Path(segmentation_file)

        self.part_vertices = {}
        self.load_segmentation()

    def load_segmentation(self):
        """セグメンテーションファイルを読み込む"""
        try:
            with open(self.segmentation_file, "r", encoding="utf-8") as f:
                self.part_vertices = json.load(f)
            print(
                f"セグメンテーションファイルを読み込みました: {self.segmentation_file}"
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"セグメンテーションファイルが見つかりません: {self.segmentation_file}"
            )
        except json.JSONDecodeError:
            raise ValueError(f"JSONファイルの形式が不正です: {self.segmentation_file}")

    def count_vertices_per_part(self) -> Dict[str, int]:
        """各部位の頂点数をカウント"""
        vertex_counts = {}

        for part_name, vertices in self.part_vertices.items():
            vertex_counts[part_name] = len(vertices)

        return vertex_counts

    def get_total_vertices(self) -> int:
        """総頂点数を取得"""
        all_vertices = set()
        for vertices in self.part_vertices.values():
            all_vertices.update(vertices)
        return len(all_vertices)

    def get_part_vertices(self, part_name: str) -> List[int]:
        """指定した部位の頂点リストを取得"""
        return self.part_vertices.get(part_name, [])

    def get_vertex_to_part_mapping(self) -> Dict[int, str]:
        """頂点番号から部位名へのマッピングを作成"""
        vertex_to_part = {}

        for part_name, vertices in self.part_vertices.items():
            for vertex in vertices:
                vertex_to_part[vertex] = part_name

        return vertex_to_part

    def print_summary(self):
        """頂点数の要約を表示"""
        vertex_counts = self.count_vertices_per_part()
        total_vertices = self.get_total_vertices()

        print("\n=== SMPL 頂点数サマリー ===")
        print(f"総頂点数: {total_vertices}")
        print(f"部位数: {len(self.part_vertices)}")
        print("\n部位別頂点数:")
        print("-" * 30)

        # 頂点数で降順ソート
        sorted_parts = sorted(vertex_counts.items(), key=lambda x: x[1], reverse=True)

        for part_name, count in sorted_parts:
            percentage = (count / total_vertices) * 100
            print(f"{part_name:15}: {count:4d} ({percentage:5.1f}%)")

        print("-" * 30)
        print(f"合計確認: {sum(vertex_counts.values())} 頂点")

    def save_counts_to_csv(self, output_file: str = "vertex_counts.csv"):
        """頂点数をCSVファイルに保存"""
        import csv

        vertex_counts = self.count_vertices_per_part()
        total_vertices = self.get_total_vertices()

        output_path = Path(output_file)

        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["part_name", "vertex_count", "percentage"])

            # 頂点数で降順ソート
            sorted_parts = sorted(
                vertex_counts.items(), key=lambda x: x[1], reverse=True
            )

            for part_name, count in sorted_parts:
                percentage = (count / total_vertices) * 100
                writer.writerow([part_name, count, f"{percentage:.2f}"])

        print(f"頂点数をCSVファイルに保存しました: {output_path}")

    def save_counts_to_json(self, output_file: str = "vertex_counts.json"):
        """頂点数をJSONファイルに保存"""
        vertex_counts = self.count_vertices_per_part()
        total_vertices = self.get_total_vertices()

        result = {
            "total_vertices": total_vertices,
            "total_parts": len(self.part_vertices),
            "part_counts": vertex_counts,
            "part_percentages": {
                part: (count / total_vertices) * 100
                for part, count in vertex_counts.items()
            },
        }

        output_path = Path(output_file)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"頂点数をJSONファイルに保存しました: {output_path}")


def main():
    """メイン関数"""
    try:
        # 頂点カウンターを作成
        counter = VertexCounter()

        # サマリーを表示
        counter.print_summary()

        # CSVとJSONで保存
        counter.save_counts_to_csv("lib/vertex_counts.csv")
        counter.save_counts_to_json("lib/vertex_counts.json")

        # 特定の部位の詳細情報を表示（例）
        print("\n=== 特定部位の詳細 ===")
        head_vertices = counter.get_part_vertices("head")
        print(f"頭部の頂点数: {len(head_vertices)}")
        print(f"頭部の頂点範囲: {min(head_vertices)} - {max(head_vertices)}")

        hand_vertices = counter.get_part_vertices("rightHand")
        print(f"右手の頂点数: {len(hand_vertices)}")
        print(f"右手の頂点範囲: {min(hand_vertices)} - {max(hand_vertices)}")

    except Exception as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
