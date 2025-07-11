"""
視線データ分布解析モジュール

このモジュールは視線データと3Dオブジェクトメッシュから、
時系列ごとの各頂点への視線データの分布確率を計算します。

主な機能:
- 視線データの読み込みと前処理
- 3Dメッシュの頂点データ処理
- 視線と頂点の空間的関係計算
- 確率分布のモデリング（ガウス混合モデル）
- 時系列解析
- 結果の可視化とエクスポート

著者: Claude Code
作成日: 2025-07-10
"""

import numpy as np
import trimesh
import json
import csv
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class GazePoint:
    """
    視線データポイントを表すデータクラス

    Attributes:
        timestamp: タイムスタンプ（秒）
        direction: 視線方向ベクトル (x, y, z)
        origin: 視線原点 (x, y, z) - カメラ/目の位置
        confidence: 視線データの信頼度 (0.0-1.0)
    """

    timestamp: float
    direction: np.ndarray
    origin: np.ndarray
    confidence: float = 1.0


@dataclass
class VertexGazeProbability:
    """
    頂点ごとの視線確率データ

    Attributes:
        vertex_index: 頂点インデックス
        probability: 視線が向けられる確率
        distance_score: 視線からの距離スコア
        angle_score: 視線角度スコア
        temporal_weight: 時間的重み
    """

    vertex_index: int
    probability: float
    distance_score: float
    angle_score: float
    temporal_weight: float


class GazeDataLoader:
    """
    視線データの読み込みと前処理を担当するクラス

    対応フォーマット:
    - CSV形式 (timestamp, gaze_x, gaze_y, gaze_z, origin_x, origin_y, origin_z, confidence)
    - JSON形式
    - NumPy形式
    """

    @staticmethod
    def load_from_csv(csv_path: Union[str, Path]) -> List[GazePoint]:
        """
        CSVファイルから視線データを読み込む

        CSVフォーマット:
        timestamp,gaze_x,gaze_y,gaze_z,origin_x,origin_y,origin_z,confidence

        Args:
            csv_path: CSVファイルのパス

        Returns:
            視線データポイントのリスト

        Raises:
            FileNotFoundError: ファイルが見つからない場合
            ValueError: データフォーマットが不正な場合
        """
        gaze_points = []

        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    gaze_point = GazePoint(
                        timestamp=float(row["timestamp"]),
                        direction=np.array(
                            [
                                float(row["gaze_x"]),
                                float(row["gaze_y"]),
                                float(row["gaze_z"]),
                            ]
                        ),
                        origin=np.array(
                            [
                                float(row["origin_x"]),
                                float(row["origin_y"]),
                                float(row["origin_z"]),
                            ]
                        ),
                        confidence=float(row.get("confidence", 1.0)),
                    )
                    gaze_points.append(gaze_point)

        except FileNotFoundError:
            raise FileNotFoundError(f"視線データファイルが見つかりません: {csv_path}")
        except (ValueError, KeyError) as e:
            raise ValueError(f"視線データのフォーマットが不正です: {e}")

        return gaze_points

    @staticmethod
    def load_from_json(json_path: Union[str, Path]) -> List[GazePoint]:
        """
        JSONファイルから視線データを読み込む

        JSONフォーマット:
        {
            "gaze_data": [
                {
                    "timestamp": 0.0,
                    "direction": [x, y, z],
                    "origin": [x, y, z],
                    "confidence": 0.95
                },
                ...
            ]
        }

        Args:
            json_path: JSONファイルのパス

        Returns:
            視線データポイントのリスト
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        gaze_points = []
        for item in data["gaze_data"]:
            gaze_point = GazePoint(
                timestamp=item["timestamp"],
                direction=np.array(item["direction"]),
                origin=np.array(item["origin"]),
                confidence=item.get("confidence", 1.0),
            )
            gaze_points.append(gaze_point)

        return gaze_points

    @staticmethod
    def normalize_gaze_directions(gaze_points: List[GazePoint]) -> List[GazePoint]:
        """
        視線方向ベクトルを正規化する

        Args:
            gaze_points: 視線データポイントのリスト

        Returns:
            正規化された視線データポイントのリスト
        """
        normalized_points = []
        for point in gaze_points:
            # 方向ベクトルを正規化
            direction_norm = np.linalg.norm(point.direction)
            if direction_norm > 0:
                normalized_direction = point.direction / direction_norm
            else:
                normalized_direction = point.direction

            normalized_point = GazePoint(
                timestamp=point.timestamp,
                direction=normalized_direction,
                origin=point.origin,
                confidence=point.confidence,
            )
            normalized_points.append(normalized_point)

        return normalized_points


class MeshProcessor:
    """
    3Dメッシュの処理と頂点データの管理を担当するクラス
    """

    def __init__(self, mesh: trimesh.Trimesh):
        """
        メッシュプロセッサーを初期化

        Args:
            mesh: trimeshオブジェクト
        """
        self.mesh = mesh
        self.vertices = mesh.vertices
        self.faces = mesh.faces

    @classmethod
    def load_from_obj(cls, obj_path: Union[str, Path]) -> "MeshProcessor":
        """
        OBJファイルからメッシュを読み込む

        Args:
            obj_path: OBJファイルのパス

        Returns:
            MeshProcessorインスタンス
        """
        mesh = trimesh.load(obj_path, process=False)
        return cls(mesh)

    def get_vertex_normals(self) -> np.ndarray:
        """
        各頂点の法線ベクトルを取得

        Returns:
            頂点法線の配列 (N, 3)
        """
        return self.mesh.vertex_normals

    def get_face_centers(self) -> np.ndarray:
        """
        各面の中心点を取得

        Returns:
            面中心点の配列 (M, 3)
        """
        face_vertices = self.vertices[self.faces]
        return np.mean(face_vertices, axis=1)

    def calculate_surface_area(self) -> float:
        """
        メッシュの表面積を計算

        Returns:
            表面積
        """
        return self.mesh.area


class SpatialRelationCalculator:
    """
    視線と3D頂点の空間的関係を計算するクラス
    """

    @staticmethod
    def point_to_line_distance(
        point: np.ndarray, line_origin: np.ndarray, line_direction: np.ndarray
    ) -> float:
        """
        点から直線（視線）への最短距離を計算

        Args:
            point: 3D点座標
            line_origin: 直線の原点
            line_direction: 直線の方向ベクトル（正規化済み）

        Returns:
            最短距離
        """
        # 点から直線原点へのベクトル
        point_vector = point - line_origin

        # 直線上の最近点までの投影距離
        projection_length = np.dot(point_vector, line_direction)

        # 最近点の座標
        closest_point = line_origin + projection_length * line_direction

        # 点から最近点への距離
        distance = np.linalg.norm(point - closest_point)

        return distance

    @staticmethod
    def calculate_gaze_angle(
        vertex: np.ndarray,
        gaze_origin: np.ndarray,
        gaze_direction: np.ndarray,
        vertex_normal: np.ndarray,
    ) -> float:
        """
        視線と頂点法線の角度を計算

        Args:
            vertex: 頂点座標
            gaze_origin: 視線原点
            gaze_direction: 視線方向
            vertex_normal: 頂点法線

        Returns:
            角度（ラジアン）
        """
        # 視線原点から頂点への方向ベクトル
        vertex_direction = vertex - gaze_origin
        vertex_direction_norm = np.linalg.norm(vertex_direction)

        if vertex_direction_norm > 0:
            vertex_direction = vertex_direction / vertex_direction_norm

        # 視線方向と頂点方向の角度
        gaze_angle = np.arccos(
            np.clip(np.dot(gaze_direction, vertex_direction), -1.0, 1.0)
        )

        # 視線方向と頂点法線の角度
        normal_angle = np.arccos(
            np.clip(np.dot(-gaze_direction, vertex_normal), -1.0, 1.0)
        )

        # 両方の角度を考慮した総合角度スコア
        combined_angle = (gaze_angle + normal_angle) / 2.0

        return combined_angle


class GazeDistributionAnalyzer:
    """
    視線データ分布解析のメインクラス

    時系列ごとの各頂点への視線データの分布確率を計算し、
    ガウス混合モデルによる確率分布のモデリングを行います。
    """

    def __init__(
        self,
        mesh_processor: MeshProcessor,
        distance_threshold: float = 0.1,
        angle_threshold: float = np.pi / 4,
        temporal_window: float = 1.0,
    ):
        """
        解析器を初期化

        Args:
            mesh_processor: メッシュプロセッサー
            distance_threshold: 視線距離の閾値
            angle_threshold: 視線角度の閾値（ラジアン）
            temporal_window: 時間窓のサイズ（秒）
        """
        self.mesh_processor = mesh_processor
        self.distance_threshold = distance_threshold
        self.angle_threshold = angle_threshold
        self.temporal_window = temporal_window
        self.spatial_calculator = SpatialRelationCalculator()

    def calculate_vertex_probabilities(
        self, gaze_points: List[GazePoint], timestamp: float
    ) -> List[VertexGazeProbability]:
        """
        指定時刻における各頂点への視線確率を計算

        Args:
            gaze_points: 視線データポイントのリスト
            timestamp: 対象時刻

        Returns:
            頂点ごとの視線確率リスト
        """
        # 時間窓内の視線データを抽出
        time_filtered_gazes = [
            gaze
            for gaze in gaze_points
            if abs(gaze.timestamp - timestamp) <= self.temporal_window / 2
        ]

        if not time_filtered_gazes:
            return []

        vertex_probabilities = []
        vertices = self.mesh_processor.vertices
        vertex_normals = self.mesh_processor.get_vertex_normals()

        for vertex_idx, (vertex, normal) in enumerate(zip(vertices, vertex_normals)):
            total_probability = 0.0
            total_weight = 0.0

            for gaze in time_filtered_gazes:
                # 距離スコアの計算
                distance = self.spatial_calculator.point_to_line_distance(
                    vertex, gaze.origin, gaze.direction
                )
                distance_score = np.exp(-distance / self.distance_threshold)

                # 角度スコアの計算
                angle = self.spatial_calculator.calculate_gaze_angle(
                    vertex, gaze.origin, gaze.direction, normal
                )
                angle_score = np.exp(-angle / self.angle_threshold)

                # 時間的重みの計算
                time_diff = abs(gaze.timestamp - timestamp)
                temporal_weight = np.exp(-time_diff / (self.temporal_window / 4))

                # 総合確率スコア
                combined_score = (
                    distance_score * angle_score * temporal_weight * gaze.confidence
                )

                total_probability += combined_score
                total_weight += temporal_weight * gaze.confidence

            # 正規化された確率
            if total_weight > 0:
                normalized_probability = total_probability / total_weight
            else:
                normalized_probability = 0.0

            vertex_prob = VertexGazeProbability(
                vertex_index=vertex_idx,
                probability=normalized_probability,
                distance_score=distance_score if time_filtered_gazes else 0.0,
                angle_score=angle_score if time_filtered_gazes else 0.0,
                temporal_weight=total_weight,
            )
            vertex_probabilities.append(vertex_prob)

        return vertex_probabilities

    def analyze_temporal_distribution(
        self, gaze_points: List[GazePoint], time_step: float = 0.1
    ) -> Dict[float, List[VertexGazeProbability]]:
        """
        時系列にわたる視線分布を解析

        Args:
            gaze_points: 視線データポイントのリスト
            time_step: 時間ステップ（秒）

        Returns:
            時刻ごとの頂点確率辞書
        """
        if not gaze_points:
            return {}

        # 時間範囲の決定
        timestamps = [gaze.timestamp for gaze in gaze_points]
        min_time = min(timestamps)
        max_time = max(timestamps)

        # 時系列解析
        temporal_distribution = {}
        current_time = min_time

        while current_time <= max_time:
            vertex_probs = self.calculate_vertex_probabilities(
                gaze_points, current_time
            )
            temporal_distribution[current_time] = vertex_probs
            current_time += time_step

        return temporal_distribution

    def fit_gaussian_mixture_model(
        self,
        temporal_distribution: Dict[float, List[VertexGazeProbability]],
        n_components: int = 3,
    ) -> GaussianMixture:
        """
        視線分布データにガウス混合モデルを適用

        Args:
            temporal_distribution: 時系列視線分布データ
            n_components: ガウス成分数

        Returns:
            学習済みガウス混合モデル
        """
        # 特徴量の抽出
        features = []
        for timestamp, vertex_probs in temporal_distribution.items():
            for vertex_prob in vertex_probs:
                if vertex_prob.probability > 0.01:  # 閾値以上の確率のみ
                    feature = [
                        timestamp,
                        vertex_prob.vertex_index,
                        vertex_prob.probability,
                        vertex_prob.distance_score,
                        vertex_prob.angle_score,
                    ]
                    features.append(feature)

        if not features:
            raise ValueError("ガウス混合モデルの学習に十分なデータがありません")

        features_array = np.array(features)

        # ガウス混合モデルの学習
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(features_array)

        return gmm


class ResultExporter:
    """
    解析結果のエクスポートと可視化を担当するクラス
    """

    @staticmethod
    def export_to_csv(
        temporal_distribution: Dict[float, List[VertexGazeProbability]],
        output_path: Union[str, Path],
    ) -> None:
        """
        解析結果をCSVファイルにエクスポート

        Args:
            temporal_distribution: 時系列視線分布データ
            output_path: 出力ファイルパス
        """
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # ヘッダー
            writer.writerow(
                [
                    "timestamp",
                    "vertex_index",
                    "probability",
                    "distance_score",
                    "angle_score",
                    "temporal_weight",
                ]
            )

            # データ
            for timestamp, vertex_probs in temporal_distribution.items():
                for vertex_prob in vertex_probs:
                    writer.writerow(
                        [
                            timestamp,
                            vertex_prob.vertex_index,
                            vertex_prob.probability,
                            vertex_prob.distance_score,
                            vertex_prob.angle_score,
                            vertex_prob.temporal_weight,
                        ]
                    )

    @staticmethod
    def export_to_json(
        temporal_distribution: Dict[float, List[VertexGazeProbability]],
        output_path: Union[str, Path],
    ) -> None:
        """
        解析結果をJSONファイルにエクスポート

        Args:
            temporal_distribution: 時系列視線分布データ
            output_path: 出力ファイルパス
        """
        export_data = {"temporal_distribution": {}}

        for timestamp, vertex_probs in temporal_distribution.items():
            export_data["temporal_distribution"][str(timestamp)] = [
                {
                    "vertex_index": vp.vertex_index,
                    "probability": vp.probability,
                    "distance_score": vp.distance_score,
                    "angle_score": vp.angle_score,
                    "temporal_weight": vp.temporal_weight,
                }
                for vp in vertex_probs
            ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def visualize_temporal_heatmap(
        temporal_distribution: Dict[float, List[VertexGazeProbability]],
        mesh_processor: MeshProcessor,
        output_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        時系列視線分布のヒートマップを可視化

        Args:
            temporal_distribution: 時系列視線分布データ
            mesh_processor: メッシュプロセッサー
            output_path: 保存先パス（Noneの場合は表示のみ）
        """
        timestamps = sorted(temporal_distribution.keys())
        num_vertices = len(mesh_processor.vertices)

        # ヒートマップデータの準備
        heatmap_data = np.zeros((len(timestamps), num_vertices))

        for i, timestamp in enumerate(timestamps):
            vertex_probs = temporal_distribution[timestamp]
            for vertex_prob in vertex_probs:
                heatmap_data[i, vertex_prob.vertex_index] = vertex_prob.probability

        # 可視化
        plt.figure(figsize=(12, 8))
        plt.imshow(heatmap_data, aspect="auto", cmap="hot", interpolation="nearest")
        plt.colorbar(label="視線確率")
        plt.xlabel("頂点インデックス")
        plt.ylabel("時刻インデックス")
        plt.title("時系列視線分布ヒートマップ")

        # 時刻ラベルの設定
        time_ticks = np.linspace(
            0, len(timestamps) - 1, min(10, len(timestamps)), dtype=int
        )
        plt.yticks(time_ticks, [f"{timestamps[i]:.1f}s" for i in time_ticks])

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()


def main():
    """
    メイン実行関数 - 使用例のデモンストレーション
    """
    print("視線データ分布解析システムのデモンストレーション")
    print("=" * 50)

    # デモ用のサンプルデータ生成
    print("1. サンプルデータの生成...")

    # サンプル視線データの生成
    sample_gaze_data = []
    for i in range(100):
        timestamp = i * 0.1
        # ランダムな視線方向
        direction = np.random.normal(0, 1, 3)
        direction = direction / np.linalg.norm(direction)

        # カメラ位置
        origin = np.array([0, 0, 2])

        gaze_point = GazePoint(
            timestamp=timestamp,
            direction=direction,
            origin=origin,
            confidence=np.random.uniform(0.8, 1.0),
        )
        sample_gaze_data.append(gaze_point)

    # サンプルメッシュの生成（球体）
    sphere = trimesh.creation.icosphere(subdivisions=2)
    mesh_processor = MeshProcessor(sphere)

    print(f"   - 視線データポイント数: {len(sample_gaze_data)}")
    print(f"   - メッシュ頂点数: {len(mesh_processor.vertices)}")

    # 解析の実行
    print("\n2. 視線分布解析の実行...")
    analyzer = GazeDistributionAnalyzer(mesh_processor)

    temporal_distribution = analyzer.analyze_temporal_distribution(
        sample_gaze_data, time_step=0.5
    )

    print(f"   - 解析時刻数: {len(temporal_distribution)}")

    # 結果のエクスポート
    print("\n3. 結果のエクスポート...")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # CSV出力
    ResultExporter.export_to_csv(
        temporal_distribution, output_dir / "gaze_distribution.csv"
    )

    # JSON出力
    ResultExporter.export_to_json(
        temporal_distribution, output_dir / "gaze_distribution.json"
    )

    print(f"   - CSV出力: {output_dir / 'gaze_distribution.csv'}")
    print(f"   - JSON出力: {output_dir / 'gaze_distribution.json'}")

    # 可視化
    print("\n4. 結果の可視化...")
    ResultExporter.visualize_temporal_heatmap(
        temporal_distribution, mesh_processor, output_dir / "gaze_heatmap.png"
    )

    print(f"   - ヒートマップ: {output_dir / 'gaze_heatmap.png'}")

    print("\n解析完了！")


if __name__ == "__main__":
    main()
