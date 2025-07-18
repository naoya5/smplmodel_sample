#!/usr/bin/env python3
"""
シンプル視線ヒートマップ可視化

既存のtpose_mesh.objを使用して視線データをヒートマップで可視化する
uv run src/simple_gaze_heatmap.py data/User11/ tpose_mesh.obj -o output/heatmap --colormap cool
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import re

import numpy as np
import trimesh
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap


class SimpleGazeHeatmapVisualizer:
    """シンプルな視線ヒートマップ可視化クラス"""

    def __init__(self, tpose_mesh_path: Union[str, Path]):
        """
        初期化

        Args:
            tpose_mesh_path: T-poseメッシュファイルのパス
        """
        self.tpose_mesh_path = Path(tpose_mesh_path)
        self.gaze_data: Dict[int, np.ndarray] = {}
        self.accumulated_gaze: Optional[np.ndarray] = None

    def load_tpose_mesh(self) -> trimesh.Trimesh:
        """
        T-poseメッシュを読み込み

        Returns:
            T-poseメッシュ
        """
        if not self.tpose_mesh_path.exists():
            raise FileNotFoundError(
                f"T-poseメッシュファイルが見つかりません: {self.tpose_mesh_path}"
            )

        mesh = trimesh.load(str(self.tpose_mesh_path))

        if not isinstance(mesh, trimesh.Trimesh):
            if hasattr(mesh, "geometry") and len(mesh.geometry) > 0:
                mesh = list(mesh.geometry.values())[0]
            else:
                raise ValueError("有効なメッシュが見つかりません")

        print(
            f"T-poseメッシュを読み込み: {len(mesh.vertices)}頂点, {len(mesh.faces)}面"
        )
        return mesh

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
            frame_match = re.search(r"(\d+)", npy_file.stem)
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

    def accumulate_gaze_data(self, normalize: bool = True) -> np.ndarray:
        """
        全フレームの視線データを累積

        Args:
            normalize: データを正規化するか

        Returns:
            累積視線データ (6890,)
        """
        if not self.gaze_data:
            raise ValueError("視線データが読み込まれていません")

        print("全フレームの視線データを累積中...")

        # 6890頂点分の累積配列を初期化
        accumulated = np.zeros(6890)

        for frame_num, gaze_values in self.gaze_data.items():
            accumulated += gaze_values

        if normalize:
            # 0-1範囲に正規化
            max_val = np.max(accumulated)
            if max_val > 0:
                accumulated = accumulated / max_val

        print(
            f"累積完了: 最大値={np.max(accumulated):.6f}, 平均値={np.mean(accumulated):.6f}"
        )

        self.accumulated_gaze = accumulated
        return accumulated

    def create_heatmap_colormap(
        self, colormap_name: str = "hot"
    ) -> LinearSegmentedColormap:
        """
        ヒートマップ用のカラーマップを作成

        Args:
            colormap_name: ベースとなるカラーマップ名

        Returns:
            カスタムカラーマップ
        """
        if colormap_name == "custom_blue_red":
            # カスタム青-赤カラーマップ
            colors = ["darkblue", "blue", "cyan", "yellow", "orange", "red", "darkred"]
            n_bins = 256
            cmap = LinearSegmentedColormap.from_list(
                "custom_blue_red", colors, N=n_bins
            )
        else:
            cmap = plt.get_cmap(colormap_name)

        return cmap

    def apply_heatmap_to_mesh(
        self,
        mesh: trimesh.Trimesh,
        gaze_values: np.ndarray,
        colormap: str = "hot",
        alpha: float = 1.0,
        min_threshold: float = 0.0,
        base_color: tuple = (0.7, 0.7, 0.7, 1.0),
    ) -> trimesh.Trimesh:
        """
        メッシュに視線ヒートマップを適用

        Args:
            mesh: 対象メッシュ
            gaze_values: 頂点ごとの視線値 (6890,)
            colormap: カラーマップ名
            alpha: 透明度
            min_threshold: 最小閾値（この値以下は基本色のまま）
            base_color: 基本色（灰色）

        Returns:
            ヒートマップ適用済みメッシュ
        """
        cmap = self.create_heatmap_colormap(colormap)

        # 頂点数の確認
        if len(gaze_values) != len(mesh.vertices):
            raise ValueError(
                f"視線データ数({len(gaze_values)})とメッシュ頂点数({len(mesh.vertices)})が一致しません"
            )

        # 基本色（灰色）で全頂点を初期化
        colors = np.tile(base_color, (len(mesh.vertices), 1))

        # 閾値を超える頂点のマスク
        above_threshold_mask = gaze_values > min_threshold

        if np.any(above_threshold_mask):
            # 閾値を超える部分のみヒートマップ色を適用
            heatmap_colors = cmap(gaze_values[above_threshold_mask])

            # アルファ値の調整
            heatmap_colors[:, 3] *= alpha

            # 閾値を超える頂点にヒートマップ色を適用
            colors[above_threshold_mask] = heatmap_colors

        # メッシュに色を適用
        mesh_copy = mesh.copy()
        mesh_copy.visual.vertex_colors = (colors * 255).astype(np.uint8)

        return mesh_copy

    def create_multiple_views(
        self,
        mesh: trimesh.Trimesh,
        output_dir: Path,
        base_name: str = "heatmap",
        views: Optional[List[str]] = None,
        resolution: tuple = (800, 800),
    ) -> List[Path]:
        """
        複数視点からのヒートマップ画像を生成

        Args:
            mesh: ヒートマップ適用済みメッシュ
            output_dir: 出力ディレクトリ
            base_name: ファイル名のベース
            views: 視点リスト
            resolution: 画像解像度

        Returns:
            生成された画像ファイルのパスリスト
        """
        if views is None:
            views = ["front", "back", "left", "right"]

        output_dir.mkdir(parents=True, exist_ok=True)
        image_paths = []

        # 視点の定義
        view_angles = {
            "front": (0, 0, 0),
            "back": (0, 0, 180),
            "left": (0, 0, 90),
            "right": (0, 0, -90),
            "top": (90, 0, 0),
            "bottom": (-90, 0, 0),
        }

        scene = mesh.scene()

        for view_name in views:
            if view_name not in view_angles:
                print(f"警告: 未知の視点名 '{view_name}' をスキップ")
                continue

            # カメラ位置の設定
            angles = view_angles[view_name]

            # シーンの回転
            scene_copy = scene.copy()

            # 回転行列の適用
            rx, ry, rz = np.radians(angles)

            # X軸回転
            if rx != 0:
                rotation_x = trimesh.transformations.rotation_matrix(rx, [1, 0, 0])
                scene_copy.apply_transform(rotation_x)

            # Y軸回転
            if ry != 0:
                rotation_y = trimesh.transformations.rotation_matrix(ry, [0, 1, 0])
                scene_copy.apply_transform(rotation_y)

            # Z軸回転
            if rz != 0:
                rotation_z = trimesh.transformations.rotation_matrix(rz, [0, 0, 1])
                scene_copy.apply_transform(rotation_z)

            # 画像をレンダリング
            try:
                png_data = scene_copy.save_image(resolution=resolution)

                # ファイルに保存
                image_path = output_dir / f"{base_name}_{view_name}.png"
                with open(image_path, "wb") as f:
                    f.write(png_data)

                image_paths.append(image_path)
                print(f"画像を保存: {image_path}")

            except Exception as e:
                print(f"警告: {view_name}視点の画像生成に失敗: {e}")
                continue

        return image_paths

    def save_heatmap_mesh(
        self, mesh: trimesh.Trimesh, output_path: Path, file_format: str = "obj"
    ) -> Path:
        """
        ヒートマップメッシュをファイルに保存

        Args:
            mesh: 保存するメッシュ
            output_path: 出力パス
            file_format: ファイル形式

        Returns:
            保存されたファイルのパス
        """
        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(f".{file_format}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        mesh.export(str(output_path))
        print(f"ヒートマップメッシュを保存: {output_path}")

        return output_path

    def create_colorbar_legend(
        self,
        output_dir: Path,
        colormap: str = "hot",
        max_value: float = 1.0,
        min_value: float = 0.0,
        title: str = "視線強度",
        figsize: tuple = (8, 2),
    ) -> Path:
        """
        カラーバー凡例を作成

        Args:
            output_dir: 出力ディレクトリ
            colormap: カラーマップ名
            max_value: 最大値
            min_value: 最小値
            title: タイトル
            figsize: 図のサイズ

        Returns:
            凡例画像のパス
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=figsize)

        cmap = self.create_heatmap_colormap(colormap)

        # カラーバーの作成
        gradient = np.linspace(min_value, max_value, 256).reshape(1, -1)
        ax.imshow(
            gradient, aspect="auto", cmap=cmap, extent=[min_value, max_value, 0, 1]
        )

        ax.set_xlim(min_value, max_value)
        ax.set_yticks([])
        ax.set_xlabel(title)
        ax.set_title("ヒートマップ強度スケール")

        plt.tight_layout()

        legend_path = output_dir / "heatmap_colorbar.png"
        plt.savefig(legend_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"カラーバー凡例を保存: {legend_path}")
        return legend_path

    def visualize_gaze_heatmap(
        self,
        data_folder: Path,
        output_dir: Path = Path("heatmap_output"),
        colormap: str = "hot",
        views: Optional[List[str]] = None,
        save_mesh: bool = True,
        alpha: float = 0.8,
        min_threshold: float = 0.01,
        resolution: tuple = (1200, 1200),
    ) -> Dict[str, Any]:
        """
        視線ヒートマップの完全な可視化を実行

        Args:
            data_folder: 視線データフォルダ
            output_dir: 出力ディレクトリ
            colormap: カラーマップ名
            views: 表示する視点のリスト
            save_mesh: メッシュファイルを保存するか
            alpha: 透明度
            min_threshold: 最小表示閾値
            resolution: 画像解像度

        Returns:
            実行結果
        """
        print("視線ヒートマップ可視化を開始")
        print(f"T-poseメッシュ: {self.tpose_mesh_path}")
        print(f"データフォルダ: {data_folder}")
        print(f"出力ディレクトリ: {output_dir}")
        print("-" * 50)

        try:
            # 1. 視線データの読み込み
            self.load_gaze_data(data_folder)

            # 2. T-poseメッシュの読み込み
            print("T-poseメッシュを読み込み中...")
            tpose_mesh = self.load_tpose_mesh()

            # 3. 視線データの累積
            accumulated_gaze = self.accumulate_gaze_data(normalize=True)

            # 4. ヒートマップの適用
            print("ヒートマップを適用中...")
            print(f"閾値: {min_threshold} (この値以下は灰色で表示)")
            heatmap_mesh = self.apply_heatmap_to_mesh(
                mesh=tpose_mesh,
                gaze_values=accumulated_gaze,
                colormap=colormap,
                alpha=alpha,
                min_threshold=min_threshold,
            )

            # 5. 出力ディレクトリの作成
            output_dir.mkdir(parents=True, exist_ok=True)

            result = {
                "total_frames": len(self.gaze_data),
                "max_gaze_value": np.max(accumulated_gaze),
                "mean_gaze_value": np.mean(accumulated_gaze),
                "output_files": [],
            }

            # 6. メッシュファイルの保存
            if save_mesh:
                mesh_path = self.save_heatmap_mesh(
                    mesh=heatmap_mesh, output_path=output_dir / "gaze_heatmap_tpose.obj"
                )
                result["output_files"].append(mesh_path)

            # 7. 複数視点の画像生成
            print("複数視点画像を生成中...")
            image_paths = self.create_multiple_views(
                mesh=heatmap_mesh,
                output_dir=output_dir,
                base_name="gaze_heatmap",
                views=views,
                resolution=resolution,
            )
            result["output_files"].extend(image_paths)

            # 8. カラーバー凡例の生成
            legend_path = self.create_colorbar_legend(
                output_dir=output_dir,
                colormap=colormap,
                max_value=np.max(accumulated_gaze),
                min_value=0.0,
            )
            result["output_files"].append(legend_path)

            # 9. 統計情報の保存
            stats_path = output_dir / "heatmap_statistics.json"
            stats = {
                "total_frames": len(self.gaze_data),
                "total_vertices": len(accumulated_gaze),
                "max_gaze_value": float(np.max(accumulated_gaze)),
                "min_gaze_value": float(np.min(accumulated_gaze)),
                "mean_gaze_value": float(np.mean(accumulated_gaze)),
                "std_gaze_value": float(np.std(accumulated_gaze)),
                "vertices_above_threshold": int(
                    np.sum(accumulated_gaze > min_threshold)
                ),
                "colormap": colormap,
                "alpha": alpha,
                "min_threshold": min_threshold,
            }

            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            result["output_files"].append(stats_path)

            print(f"\nヒートマップ可視化が完了しました!")
            print(f"出力ファイル数: {len(result['output_files'])}")
            print(f"最大視線強度: {result['max_gaze_value']:.6f}")
            print(f"平均視線強度: {result['mean_gaze_value']:.6f}")

            return result

        except Exception as e:
            print(f"エラーが発生しました: {e}")
            raise


def create_simple_gaze_heatmap(
    data_folder: Union[str, Path],
    tpose_mesh_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    colormap: str = "hot",
    views: Optional[List[str]] = None,
    save_mesh: bool = True,
) -> Dict[str, Any]:
    """
    シンプル視線ヒートマップ作成のユーティリティ関数

    Args:
        data_folder: 視線データフォルダ
        tpose_mesh_path: T-poseメッシュファイルのパス
        output_dir: 出力ディレクトリ
        colormap: カラーマップ名
        views: 表示する視点のリスト
        save_mesh: メッシュファイルを保存するか

    Returns:
        実行結果
    """
    if output_dir is None:
        output_dir = Path(data_folder).parent / "heatmap_output"

    visualizer = SimpleGazeHeatmapVisualizer(tpose_mesh_path=tpose_mesh_path)

    return visualizer.visualize_gaze_heatmap(
        data_folder=Path(data_folder),
        output_dir=Path(output_dir),
        colormap=colormap,
        views=views,
        save_mesh=save_mesh,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="既存T-poseメッシュを使用した視線ヒートマップ可視化"
    )
    parser.add_argument("data_folder", help="視線データ(.npy)が格納されたフォルダ")
    parser.add_argument("tpose_mesh_path", help="T-poseメッシュファイルのパス(.obj)")
    parser.add_argument("-o", "--output", help="出力ディレクトリ")
    parser.add_argument(
        "--colormap",
        default="hot",
        choices=["hot", "viridis", "plasma", "inferno", "cool", "custom_blue_red"],
        help="カラーマップ",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        choices=["front", "back", "left", "right", "top", "bottom"],
        default=["front", "back", "left", "right"],
        help="表示する視点",
    )
    parser.add_argument(
        "--no-mesh", action="store_true", help="メッシュファイル保存をスキップ"
    )
    parser.add_argument("--alpha", type=float, default=0.8, help="ヒートマップの透明度")
    parser.add_argument("--threshold", type=float, default=0.01, help="最小表示閾値")

    args = parser.parse_args()

    try:
        result = create_simple_gaze_heatmap(
            data_folder=args.data_folder,
            tpose_mesh_path=args.tpose_mesh_path,
            output_dir=args.output,
            colormap=args.colormap,
            views=args.views,
            save_mesh=not args.no_mesh,
        )

        print("\n生成されたファイル:")
        for file_path in result["output_files"]:
            print(f"  {file_path}")

    except Exception as e:
        print(f"エラー: {e}")
        import traceback

        traceback.print_exc()
