#!/usr/bin/env python3
"""
T-pose メッシュ視線可視化システム

T-poseのOBJファイルに各フレームの視線データを適用して、
視線の強度をカラーマップで可視化するシステムです。
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import trimesh
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import requests


class GazeMeshVisualizer:
    """T-poseメッシュに視線データを可視化するクラス"""
    
    def __init__(self, 
                 obj_path: Union[str, Path],
                 segmentation_file: str = "smpl_vert_segmentation.json"):
        """
        初期化
        
        Args:
            obj_path: T-poseのOBJファイルパス
            segmentation_file: SMPLセグメンテーションファイルパス
        """
        self.obj_path = Path(obj_path)
        self.segmentation_file = segmentation_file
        self.base_mesh: Optional[trimesh.Trimesh] = None
        self.segmentation_data: Optional[Dict[str, List[int]]] = None
        self.gaze_data: Dict[int, np.ndarray] = {}
        
        # カラーマップの設定
        self.colormap = cm.get_cmap('hot')  # 'hot', 'viridis', 'plasma'等
        self.default_color = [0.7, 0.7, 0.7, 1.0]  # グレー
        
    def load_base_mesh(self) -> trimesh.Trimesh:
        """
        ベースとなるT-poseメッシュを読み込み
        
        Returns:
            読み込まれたメッシュ
        """
        if not self.obj_path.exists():
            raise FileNotFoundError(f"OBJファイルが見つかりません: {self.obj_path}")
        
        print(f"T-poseメッシュを読み込み: {self.obj_path}")
        mesh = trimesh.load(self.obj_path, process=False)
        
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"有効なメッシュファイルではありません: {self.obj_path}")
        
        # SMPLモデルの頂点数チェック
        if len(mesh.vertices) != 6890:
            print(f"警告: 頂点数が予期した値と異なります: {len(mesh.vertices)} (期待値: 6890)")
        
        print(f"メッシュ情報: 頂点数={len(mesh.vertices)}, 面数={len(mesh.faces)}")
        self.base_mesh = mesh
        return mesh
    
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
        
        print(f"セグメンテーションファイルをダウンロード: {url}")
        response = requests.get(url)
        response.raise_for_status()
        
        seg_data = response.json()
        with open(self.segmentation_file, "w") as f:
            json.dump(seg_data, f, indent=2)
        
        print(f"セグメンテーションファイルを保存: {self.segmentation_file}")
        return seg_data
    
    def load_gaze_data(self, data_folder: Path) -> Dict[int, np.ndarray]:
        """
        視線データフォルダから.npyファイルを読み込み
        
        Args:
            data_folder: 視線データフォルダ
            
        Returns:
            フレーム番号をキーとした視線データ辞書
        """
        if not data_folder.exists():
            raise FileNotFoundError(f"データフォルダが見つかりません: {data_folder}")
        
        npy_files = list(data_folder.glob("*.npy"))
        if not npy_files:
            raise ValueError(f"npyファイルが見つかりません: {data_folder}")
        
        print(f"{len(npy_files)}個の視線データファイルを発見")
        
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
                
                # データ形状の正規化
                if len(data.shape) == 2 and data.shape[1] == 1:
                    data = data.flatten()
                elif len(data.shape) != 1:
                    print(f"警告: 予期しないデータ形状: {data.shape} ({npy_file.name})")
                    continue
                
                # 頂点数チェック
                if self.base_mesh is not None and len(data) != len(self.base_mesh.vertices):
                    print(f"警告: 頂点数不一致: データ={len(data)}, メッシュ={len(self.base_mesh.vertices)} ({npy_file.name})")
                    # 最小値に合わせて切り詰め
                    min_vertices = min(len(data), len(self.base_mesh.vertices))
                    data = data[:min_vertices]
                
                gaze_data[frame_num] = data
                
            except Exception as e:
                print(f"エラー: ファイル読み込み失敗 {npy_file.name}: {e}")
                continue
        
        if not gaze_data:
            raise ValueError("有効な視線データが見つかりませんでした")
        
        print(f"{len(gaze_data)}フレームの視線データを読み込み完了")
        self.gaze_data = gaze_data
        return gaze_data
    
    def normalize_gaze_values(self, 
                             gaze_values: np.ndarray,
                             method: str = "global",
                             vmin: Optional[float] = None,
                             vmax: Optional[float] = None) -> np.ndarray:
        """
        視線値を正規化
        
        Args:
            gaze_values: 視線値配列
            method: 正規化方法 ("global", "frame", "percentile")
            vmin: 最小値（指定時）
            vmax: 最大値（指定時）
            
        Returns:
            正規化された視線値（0-1範囲）
        """
        if method == "global":
            # 全フレーム通しての最小・最大値で正規化
            if vmin is None or vmax is None:
                all_values = np.concatenate(list(self.gaze_data.values()))
                vmin = np.min(all_values) if vmin is None else vmin
                vmax = np.max(all_values) if vmax is None else vmax
        elif method == "frame":
            # フレーム内での最小・最大値で正規化
            vmin = np.min(gaze_values) if vmin is None else vmin
            vmax = np.max(gaze_values) if vmax is None else vmax
        elif method == "percentile":
            # パーセンタイル基準の正規化
            vmin = np.percentile(gaze_values, 5) if vmin is None else vmin
            vmax = np.percentile(gaze_values, 95) if vmax is None else vmax
        
        # 正規化実行
        if vmax > vmin:
            normalized = (gaze_values - vmin) / (vmax - vmin)
            # 0-1範囲にクリップ
            normalized = np.clip(normalized, 0.0, 1.0)
        else:
            normalized = np.zeros_like(gaze_values)
            
        return normalized
    
    def apply_gaze_colors(self, 
                         gaze_values: np.ndarray,
                         normalization: str = "global",
                         colormap: str = "hot",
                         alpha: float = 1.0) -> np.ndarray:
        """
        視線値をメッシュの頂点色として適用
        
        Args:
            gaze_values: 視線値配列
            normalization: 正規化方法
            colormap: カラーマップ名
            alpha: 透明度
            
        Returns:
            頂点色配列 (N, 4)
        """
        if self.base_mesh is None:
            raise ValueError("ベースメッシュが読み込まれていません")
        
        # 視線値の正規化
        normalized_values = self.normalize_gaze_values(gaze_values, method=normalization)
        
        # カラーマップの適用
        cmap = cm.get_cmap(colormap)
        colors = cmap(normalized_values)
        
        # アルファ値の設定
        colors[:, 3] = alpha
        
        # 視線値が0の頂点はデフォルト色（グレー）に設定
        zero_mask = gaze_values == 0
        colors[zero_mask] = self.default_color
        
        return colors
    
    def create_gaze_mesh(self, 
                        frame_num: int,
                        normalization: str = "global",
                        colormap: str = "hot",
                        alpha: float = 1.0) -> trimesh.Trimesh:
        """
        指定フレームの視線データを適用したメッシュを作成
        
        Args:
            frame_num: フレーム番号
            normalization: 正規化方法
            colormap: カラーマップ名
            alpha: 透明度
            
        Returns:
            視線色付きメッシュ
        """
        if self.base_mesh is None:
            raise ValueError("ベースメッシュが読み込まれていません")
        
        if frame_num not in self.gaze_data:
            raise ValueError(f"フレーム {frame_num} の視線データが見つかりません")
        
        # ベースメッシュをコピー
        mesh = self.base_mesh.copy()
        
        # 視線データを取得
        gaze_values = self.gaze_data[frame_num]
        
        # 頂点数の調整
        vertex_count = len(mesh.vertices)
        if len(gaze_values) > vertex_count:
            gaze_values = gaze_values[:vertex_count]
        elif len(gaze_values) < vertex_count:
            # 不足分をゼロで埋める
            padded_values = np.zeros(vertex_count)
            padded_values[:len(gaze_values)] = gaze_values
            gaze_values = padded_values
        
        # 視線色を適用
        vertex_colors = self.apply_gaze_colors(
            gaze_values, 
            normalization=normalization,
            colormap=colormap,
            alpha=alpha
        )
        
        # メッシュに色を設定
        mesh.visual.vertex_colors = vertex_colors
        
        return mesh
    
    def save_gaze_mesh(self, 
                      frame_num: int,
                      output_path: Path,
                      normalization: str = "global",
                      colormap: str = "hot") -> Path:
        """
        視線付きメッシュをファイルに保存
        
        Args:
            frame_num: フレーム番号
            output_path: 出力ファイルパス
            normalization: 正規化方法
            colormap: カラーマップ名
            
        Returns:
            保存されたファイルパス
        """
        mesh = self.create_gaze_mesh(
            frame_num=frame_num,
            normalization=normalization,
            colormap=colormap
        )
        
        # 出力ディレクトリの作成
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # メッシュの保存
        mesh.export(output_path)
        
        print(f"視線メッシュを保存: {output_path}")
        return output_path
    
    def create_gaze_visualization(self, 
                                 frame_num: int,
                                 save_path: Optional[Path] = None,
                                 view_angle: Tuple[float, float] = (45, 45),
                                 resolution: Tuple[int, int] = (800, 600),
                                 colormap: str = "hot") -> Optional[Path]:
        """
        視線メッシュの画像可視化
        
        Args:
            frame_num: フレーム番号
            save_path: 保存パス（Noneの場合は表示のみ）
            view_angle: 視点角度 (elevation, azimuth)
            resolution: 画像解像度
            colormap: カラーマップ名
            
        Returns:
            保存されたファイルパス（保存した場合）
        """
        mesh = self.create_gaze_mesh(frame_num=frame_num, colormap=colormap)
        
        try:
            # シーンの作成
            scene = mesh.scene()
            
            # 視点の設定（trimeshのバージョンに応じて調整）
            try:
                # 新しいバージョンのtrimesh
                scene.set_camera(angles=view_angle, distance=2.0)
            except TypeError:
                # 古いバージョンのtrimesh
                elevation, azimuth = view_angle
                scene.set_camera(
                    angles=[elevation, azimuth, 0], 
                    distance=2.0
                )
            
            # 画像のレンダリング
            try:
                image = scene.save_image(resolution=resolution)
            except Exception as e:
                print(f"画像レンダリングエラー: {e}")
                # matplotlib を使用したフォールバック可視化
                return self._create_matplotlib_visualization(mesh, save_path, resolution)
            
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'wb') as f:
                    f.write(image)
                print(f"視線可視化画像を保存: {save_path}")
                return save_path
            else:
                # 画像を表示（Jupyter環境等）
                try:
                    from PIL import Image
                    import io
                    Image.open(io.BytesIO(image)).show()
                except ImportError:
                    print("PIL (Pillow) がインストールされていません。画像表示をスキップします。")
                return None
                
        except Exception as e:
            print(f"3D可視化エラー: {e}")
            # matplotlib フォールバック
            return self._create_matplotlib_visualization(mesh, save_path, resolution)
    
    def _create_matplotlib_visualization(self, 
                                       mesh: trimesh.Trimesh,
                                       save_path: Optional[Path] = None,
                                       resolution: Tuple[int, int] = (800, 600)) -> Optional[Path]:
        """
        matplotlib を使用したフォールバック可視化
        
        Args:
            mesh: 可視化するメッシュ
            save_path: 保存パス
            resolution: 画像解像度
            
        Returns:
            保存されたファイルパス（保存した場合）
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(resolution[0]/100, resolution[1]/100))
            ax = fig.add_subplot(111, projection='3d')
            
            # 頂点の3D散布図
            vertices = mesh.vertices
            colors = mesh.visual.vertex_colors[:, :3] / 255.0  # RGB正規化
            
            scatter = ax.scatter(
                vertices[:, 0], 
                vertices[:, 1], 
                vertices[:, 2],
                c=colors,
                s=1,
                alpha=0.8
            )
            
            # 軸の設定
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('視線強度可視化 (3D点群)')
            
            # 視点の調整
            ax.view_init(elev=30, azim=45)
            
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"matplotlib可視化画像を保存: {save_path}")
                return save_path
            else:
                plt.show()
                return None
                
        except Exception as e:
            print(f"matplotlib可視化エラー: {e}")
            return None
    
    def create_frame_sequence(self, 
                             output_dir: Path,
                             frame_range: Optional[Tuple[int, int]] = None,
                             normalization: str = "global",
                             colormap: str = "hot",
                             file_format: str = "obj") -> List[Path]:
        """
        複数フレームの視線メッシュを一括生成
        
        Args:
            output_dir: 出力ディレクトリ
            frame_range: フレーム範囲 (start, end)、Noneの場合は全フレーム
            normalization: 正規化方法
            colormap: カラーマップ名
            file_format: ファイル形式 ("obj", "ply", "stl")
            
        Returns:
            保存されたファイルパスのリスト
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # フレーム範囲の決定
        if frame_range is None:
            frames = sorted(self.gaze_data.keys())
        else:
            start, end = frame_range
            frames = [f for f in sorted(self.gaze_data.keys()) if start <= f <= end]
        
        print(f"{len(frames)}フレームの視線メッシュを生成中...")
        
        saved_files = []
        for i, frame_num in enumerate(frames):
            # ファイル名の生成
            filename = f"gaze_frame_{frame_num:06d}.{file_format}"
            output_path = output_dir / filename
            
            try:
                # 視線メッシュの保存
                saved_path = self.save_gaze_mesh(
                    frame_num=frame_num,
                    output_path=output_path,
                    normalization=normalization,
                    colormap=colormap
                )
                saved_files.append(saved_path)
                
                # 進捗表示
                if (i + 1) % 10 == 0 or i == len(frames) - 1:
                    print(f"進捗: {i + 1}/{len(frames)} フレーム完了")
                    
            except Exception as e:
                print(f"エラー: フレーム {frame_num} の処理に失敗: {e}")
                continue
        
        print(f"{len(saved_files)}個のファイルを保存完了")
        return saved_files
    
    def create_colorbar_legend(self, 
                              save_path: Path,
                              colormap: str = "hot",
                              title: str = "視線強度") -> Path:
        """
        カラーバー凡例を作成
        
        Args:
            save_path: 保存パス
            colormap: カラーマップ名
            title: タイトル
            
        Returns:
            保存されたファイルパス
        """
        # 全フレームの視線値範囲を計算
        all_values = np.concatenate(list(self.gaze_data.values()))
        vmin, vmax = np.min(all_values), np.max(all_values)
        
        # カラーバーの作成
        fig, ax = plt.subplots(figsize=(8, 2))
        
        # 正規化とカラーマップ
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap(colormap)
        
        # カラーバーの描画
        cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), 
                         ax=ax, orientation='horizontal')
        cb.set_label(title, fontsize=12)
        
        # 軸を非表示
        ax.set_visible(False)
        
        # 保存
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"カラーバー凡例を保存: {save_path}")
        return save_path
    
    def get_gaze_statistics(self) -> Dict[str, Any]:
        """
        視線データの統計情報を取得
        
        Returns:
            統計情報辞書
        """
        if not self.gaze_data:
            return {}
        
        all_values = np.concatenate(list(self.gaze_data.values()))
        
        stats = {
            "total_frames": len(self.gaze_data),
            "total_vertices": len(self.base_mesh.vertices) if self.base_mesh else None,
            "gaze_range": {
                "min": float(np.min(all_values)),
                "max": float(np.max(all_values)),
                "mean": float(np.mean(all_values)),
                "std": float(np.std(all_values))
            },
            "non_zero_ratio": float(np.count_nonzero(all_values) / len(all_values)),
            "frame_range": {
                "start": min(self.gaze_data.keys()),
                "end": max(self.gaze_data.keys())
            }
        }
        
        return stats
    
    def run_visualization(self, 
                         data_folder: Path,
                         output_dir: Path = Path("output/gaze_meshes"),
                         normalization: str = "global",
                         colormap: str = "hot",
                         create_images: bool = True,
                         create_legend: bool = True) -> Dict[str, Any]:
        """
        完全な視線可視化処理を実行
        
        Args:
            data_folder: 視線データフォルダ
            output_dir: 出力ディレクトリ
            normalization: 正規化方法
            colormap: カラーマップ名
            create_images: 画像出力するか
            create_legend: 凡例を作成するか
            
        Returns:
            処理結果の辞書
        """
        print("視線メッシュ可視化処理を開始")
        print(f"T-poseメッシュ: {self.obj_path}")
        print(f"視線データフォルダ: {data_folder}")
        print(f"出力ディレクトリ: {output_dir}")
        print("-" * 50)
        
        try:
            # 1. ベースメッシュの読み込み
            self.load_base_mesh()
            
            # 2. 視線データの読み込み
            self.load_gaze_data(data_folder)
            
            # 3. セグメンテーションデータの読み込み
            self.segmentation_data = self.download_segmentation_file()
            
            # 4. フレーム別メッシュの生成
            mesh_files = self.create_frame_sequence(
                output_dir=output_dir / "meshes",
                normalization=normalization,
                colormap=colormap
            )
            
            result = {
                "mesh_files": mesh_files,
                "statistics": self.get_gaze_statistics()
            }
            
            # 5. 画像出力（オプション）
            if create_images:
                image_dir = output_dir / "images"
                image_files = []
                
                print("\n視線可視化画像を生成中...")
                for i, frame_num in enumerate(sorted(self.gaze_data.keys())[:10]):  # 最初の10フレーム
                    image_path = image_dir / f"gaze_frame_{frame_num:06d}.png"
                    try:
                        self.create_gaze_visualization(
                            frame_num=frame_num,
                            save_path=image_path,
                            colormap=colormap
                        )
                        image_files.append(image_path)
                    except Exception as e:
                        print(f"画像生成エラー (フレーム {frame_num}): {e}")
                
                result["image_files"] = image_files
            
            # 6. カラーバー凡例の作成（オプション）
            if create_legend:
                legend_path = output_dir / "colorbar_legend.png"
                self.create_colorbar_legend(
                    save_path=legend_path,
                    colormap=colormap
                )
                result["legend_file"] = legend_path
            
            print(f"\n視線可視化処理が完了しました!")
            print(f"出力ファイル数: {len(mesh_files)}個のメッシュ")
            if create_images:
                print(f"             {len(result.get('image_files', []))}個の画像")
            
            return result
            
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            raise