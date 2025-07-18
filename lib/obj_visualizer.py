"""
OBJファイル可視化ライブラリ

このモジュールはOBJファイルを読み込み、様々な方法で可視化する機能を提供します。
"""

from typing import Optional, Union, Tuple, Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import trimesh
from PIL import Image
import io


class ObjVisualizer:
    """
    OBJファイルの可視化を行うクラス
    """
    
    def __init__(self, obj_path: Union[str, Path]):
        """
        OBJファイルの可視化器を初期化
        
        Args:
            obj_path (Union[str, Path]): OBJファイルのパス
            
        Raises:
            FileNotFoundError: OBJファイルが見つからない場合
            ValueError: OBJファイルの読み込みに失敗した場合
        """
        self.obj_path = Path(obj_path)
        if not self.obj_path.exists():
            raise FileNotFoundError(f"OBJファイルが見つかりません: {self.obj_path}")
        
        self.mesh: Optional[trimesh.Trimesh] = None
        self.original_colors: Optional[np.ndarray] = None
        
    def load_mesh(self) -> trimesh.Trimesh:
        """
        OBJファイルを読み込む
        
        Returns:
            trimesh.Trimesh: 読み込まれたメッシュ
            
        Raises:
            ValueError: メッシュの読み込みに失敗した場合
        """
        try:
            self.mesh = trimesh.load(str(self.obj_path))
            
            if not isinstance(self.mesh, trimesh.Trimesh):
                if hasattr(self.mesh, 'geometry') and len(self.mesh.geometry) > 0:
                    # シーンの場合、最初のメッシュを取得
                    self.mesh = list(self.mesh.geometry.values())[0]
                else:
                    raise ValueError("有効なメッシュが見つかりません")
            
            # 元の色を保存
            if hasattr(self.mesh.visual, 'vertex_colors'):
                self.original_colors = self.mesh.visual.vertex_colors.copy()
            
            return self.mesh
            
        except Exception as e:
            raise ValueError(f"OBJファイルの読み込みに失敗しました: {e}")
    
    def get_mesh_info(self) -> Dict[str, Any]:
        """
        メッシュの基本情報を取得
        
        Returns:
            Dict[str, Any]: メッシュの情報
        """
        if self.mesh is None:
            self.load_mesh()
        
        return {
            'vertices': len(self.mesh.vertices),
            'faces': len(self.mesh.faces),
            'surface_area': self.mesh.area,
            'volume': self.mesh.volume if self.mesh.is_watertight else None,
            'bounds': self.mesh.bounds,
            'center_mass': self.mesh.center_mass,
            'is_watertight': self.mesh.is_watertight,
            'euler_number': self.mesh.euler_number
        }
    
    def apply_vertex_colors(
        self, 
        colors: Union[np.ndarray, List[float]], 
        color_mode: str = "rgb"
    ) -> None:
        """
        頂点に色を適用
        
        Args:
            colors (Union[np.ndarray, List[float]]): 色配列
            color_mode (str): 色のモード ("rgb", "rgba", "scalar")
        """
        if self.mesh is None:
            self.load_mesh()
        
        colors = np.array(colors)
        
        if color_mode == "scalar":
            # スカラー値をカラーマップで変換
            colors = self._scalar_to_colors(colors)
        elif color_mode == "rgb":
            if colors.shape[1] == 3:
                # アルファチャンネルを追加
                alpha = np.ones((len(colors), 1))
                colors = np.hstack([colors, alpha])
        
        # 色の範囲を0-255に正規化
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        
        self.mesh.visual.vertex_colors = colors
    
    def apply_face_colors(
        self, 
        colors: Union[np.ndarray, List[float]], 
        color_mode: str = "rgb"
    ) -> None:
        """
        面に色を適用
        
        Args:
            colors (Union[np.ndarray, List[float]]): 色配列
            color_mode (str): 色のモード ("rgb", "rgba", "scalar")
        """
        if self.mesh is None:
            self.load_mesh()
        
        colors = np.array(colors)
        
        if color_mode == "scalar":
            colors = self._scalar_to_colors(colors)
        elif color_mode == "rgb":
            if colors.shape[1] == 3:
                alpha = np.ones((len(colors), 1))
                colors = np.hstack([colors, alpha])
        
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        
        self.mesh.visual.face_colors = colors
    
    def _scalar_to_colors(
        self, 
        values: np.ndarray, 
        colormap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None
    ) -> np.ndarray:
        """
        スカラー値をカラーマップで色に変換
        
        Args:
            values (np.ndarray): スカラー値
            colormap (str): カラーマップ名
            vmin (Optional[float]): 最小値
            vmax (Optional[float]): 最大値
            
        Returns:
            np.ndarray: RGBA色配列
        """
        if vmin is None:
            vmin = values.min()
        if vmax is None:
            vmax = values.max()
        
        # 正規化
        if vmax > vmin:
            normalized = (values - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(values)
        
        # カラーマップを適用
        cmap = plt.get_cmap(colormap)
        colors = cmap(normalized)
        
        return colors
    
    def create_screenshot(
        self,
        resolution: Tuple[int, int] = (800, 600),
        view_angle: Optional[Tuple[float, float]] = None,
        background_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        wireframe: bool = False
    ) -> np.ndarray:
        """
        メッシュのスクリーンショットを作成
        
        Args:
            resolution (Tuple[int, int]): 画像解像度 (width, height)
            view_angle (Optional[Tuple[float, float]]): 視点角度 (elevation, azimuth)
            background_color (Tuple[float, float, float, float]): 背景色 (RGBA)
            wireframe (bool): ワイヤーフレーム表示
            
        Returns:
            np.ndarray: 画像データ (RGB)
        """
        if self.mesh is None:
            self.load_mesh()
        
        # シーンの作成
        scene = self.mesh.scene()
        
        # カメラ位置の設定
        if view_angle is not None:
            try:
                elevation, azimuth = view_angle
                # 球座標からカメラ位置を計算
                distance = np.linalg.norm(self.mesh.bounds[1] - self.mesh.bounds[0]) * 2
                camera_pos = self._spherical_to_cartesian(distance, elevation, azimuth)
                camera_pos += self.mesh.center_mass
                
                # カメラの設定（trimeshバージョンに対応）
                try:
                    # 新しいAPIを試行
                    camera_transform = trimesh.scene.cameras.look_at(
                        points=[camera_pos],
                        fov=scene.camera.fov
                    )
                    if isinstance(camera_transform, list):
                        scene.camera.transform = camera_transform[0]
                    else:
                        scene.camera.transform = camera_transform
                except TypeError:
                    # 古いAPIにフォールバック
                    from trimesh.transformations import translation_matrix
                    
                    # 簡単な変換行列を設定
                    transform = translation_matrix(camera_pos)
                    scene.camera.transform = transform
            except Exception:
                # カメラ設定に失敗した場合はデフォルトを使用
                pass
        
        # レンダリング設定
        try:
            # trimeshのバージョンによって異なるメソッド名
            if hasattr(scene, 'set_background'):
                scene.set_background(background_color)
            elif hasattr(scene, 'background'):
                scene.background = background_color
        except Exception:
            # 背景設定に失敗した場合は続行
            pass
        
        try:
            # スクリーンショットを撮影
            if hasattr(scene, 'save_image'):
                image_data = scene.save_image(resolution=resolution)
            else:
                # 古いAPIの場合
                image_data = scene.save_image(resolution=resolution, visible=True)
            
            # PILイメージをnumpy配列に変換
            if isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
                return np.array(image)
            elif hasattr(image_data, 'convert'):
                # PILイメージの場合
                return np.array(image_data.convert('RGB'))
            else:
                # 既にnumpy配列の場合
                return np.array(image_data)
            
        except Exception:
            # フォールバック: matplotlibを使用
            return self._create_matplotlib_screenshot(
                resolution, view_angle, wireframe
            )
    
    def _spherical_to_cartesian(
        self, 
        distance: float, 
        elevation: float, 
        azimuth: float
    ) -> np.ndarray:
        """
        球座標をデカルト座標に変換
        
        Args:
            distance (float): 距離
            elevation (float): 仰角（度）
            azimuth (float): 方位角（度）
            
        Returns:
            np.ndarray: デカルト座標 (x, y, z)
        """
        elevation_rad = np.radians(elevation)
        azimuth_rad = np.radians(azimuth)
        
        x = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = distance * np.sin(elevation_rad)
        
        return np.array([x, y, z])
    
    def _create_matplotlib_screenshot(
        self,
        resolution: Tuple[int, int],
        view_angle: Optional[Tuple[float, float]],
        wireframe: bool
    ) -> np.ndarray:
        """
        Matplotlibを使用してスクリーンショットを作成（フォールバック）
        
        Args:
            resolution (Tuple[int, int]): 解像度
            view_angle (Optional[Tuple[float, float]]): 視点角度
            wireframe (bool): ワイヤーフレーム表示
            
        Returns:
            np.ndarray: 画像データ
        """
        fig = plt.figure(figsize=(resolution[0]/100, resolution[1]/100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        vertices = self.mesh.vertices
        faces = self.mesh.faces
        
        if wireframe:
            # ワイヤーフレーム表示
            for face in faces:
                triangle = vertices[face]
                triangle = np.vstack([triangle, triangle[0]])  # 三角形を閉じる
                ax.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], 'k-', alpha=0.3)
        else:
            # ソリッド表示
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            
            colors = None
            if hasattr(self.mesh.visual, 'vertex_colors'):
                # 頂点色から面色を計算
                vertex_colors = self.mesh.visual.vertex_colors
                face_colors = vertex_colors[faces].mean(axis=1)
                colors = face_colors / 255.0 if face_colors.max() > 1 else face_colors
            
            poly3d = Poly3DCollection(vertices[faces], alpha=0.8, facecolors=colors)
            ax.add_collection3d(poly3d)
        
        # 視点の設定
        if view_angle is not None:
            ax.view_init(elev=view_angle[0], azim=view_angle[1])
        
        # 軸の設定
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # 等軸設定
        bounds = self.mesh.bounds
        max_range = np.max(bounds[1] - bounds[0]) / 2
        center = self.mesh.center_mass
        ax.set_xlim([center[0] - max_range, center[0] + max_range])
        ax.set_ylim([center[1] - max_range, center[1] + max_range])
        ax.set_zlim([center[2] - max_range, center[2] + max_range])
        
        # 画像データを取得（より安全な方法）
        import tempfile
        import os
        
        try:
            # 一時ファイルに保存して読み込む方法（最も確実）
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                fig.savefig(tmp.name, format='png', dpi=100, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                tmp.flush()
                
                # 画像を読み込み
                image = Image.open(tmp.name)
                buf = np.array(image.convert('RGB'))
                
                # 一時ファイルを削除
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass
                
                plt.close(fig)
                return buf
                
        except Exception:
            # さらなるフォールバック: メモリ上でバッファを使用
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            image = Image.open(buffer)
            buf = np.array(image.convert('RGB'))
            buffer.close()
        
        plt.close(fig)
        return buf
    
    def save_image(
        self,
        output_path: Union[str, Path],
        resolution: Tuple[int, int] = (800, 600),
        view_angle: Optional[Tuple[float, float]] = None,
        **kwargs
    ) -> Path:
        """
        メッシュの画像を保存
        
        Args:
            output_path (Union[str, Path]): 出力パス
            resolution (Tuple[int, int]): 解像度
            view_angle (Optional[Tuple[float, float]]): 視点角度
            **kwargs: create_screenshotの追加引数
            
        Returns:
            Path: 保存されたファイルのパス
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        image_data = self.create_screenshot(
            resolution=resolution,
            view_angle=view_angle,
            **kwargs
        )
        
        if isinstance(image_data, np.ndarray):
            image = Image.fromarray(image_data)
        else:
            image = image_data
        
        image.save(output_path)
        return output_path
    
    def save_mesh(
        self,
        output_path: Union[str, Path],
        file_format: Optional[str] = None
    ) -> Path:
        """
        メッシュファイルを保存
        
        Args:
            output_path (Union[str, Path]): 出力パス
            file_format (Optional[str]): ファイル形式（自動判定される場合はNone）
            
        Returns:
            Path: 保存されたファイルのパス
        """
        if self.mesh is None:
            self.load_mesh()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.mesh.export(str(output_path), file_type=file_format)
        return output_path
    
    def create_colorbar_legend(
        self,
        values_range: Tuple[float, float],
        colormap: str = "viridis",
        title: str = "値",
        output_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[float, float] = (2, 8)
    ) -> Optional[Path]:
        """
        カラーバー凡例を作成
        
        Args:
            values_range (Tuple[float, float]): 値の範囲 (min, max)
            colormap (str): カラーマップ名
            title (str): タイトル
            output_path (Optional[Union[str, Path]]): 出力パス
            figsize (Tuple[float, float]): 図のサイズ
            
        Returns:
            Optional[Path]: 保存されたファイルのパス
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # カラーマップの作成
        cmap = plt.get_cmap(colormap)
        norm = mcolors.Normalize(vmin=values_range[0], vmax=values_range[1])
        
        # カラーバーの作成
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=ax,
            orientation='vertical'
        )
        cbar.set_label(title, fontsize=12)
        
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            return output_path
        else:
            plt.show()
            return None
    
    def reset_colors(self) -> None:
        """
        メッシュの色を元に戻す
        """
        if self.mesh is None:
            self.load_mesh()
        
        if self.original_colors is not None:
            self.mesh.visual.vertex_colors = self.original_colors.copy()
        else:
            # デフォルト色（グレー）に設定
            default_color = np.array([128, 128, 128, 255], dtype=np.uint8)
            colors = np.tile(default_color, (len(self.mesh.vertices), 1))
            self.mesh.visual.vertex_colors = colors


def quick_visualize(
    obj_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    resolution: Tuple[int, int] = (800, 600),
    view_angle: Tuple[float, float] = (45, 45),
    colormap: Optional[str] = None,
    vertex_values: Optional[np.ndarray] = None,
    **kwargs
) -> Optional[Path]:
    """
    OBJファイルを簡単に可視化するユーティリティ関数
    
    Args:
        obj_path (Union[str, Path]): OBJファイルのパス
        output_path (Optional[Union[str, Path]]): 出力パス（Noneの場合は表示のみ）
        resolution (Tuple[int, int]): 解像度
        view_angle (Tuple[float, float]): 視点角度
        colormap (Optional[str]): カラーマップ名
        vertex_values (Optional[np.ndarray]): 頂点値（色付け用）
        **kwargs: 追加の可視化オプション
        
    Returns:
        Optional[Path]: 保存されたファイルのパス
    """
    visualizer = ObjVisualizer(obj_path)
    visualizer.load_mesh()
    
    # 頂点値が指定されている場合は色を適用
    if vertex_values is not None and colormap is not None:
        visualizer.apply_vertex_colors(vertex_values, color_mode="scalar")
    
    if output_path is not None:
        return visualizer.save_image(
            output_path=output_path,
            resolution=resolution,
            view_angle=view_angle,
            **kwargs
        )
    else:
        # 表示のみ
        image_data = visualizer.create_screenshot(
            resolution=resolution,
            view_angle=view_angle,
            **kwargs
        )
        plt.figure(figsize=(10, 8))
        plt.imshow(image_data)
        plt.axis('off')
        plt.title("OBJ Visualization: " + Path(obj_path).name)
        plt.show()
        return None


if __name__ == "__main__":
    # 使用例
    import argparse
    
    parser = argparse.ArgumentParser(description='OBJファイルを可視化します')
    parser.add_argument('obj_path', help='OBJファイルのパス')
    parser.add_argument('-o', '--output', help='出力画像のパス')
    parser.add_argument('--resolution', type=int, nargs=2, default=[800, 600],
                       metavar=('WIDTH', 'HEIGHT'), help='画像解像度')
    parser.add_argument('--view-angle', type=float, nargs=2, default=[45, 45],
                       metavar=('ELEVATION', 'AZIMUTH'), help='視点角度')
    parser.add_argument('--wireframe', action='store_true', help='ワイヤーフレーム表示')
    parser.add_argument('--info', action='store_true', help='メッシュ情報を表示')
    
    args = parser.parse_args()
    
    try:
        visualizer = ObjVisualizer(args.obj_path)
        visualizer.load_mesh()
        
        if args.info:
            info = visualizer.get_mesh_info()
            print("メッシュ情報:")
            print(f"  頂点数: {info['vertices']}")
            print(f"  面数: {info['faces']}")
            print(f"  表面積: {info['surface_area']:.3f}")
            print(f"  体積: {info['volume']:.3f}" if info['volume'] else "  体積: N/A (非水密)")
            print(f"  境界: {info['bounds']}")
            print(f"  重心: {info['center_mass']}")
            print(f"  水密: {'はい' if info['is_watertight'] else 'いいえ'}")
        
        if args.output:
            output_path = visualizer.save_image(
                output_path=args.output,
                resolution=tuple(args.resolution),
                view_angle=tuple(args.view_angle),
                wireframe=args.wireframe
            )
            print(f"画像を保存しました: {output_path}")
        else:
            quick_visualize(
                obj_path=args.obj_path,
                resolution=tuple(args.resolution),
                view_angle=tuple(args.view_angle),
                wireframe=args.wireframe
            )
            
    except Exception as e:
        print(f"エラー: {e}")