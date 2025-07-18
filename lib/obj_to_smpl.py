"""
OBJファイルからSMPLモデル作成ライブラリ

このモジュールはOBJファイルからSMPLモデルを作成し、SMPLパラメータをフィッティングする機能を提供します。
"""

from typing import Optional, Union, Dict, Any, Tuple, List
import numpy as np
import torch
import trimesh
from pathlib import Path
import json
import requests
import os
from scipy.optimize import minimize
from smplx import SMPL
import warnings


class ObjToSmplConverter:
    """
    OBJファイルからSMPLモデルへの変換クラス
    """
    
    def __init__(
        self, 
        smpl_model_path: Union[str, Path],
        device: str = "cpu",
        gender: str = "neutral",
        segmentation_file: Optional[Union[str, Path]] = None
    ):
        """
        SMPL変換器を初期化
        
        Args:
            smpl_model_path (Union[str, Path]): SMPLモデルファイルのパス
            device (str): 計算デバイス ("cpu" または "cuda")
            gender (str): 性別 ("neutral", "male", "female")
            segmentation_file (Optional[Union[str, Path]]): セグメンテーションファイルのパス
        """
        self.device = device
        self.smpl_model_path = Path(smpl_model_path)
        self.gender = gender
        
        # SMPLモデルの初期化
        self.smpl_model = SMPL(
            model_path=str(self.smpl_model_path),
            gender=gender,
            batch_size=1
        ).to(device)
        
        # セグメンテーションデータの読み込み
        self.segmentation = self._load_segmentation(segmentation_file)
        
        # 最適化用のパラメータ範囲
        self.param_bounds = {
            'betas': (-3.0, 3.0),  # 体型パラメータ
            'global_orient': (-np.pi, np.pi),  # グローバル回転
            'body_pose': (-np.pi, np.pi),  # ボディポーズ
            'transl': (-2.0, 2.0)  # 並進
        }
    
    def _load_segmentation(self, segmentation_file: Optional[Union[str, Path]]) -> Dict[str, List[int]]:
        """
        セグメンテーションファイルを読み込む
        
        Args:
            segmentation_file (Optional[Union[str, Path]]): セグメンテーションファイルのパス
            
        Returns:
            Dict[str, List[int]]: セグメンテーションデータ
        """
        if segmentation_file is None:
            segmentation_file = "smpl_vert_segmentation.json"
        
        segmentation_file = Path(segmentation_file)
        
        if segmentation_file.exists():
            with open(segmentation_file, 'r') as f:
                return json.load(f)
        else:
            return self._download_segmentation_file(str(segmentation_file))
    
    def _download_segmentation_file(self, json_path: str) -> Dict[str, List[int]]:
        """
        SMPLセグメンテーションファイルをダウンロード
        
        Args:
            json_path (str): 保存パス
            
        Returns:
            Dict[str, List[int]]: セグメンテーションデータ
        """
        url = (
            "https://raw.githubusercontent.com/Meshcapade/wiki/main/assets/"
            "SMPL_body_segmentation/smpl/smpl_vert_segmentation.json"
        )
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            seg_data = response.json()
            
            with open(json_path, 'w') as f:
                json.dump(seg_data, f, indent=2)
            
            return seg_data
            
        except Exception as e:
            warnings.warn(f"セグメンテーションファイルのダウンロードに失敗: {e}")
            return {}
    
    def load_obj_mesh(self, obj_path: Union[str, Path]) -> trimesh.Trimesh:
        """
        OBJファイルを読み込む
        
        Args:
            obj_path (Union[str, Path]): OBJファイルのパス
            
        Returns:
            trimesh.Trimesh: 読み込まれたメッシュ
            
        Raises:
            FileNotFoundError: ファイルが見つからない場合
            ValueError: メッシュの読み込みに失敗した場合
        """
        obj_path = Path(obj_path)
        
        if not obj_path.exists():
            raise FileNotFoundError(f"OBJファイルが見つかりません: {obj_path}")
        
        try:
            mesh = trimesh.load(str(obj_path))
            
            if not isinstance(mesh, trimesh.Trimesh):
                if hasattr(mesh, 'geometry') and len(mesh.geometry) > 0:
                    mesh = list(mesh.geometry.values())[0]
                else:
                    raise ValueError("有効なメッシュが見つかりません")
            
            return mesh
            
        except Exception as e:
            raise ValueError(f"OBJファイルの読み込みに失敗: {e}")
    
    def create_smpl_from_vertices(
        self,
        target_vertices: np.ndarray,
        initial_params: Optional[Dict[str, np.ndarray]] = None,
        optimization_steps: int = 1000,
        learning_rate: float = 0.01,
        regularization_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        ターゲット頂点からSMPLパラメータをフィッティング
        
        Args:
            target_vertices (np.ndarray): ターゲット頂点 (N, 3)
            initial_params (Optional[Dict[str, np.ndarray]]): 初期パラメータ
            optimization_steps (int): 最適化ステップ数
            learning_rate (float): 学習率
            regularization_weights (Optional[Dict[str, float]]): 正則化重み
            
        Returns:
            Dict[str, Any]: フィッティング結果
        """
        if regularization_weights is None:
            regularization_weights = {
                'betas': 0.001,
                'pose': 0.001,
                'shape_prior': 0.01
            }
        
        # 初期パラメータの設定
        if initial_params is None:
            initial_params = {
                'betas': np.zeros(10),
                'global_orient': np.zeros(3),
                'body_pose': np.zeros(69),
                'transl': np.zeros(3)
            }
        
        # パラメータをTensorに変換
        params = {}
        for key, value in initial_params.items():
            params[key] = torch.tensor(
                value, dtype=torch.float32, device=self.device, requires_grad=True
            )
        
        # オプティマイザーの設定
        optimizer = torch.optim.Adam(list(params.values()), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)
        
        # ターゲット頂点をTensorに変換
        target_vertices_tensor = torch.tensor(
            target_vertices, dtype=torch.float32, device=self.device
        )
        
        # 最適化ループ
        losses = []
        best_loss = float('inf')
        best_params = None
        
        for step in range(optimization_steps):
            optimizer.zero_grad()
            
            # SMPLメッシュの生成
            smpl_output = self.smpl_model(
                betas=params['betas'].unsqueeze(0),
                global_orient=params['global_orient'].unsqueeze(0),
                body_pose=params['body_pose'].unsqueeze(0),
                transl=params['transl'].unsqueeze(0)
            )
            
            predicted_vertices = smpl_output.vertices[0]
            
            # 頂点数が異なる場合の対応
            if len(predicted_vertices) != len(target_vertices_tensor):
                # 最も近い頂点を見つけてマッピング
                vertex_loss = self._compute_vertex_matching_loss(
                    predicted_vertices, target_vertices_tensor
                )
            else:
                # 直接比較
                vertex_loss = torch.nn.functional.mse_loss(
                    predicted_vertices, target_vertices_tensor
                )
            
            # 正則化項
            regularization_loss = (
                regularization_weights['betas'] * torch.sum(params['betas'] ** 2) +
                regularization_weights['pose'] * torch.sum(params['body_pose'] ** 2) +
                regularization_weights['shape_prior'] * torch.sum(params['global_orient'] ** 2)
            )
            
            # 総損失
            total_loss = vertex_loss + regularization_loss
            
            # バックプロパゲーション
            total_loss.backward()
            
            # パラメータ制約の適用
            with torch.no_grad():
                for key, param in params.items():
                    if key in self.param_bounds:
                        min_val, max_val = self.param_bounds[key]
                        param.clamp_(min_val, max_val)
            
            optimizer.step()
            scheduler.step()
            
            # 損失の記録
            loss_value = total_loss.item()
            losses.append(loss_value)
            
            # 最良パラメータの保存
            if loss_value < best_loss:
                best_loss = loss_value
                best_params = {key: param.detach().cpu().numpy().copy() 
                              for key, param in params.items()}
            
            # プログレス表示
            if step % 100 == 0:
                print(f"Step {step:4d}: Loss = {loss_value:.6f}, "
                      f"Vertex Loss = {vertex_loss.item():.6f}, "
                      f"Reg Loss = {regularization_loss.item():.6f}")
        
        return {
            'params': best_params,
            'loss': best_loss,
            'losses': losses,
            'final_vertices': predicted_vertices.detach().cpu().numpy()
        }
    
    def _compute_vertex_matching_loss(
        self, 
        predicted_vertices: torch.Tensor, 
        target_vertices: torch.Tensor
    ) -> torch.Tensor:
        """
        頂点数が異なる場合の最適なマッチング損失を計算
        
        Args:
            predicted_vertices (torch.Tensor): 予測頂点
            target_vertices (torch.Tensor): ターゲット頂点
            
        Returns:
            torch.Tensor: マッチング損失
        """
        # 各ターゲット頂点に対して最も近い予測頂点を見つける
        distances = torch.cdist(target_vertices, predicted_vertices)
        min_distances, _ = torch.min(distances, dim=1)
        
        # 平均距離を損失として使用
        return torch.mean(min_distances)
    
    def fit_obj_to_smpl(
        self,
        obj_path: Union[str, Path],
        optimization_steps: int = 1000,
        learning_rate: float = 0.01,
        preprocess_mesh: bool = True
    ) -> Dict[str, Any]:
        """
        OBJファイルからSMPLパラメータをフィッティング
        
        Args:
            obj_path (Union[str, Path]): OBJファイルのパス
            optimization_steps (int): 最適化ステップ数
            learning_rate (float): 学習率
            preprocess_mesh (bool): メッシュの前処理を行うか
            
        Returns:
            Dict[str, Any]: フィッティング結果
        """
        # OBJメッシュの読み込み
        mesh = self.load_obj_mesh(obj_path)
        
        # メッシュの前処理
        if preprocess_mesh:
            mesh = self._preprocess_mesh(mesh)
        
        # フィッティングの実行
        result = self.create_smpl_from_vertices(
            target_vertices=mesh.vertices,
            optimization_steps=optimization_steps,
            learning_rate=learning_rate
        )
        
        result['original_mesh'] = mesh
        result['obj_path'] = str(obj_path)
        
        return result
    
    def _preprocess_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        メッシュの前処理
        
        Args:
            mesh (trimesh.Trimesh): 入力メッシュ
            
        Returns:
            trimesh.Trimesh: 前処理済みメッシュ
        """
        # メッシュのクリーニング
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_vertices()
        
        # 中心を原点に移動
        mesh.vertices -= mesh.center_mass
        
        # スケールの正規化（SMPLの標準的なサイズに合わせる）
        scale = 1.8 / (mesh.bounds[1, 1] - mesh.bounds[0, 1])  # 身長約1.8mに正規化
        mesh.vertices *= scale
        
        return mesh
    
    def create_colored_smpl_mesh(
        self,
        smpl_params: Dict[str, np.ndarray],
        part_colors: Optional[Dict[str, List[float]]] = None
    ) -> trimesh.Trimesh:
        """
        SMPLパラメータから色付きメッシュを生成
        
        Args:
            smpl_params (Dict[str, np.ndarray]): SMPLパラメータ
            part_colors (Optional[Dict[str, List[float]]]): 部位別色設定
            
        Returns:
            trimesh.Trimesh: 色付きSMPLメッシュ
        """
        # SMPLメッシュの生成
        with torch.no_grad():
            smpl_output = self.smpl_model(
                betas=torch.tensor(smpl_params['betas'], dtype=torch.float32, device=self.device).unsqueeze(0),
                global_orient=torch.tensor(smpl_params['global_orient'], dtype=torch.float32, device=self.device).unsqueeze(0),
                body_pose=torch.tensor(smpl_params['body_pose'], dtype=torch.float32, device=self.device).unsqueeze(0),
                transl=torch.tensor(smpl_params.get('transl', np.zeros(3)), dtype=torch.float32, device=self.device).unsqueeze(0)
            )
        
        vertices = smpl_output.vertices[0].cpu().numpy()
        faces = self.smpl_model.faces
        
        # Trimeshオブジェクトの作成
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # 部位別着色
        if part_colors is not None:
            self._apply_part_colors(mesh, part_colors)
        
        return mesh
    
    def _apply_part_colors(
        self,
        mesh: trimesh.Trimesh,
        part_colors: Dict[str, List[float]]
    ) -> None:
        """
        メッシュに部位別着色を適用
        
        Args:
            mesh (trimesh.Trimesh): 対象メッシュ
            part_colors (Dict[str, List[float]]): 部位別色設定
        """
        # デフォルト色（グレー）
        vertex_colors = np.ones((len(mesh.vertices), 4)) * [0.7, 0.7, 0.7, 1.0]
        
        for part_name, color in part_colors.items():
            if part_name in self.segmentation:
                indices = [i for i in self.segmentation[part_name] if i < len(mesh.vertices)]
                if len(color) == 3:
                    color = color + [1.0]  # アルファチャンネルを追加
                vertex_colors[indices] = color
        
        mesh.visual.vertex_colors = vertex_colors
    
    def save_smpl_params(
        self,
        smpl_params: Dict[str, np.ndarray],
        output_path: Union[str, Path],
        include_metadata: bool = True
    ) -> Path:
        """
        SMPLパラメータをファイルに保存
        
        Args:
            smpl_params (Dict[str, np.ndarray]): SMPLパラメータ
            output_path (Union[str, Path]): 出力パス
            include_metadata (bool): メタデータを含めるか
            
        Returns:
            Path: 保存されたファイルのパス
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # パラメータをリストに変換（JSON保存用）
        params_json = {}
        for key, value in smpl_params.items():
            params_json[key] = value.tolist() if isinstance(value, np.ndarray) else value
        
        if include_metadata:
            params_json['metadata'] = {
                'gender': self.gender,
                'smpl_model_path': str(self.smpl_model_path),
                'device': self.device
            }
        
        with open(output_path, 'w') as f:
            json.dump(params_json, f, indent=2)
        
        return output_path
    
    def load_smpl_params(self, params_path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """
        SMPLパラメータをファイルから読み込み
        
        Args:
            params_path (Union[str, Path]): パラメータファイルのパス
            
        Returns:
            Dict[str, np.ndarray]: SMPLパラメータ
        """
        params_path = Path(params_path)
        
        with open(params_path, 'r') as f:
            params_json = json.load(f)
        
        # numpy配列に変換
        params = {}
        for key, value in params_json.items():
            if key != 'metadata':
                params[key] = np.array(value)
        
        return params


def obj_to_smpl(
    obj_path: Union[str, Path],
    smpl_model_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    gender: str = "neutral",
    optimization_steps: int = 1000,
    part_colors: Optional[Dict[str, List[float]]] = None,
    save_mesh: bool = True,
    save_params: bool = True
) -> Dict[str, Any]:
    """
    OBJファイルからSMPLモデルを作成するユーティリティ関数
    
    Args:
        obj_path (Union[str, Path]): OBJファイルのパス
        smpl_model_path (Union[str, Path]): SMPLモデルファイルのパス
        output_dir (Optional[Union[str, Path]]): 出力ディレクトリ
        gender (str): 性別
        optimization_steps (int): 最適化ステップ数
        part_colors (Optional[Dict[str, List[float]]]): 部位別色設定
        save_mesh (bool): メッシュを保存するか
        save_params (bool): パラメータを保存するか
        
    Returns:
        Dict[str, Any]: 変換結果
    """
    converter = ObjToSmplConverter(smpl_model_path, gender=gender)
    
    # フィッティングの実行
    result = converter.fit_obj_to_smpl(
        obj_path=obj_path,
        optimization_steps=optimization_steps
    )
    
    # 出力ディレクトリの設定
    if output_dir is None:
        output_dir = Path(obj_path).parent / "smpl_output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(obj_path).stem
    
    # 色付きSMPLメッシュの作成
    smpl_mesh = converter.create_colored_smpl_mesh(
        smpl_params=result['params'],
        part_colors=part_colors
    )
    
    result['smpl_mesh'] = smpl_mesh
    
    # ファイルの保存
    if save_mesh:
        mesh_path = output_dir / f"{base_name}_smpl.obj"
        smpl_mesh.export(str(mesh_path))
        result['mesh_path'] = mesh_path
    
    if save_params:
        params_path = output_dir / f"{base_name}_smpl_params.json"
        converter.save_smpl_params(result['params'], params_path)
        result['params_path'] = params_path
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='OBJファイルからSMPLモデルを作成します')
    parser.add_argument('obj_path', help='OBJファイルのパス')
    parser.add_argument('smpl_model_path', help='SMPLモデルファイルのパス')
    parser.add_argument('-o', '--output', help='出力ディレクトリ')
    parser.add_argument('--gender', choices=['neutral', 'male', 'female'], 
                       default='neutral', help='性別')
    parser.add_argument('--steps', type=int, default=1000, help='最適化ステップ数')
    parser.add_argument('--no-mesh', action='store_true', help='メッシュ保存をスキップ')
    parser.add_argument('--no-params', action='store_true', help='パラメータ保存をスキップ')
    parser.add_argument('--device', default='cpu', help='計算デバイス')
    
    args = parser.parse_args()
    
    try:
        result = obj_to_smpl(
            obj_path=args.obj_path,
            smpl_model_path=args.smpl_model_path,
            output_dir=args.output,
            gender=args.gender,
            optimization_steps=args.steps,
            save_mesh=not args.no_mesh,
            save_params=not args.no_params
        )
        
        print("変換完了:")
        print(f"  最終損失: {result['loss']:.6f}")
        if 'mesh_path' in result:
            print(f"  SMPLメッシュ: {result['mesh_path']}")
        if 'params_path' in result:
            print(f"  SMPLパラメータ: {result['params_path']}")
            
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()