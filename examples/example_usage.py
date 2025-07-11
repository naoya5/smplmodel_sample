#!/usr/bin/env python3
"""
視線データ分布解析システム 使用例

このファイルは視線データ分布解析システムの基本的な使用方法を示すサンプルコードです。
実際のプロジェクトでの使用時の参考にしてください。

実行方法:
    uv run --frozen python examples/example_usage.py

必要なファイル:
    - gaze_data.csv (視線データ)
    - model.obj (3Dメッシュファイル)

著者: Claude Code
作成日: 2025-07-10
"""

import sys
from pathlib import Path
import numpy as np

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.gaze_distribution_analyzer import (
    GazeDataLoader, 
    MeshProcessor, 
    GazeDistributionAnalyzer, 
    ResultExporter
)


def create_sample_data():
    """
    デモンストレーション用のサンプルデータを生成
    
    実際の使用では、この関数は不要です。
    実際の視線データとメッシュファイルを使用してください。
    """
    print("サンプルデータを生成中...")
    
    # サンプルディレクトリの作成
    sample_dir = project_root / "examples" / "sample_data"
    sample_dir.mkdir(exist_ok=True)
    
    # 1. サンプル視線データ（CSV）の生成
    gaze_csv_path = sample_dir / "sample_gaze_data.csv"
    
    with open(gaze_csv_path, 'w', encoding='utf-8') as f:
        f.write("timestamp,gaze_x,gaze_y,gaze_z,origin_x,origin_y,origin_z,confidence\n")
        
        # 10秒間のサンプルデータ（100Hz）
        for i in range(1000):
            timestamp = i * 0.01  # 10ms間隔
            
            # 視線方向をシミュレート（顔の周りを見回す動き）
            angle = timestamp * 0.5  # ゆっくりとした回転
            gaze_x = 0.3 * np.sin(angle) + np.random.normal(0, 0.05)
            gaze_y = 0.2 * np.cos(angle * 1.3) + np.random.normal(0, 0.03)
            gaze_z = 0.9 + np.random.normal(0, 0.02)
            
            # 視線方向を正規化
            gaze_norm = np.sqrt(gaze_x**2 + gaze_y**2 + gaze_z**2)
            gaze_x /= gaze_norm
            gaze_y /= gaze_norm
            gaze_z /= gaze_norm
            
            # カメラ位置（固定）
            origin_x, origin_y, origin_z = 0.0, 0.0, 2.0
            
            # 信頼度（時々低くなる）
            confidence = 0.95 if np.random.random() > 0.1 else np.random.uniform(0.6, 0.8)
            
            f.write(f"{timestamp:.3f},{gaze_x:.6f},{gaze_y:.6f},{gaze_z:.6f},"
                   f"{origin_x:.1f},{origin_y:.1f},{origin_z:.1f},{confidence:.2f}\n")
    
    print(f"サンプル視線データを生成: {gaze_csv_path}")
    
    # 2. サンプルメッシュ（球体）の生成とOBJ保存
    import trimesh
    
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    obj_path = sample_dir / "sample_sphere.obj"
    sphere.export(str(obj_path))
    
    print(f"サンプルメッシュを生成: {obj_path}")
    
    return gaze_csv_path, obj_path


def example_basic_analysis():
    """
    基本的な視線分布解析の例
    """
    print("\n" + "="*60)
    print("例1: 基本的な視線分布解析")
    print("="*60)
    
    # サンプルデータの生成
    gaze_csv_path, obj_path = create_sample_data()
    
    # 1. 視線データの読み込み
    print("\n1. 視線データの読み込み...")
    gaze_data = GazeDataLoader.load_from_csv(gaze_csv_path)
    print(f"   読み込んだ視線データポイント数: {len(gaze_data)}")
    print(f"   時間範囲: {gaze_data[0].timestamp:.2f}s - {gaze_data[-1].timestamp:.2f}s")
    
    # 視線方向の正規化
    gaze_data = GazeDataLoader.normalize_gaze_directions(gaze_data)
    print("   視線方向ベクトルを正規化しました")
    
    # 2. メッシュの読み込み
    print("\n2. 3Dメッシュの読み込み...")
    mesh_processor = MeshProcessor.load_from_obj(obj_path)
    print(f"   頂点数: {len(mesh_processor.vertices)}")
    print(f"   面数: {len(mesh_processor.faces)}")
    print(f"   表面積: {mesh_processor.calculate_surface_area():.4f}")
    
    # 3. 解析器の初期化
    print("\n3. 解析器の初期化...")
    analyzer = GazeDistributionAnalyzer(
        mesh_processor,
        distance_threshold=0.1,    # 10cm以内
        angle_threshold=np.pi/4,   # 45度以内
        temporal_window=0.5        # 0.5秒の時間窓
    )
    print("   解析器を初期化しました")
    
    # 4. 時系列視線分布解析
    print("\n4. 時系列視線分布解析の実行...")
    temporal_distribution = analyzer.analyze_temporal_distribution(
        gaze_data, 
        time_step=0.1  # 0.1秒間隔
    )
    print(f"   解析した時刻数: {len(temporal_distribution)}")
    
    # 5. 結果の統計情報
    print("\n5. 解析結果の統計...")
    total_probabilities = []
    max_probabilities = []
    
    for timestamp, vertex_probs in temporal_distribution.items():
        total_prob = sum(vp.probability for vp in vertex_probs)
        max_prob = max(vp.probability for vp in vertex_probs) if vertex_probs else 0
        
        total_probabilities.append(total_prob)
        max_probabilities.append(max_prob)
    
    print(f"   平均総確率: {np.mean(total_probabilities):.4f}")
    print(f"   最大確率の平均: {np.mean(max_probabilities):.4f}")
    print(f"   最大確率の最大値: {np.max(max_probabilities):.4f}")
    
    # 6. 結果の出力
    print("\n6. 結果の出力...")
    output_dir = project_root / "examples" / "output"
    output_dir.mkdir(exist_ok=True)
    
    # CSV出力
    csv_output = output_dir / "basic_analysis_results.csv"
    ResultExporter.export_to_csv(temporal_distribution, csv_output)
    print(f"   CSV出力: {csv_output}")
    
    # JSON出力
    json_output = output_dir / "basic_analysis_results.json"
    ResultExporter.export_to_json(temporal_distribution, json_output)
    print(f"   JSON出力: {json_output}")
    
    # ヒートマップ出力
    heatmap_output = output_dir / "basic_analysis_heatmap.png"
    try:
        ResultExporter.visualize_temporal_heatmap(
            temporal_distribution, mesh_processor, heatmap_output
        )
        print(f"   ヒートマップ: {heatmap_output}")
    except Exception as e:
        print(f"   ヒートマップ生成エラー: {e}")
    
    return temporal_distribution, mesh_processor


def example_attention_hotspots(temporal_distribution, mesh_processor):
    """
    注目ホットスポットの特定例
    """
    print("\n" + "="*60)
    print("例2: 注目ホットスポットの特定")
    print("="*60)
    
    # 1. 各頂点の総注目度を計算
    print("\n1. 各頂点の総注目度を計算...")
    vertex_attention = {}
    
    for timestamp, vertex_probs in temporal_distribution.items():
        for vp in vertex_probs:
            if vp.vertex_index not in vertex_attention:
                vertex_attention[vp.vertex_index] = []
            vertex_attention[vp.vertex_index].append(vp.probability)
    
    # 平均注目度を計算
    vertex_avg_attention = {
        vertex_idx: np.mean(probs) 
        for vertex_idx, probs in vertex_attention.items()
    }
    
    # 2. トップ10の注目頂点を特定
    print("\n2. 最も注目された頂点トップ10:")
    sorted_vertices = sorted(
        vertex_avg_attention.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    for i, (vertex_idx, avg_attention) in enumerate(sorted_vertices[:10]):
        vertex_pos = mesh_processor.vertices[vertex_idx]
        print(f"   {i+1:2d}. 頂点{vertex_idx:4d}: 平均注目度={avg_attention:.4f}, "
              f"位置=({vertex_pos[0]:.3f}, {vertex_pos[1]:.3f}, {vertex_pos[2]:.3f})")
    
    # 3. 注目度の時間変化を分析
    print("\n3. 注目度の時間変化分析...")
    top_vertex = sorted_vertices[0][0]
    top_vertex_timeline = []
    timestamps = []
    
    for timestamp in sorted(temporal_distribution.keys()):
        vertex_probs = temporal_distribution[timestamp]
        prob = next((vp.probability for vp in vertex_probs 
                    if vp.vertex_index == top_vertex), 0.0)
        top_vertex_timeline.append(prob)
        timestamps.append(timestamp)
    
    print(f"   最注目頂点 {top_vertex} の注目度変化:")
    print(f"   最小値: {np.min(top_vertex_timeline):.4f}")
    print(f"   最大値: {np.max(top_vertex_timeline):.4f}")
    print(f"   標準偏差: {np.std(top_vertex_timeline):.4f}")
    
    # 4. 結果の可視化（簡易版）
    print("\n4. 注目度分析結果の保存...")
    output_dir = project_root / "examples" / "output"
    
    # 注目度データをCSVで保存
    attention_csv = output_dir / "vertex_attention_analysis.csv"
    with open(attention_csv, 'w', encoding='utf-8') as f:
        f.write("vertex_index,avg_attention,x,y,z\n")
        for vertex_idx, avg_attention in sorted_vertices:
            pos = mesh_processor.vertices[vertex_idx]
            f.write(f"{vertex_idx},{avg_attention:.6f},"
                   f"{pos[0]:.6f},{pos[1]:.6f},{pos[2]:.6f}\n")
    
    print(f"   注目度分析結果: {attention_csv}")
    
    return sorted_vertices


def example_temporal_patterns(temporal_distribution):
    """
    時間パターン分析の例
    """
    print("\n" + "="*60)
    print("例3: 時間パターン分析")
    print("="*60)
    
    # 1. 時系列データの準備
    print("\n1. 時系列データの準備...")
    timestamps = sorted(temporal_distribution.keys())
    time_series_data = []
    
    for timestamp in timestamps:
        vertex_probs = temporal_distribution[timestamp]
        total_attention = sum(vp.probability for vp in vertex_probs)
        max_attention = max(vp.probability for vp in vertex_probs) if vertex_probs else 0
        vertex_count = len([vp for vp in vertex_probs if vp.probability > 0.01])
        
        time_series_data.append({
            'timestamp': timestamp,
            'total_attention': total_attention,
            'max_attention': max_attention,
            'active_vertices': vertex_count
        })
    
    print(f"   時系列データポイント数: {len(time_series_data)}")
    
    # 2. 基本統計の計算
    print("\n2. 時系列統計:")
    total_attentions = [data['total_attention'] for data in time_series_data]
    max_attentions = [data['max_attention'] for data in time_series_data]
    active_vertices = [data['active_vertices'] for data in time_series_data]
    
    print(f"   総注目度: 平均={np.mean(total_attentions):.4f}, "
          f"標準偏差={np.std(total_attentions):.4f}")
    print(f"   最大注目度: 平均={np.mean(max_attentions):.4f}, "
          f"標準偏差={np.std(max_attentions):.4f}")
    print(f"   活性頂点数: 平均={np.mean(active_vertices):.1f}, "
          f"標準偏差={np.std(active_vertices):.1f}")
    
    # 3. 変化点の検出
    print("\n3. 注目パターンの変化点検出...")
    attention_changes = []
    for i in range(1, len(total_attentions)):
        change = abs(total_attentions[i] - total_attentions[i-1])
        attention_changes.append(change)
    
    # 大きな変化点を特定
    threshold = np.mean(attention_changes) + 2 * np.std(attention_changes)
    significant_changes = []
    
    for i, change in enumerate(attention_changes):
        if change > threshold:
            significant_changes.append({
                'timestamp': timestamps[i+1],
                'change_magnitude': change,
                'from_attention': total_attentions[i],
                'to_attention': total_attentions[i+1]
            })
    
    print(f"   検出された有意な変化点: {len(significant_changes)}個")
    for change in significant_changes[:5]:  # 最大5個まで表示
        print(f"     時刻 {change['timestamp']:.2f}s: "
              f"変化量 {change['change_magnitude']:.4f}")
    
    # 4. 結果の保存
    print("\n4. 時間パターン分析結果の保存...")
    output_dir = project_root / "examples" / "output"
    
    # 時系列データをCSVで保存
    timeseries_csv = output_dir / "temporal_pattern_analysis.csv"
    with open(timeseries_csv, 'w', encoding='utf-8') as f:
        f.write("timestamp,total_attention,max_attention,active_vertices\n")
        for data in time_series_data:
            f.write(f"{data['timestamp']:.3f},{data['total_attention']:.6f},"
                   f"{data['max_attention']:.6f},{data['active_vertices']}\n")
    
    print(f"   時系列分析結果: {timeseries_csv}")
    
    # 変化点データの保存
    if significant_changes:
        changes_csv = output_dir / "attention_change_points.csv"
        with open(changes_csv, 'w', encoding='utf-8') as f:
            f.write("timestamp,change_magnitude,from_attention,to_attention\n")
            for change in significant_changes:
                f.write(f"{change['timestamp']:.3f},{change['change_magnitude']:.6f},"
                       f"{change['from_attention']:.6f},{change['to_attention']:.6f}\n")
        print(f"   変化点分析結果: {changes_csv}")


def example_smpl_integration():
    """
    SMPLモデルとの統合例
    """
    print("\n" + "="*60)
    print("例4: SMPLモデルとの統合（デモ）")
    print("="*60)
    
    print("\n注意: この例は実際のSMPLモデルファイルが必要です。")
    print("デモ用の疑似コードを表示します。\n")
    
    demo_code = """
# SMPLモデルとの統合例

from src.pkl2obj import process_4d_humans_pkl_smpl
from src.gaze_distribution_analyzer import MeshProcessor, GazeDistributionAnalyzer

def analyze_gaze_on_smpl_model():
    # 1. SMPLメッシュの生成
    smpl_mesh = process_4d_humans_pkl_smpl(
        pkl_path="demo_gymnasts.pkl",
        smpl_model_path="SMPL_NEUTRAL.npz",
        frame_idx=0,
        person_idx=0,
        part_name="head",  # 頭部を強調
        color=[1.0, 0.0, 0.0]
    )
    
    # 2. 視線解析の実行
    mesh_processor = MeshProcessor(smpl_mesh)
    analyzer = GazeDistributionAnalyzer(
        mesh_processor,
        distance_threshold=0.05,  # より精密な解析
        temporal_window=0.3
    )
    
    # 3. 視線データの読み込み
    gaze_data = GazeDataLoader.load_from_csv("human_gaze_data.csv")
    
    # 4. 人体部位別の注目度分析
    temporal_distribution = analyzer.analyze_temporal_distribution(gaze_data)
    
    # 5. 部位別着色結果の出力
    # 注目度に基づいて部位を着色
    attention_colored_mesh = color_part_based_on_attention(
        smpl_mesh, temporal_distribution
    )
    
    return attention_colored_mesh, temporal_distribution

def color_part_based_on_attention(mesh, temporal_distribution):
    \"\"\"注目度に基づいて部位を着色\"\"\"
    # 各頂点の平均注目度を計算
    vertex_attention = calculate_average_attention(temporal_distribution)
    
    # 注目度を色に変換（赤：高注目、青：低注目）
    colors = attention_to_color_map(vertex_attention)
    
    # メッシュに色を適用
    mesh.visual.vertex_colors = colors
    
    return mesh
    """
    
    print(demo_code)
    print("実際の使用時は、適切なSMPLモデルファイルとデータを使用してください。")


def main():
    """
    メイン実行関数 - すべての使用例を実行
    """
    print("視線データ分布解析システム - 使用例デモンストレーション")
    print("="*80)
    
    try:
        # 基本的な解析例
        temporal_distribution, mesh_processor = example_basic_analysis()
        
        # 注目ホットスポット分析例
        example_attention_hotspots(temporal_distribution, mesh_processor)
        
        # 時間パターン分析例
        example_temporal_patterns(temporal_distribution)
        
        # SMPLモデル統合例（デモ）
        example_smpl_integration()
        
        print("\n" + "="*80)
        print("すべての使用例が正常に実行されました！")
        print("結果ファイルは examples/output/ ディレクトリに保存されています。")
        print("="*80)
        
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        print("詳細なエラー情報:")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()