"""視線が体の部位に変化する時系列グラフを作成するモジュール."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path


class GazeTimelineVisualizer:
    """視線の時系列変化を可視化するクラス."""
    
    def __init__(self) -> None:
        """初期化."""
        self.part_colors = self._create_part_color_mapping()
    
    def _create_part_color_mapping(self) -> Dict[str, str]:
        """各身体部位に色を割り当てる."""
        parts = [
            'head', 'neck', 'spine2', 'spine1', 'spine', 'hips',
            'leftShoulder', 'rightShoulder', 'leftArm', 'rightArm',
            'leftForeArm', 'rightForeArm', 'leftHand', 'rightHand',
            'leftUpLeg', 'rightUpLeg', 'leftLeg', 'rightLeg',
            'leftFoot', 'rightFoot', 'leftToeBase', 'rightToeBase',
            'leftHandIndex1', 'rightHandIndex1'
        ]
        
        # 視覚的に区別しやすい色のパレット
        colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
            '#F8C471', '#82E0AA', '#F1948A', '#85929E', '#A9CCE3',
            '#D7BDE2', '#A3E4D7', '#FAD7A0', '#FADBD8', '#D5DBDB',
            '#FCF3CF', '#EBDEF0', '#E8F8F5', '#FEF9E7'
        ]
        
        return dict(zip(parts, colors[:len(parts)]))
    
    def load_gaze_data(self, csv_path: str) -> pd.DataFrame:
        """視線データCSVファイルを読み込む."""
        df = pd.read_csv(csv_path)
        return df
    
    def create_timeline_graph(
        self, 
        df: pd.DataFrame, 
        output_path: str,
        title: str = "視線の時系列変化",
        figsize: Tuple[int, int] = (15, 8),
        show_seconds: bool = True,
        fps: float = 30.0
    ) -> None:
        """時系列グラフを作成する."""
        plt.figure(figsize=figsize)
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
        
        frames = df['frame'].values
        max_parts = df['max_part'].values
        
        # 部位の変化点を検出
        change_points = []
        current_part = max_parts[0]
        start_frame = frames[0]
        
        for i in range(1, len(max_parts)):
            if max_parts[i] != current_part:
                change_points.append({
                    'start_frame': start_frame,
                    'end_frame': frames[i-1],
                    'part': current_part
                })
                current_part = max_parts[i]
                start_frame = frames[i]
        
        # 最後の区間を追加
        change_points.append({
            'start_frame': start_frame,
            'end_frame': frames[-1],
            'part': current_part
        })
        
        # グラフを描画
        y_pos = 0.5
        bar_height = 0.4
        
        for segment in change_points:
            start = segment['start_frame']
            end = segment['end_frame']
            part = segment['part']
            
            # 時間軸の変換
            if show_seconds:
                start_time = start / fps
                end_time = end / fps
                width = end_time - start_time
            else:
                start_time = start
                end_time = end
                width = end - start
            
            color = self.part_colors.get(part, '#999999')
            
            plt.barh(y_pos, width, left=start_time, height=bar_height, 
                    color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # 軸の設定
        if show_seconds:
            plt.xlabel('時間 (秒)', fontsize=12)
            plt.xlim(0, frames[-1] / fps)
        else:
            plt.xlabel('フレーム番号', fontsize=12)
            plt.xlim(frames[0], frames[-1])
        
        plt.ylim(0, 1)
        plt.yticks([])
        plt.title(title, fontsize=14, fontweight='bold')
        
        # 凡例を作成（主要な部位のみ）
        major_parts = df['max_part'].value_counts().head(10).index.tolist()
        legend_patches = []
        for part in major_parts:
            if part in self.part_colors:
                legend_patches.append(
                    mpatches.Patch(color=self.part_colors[part], label=part)
                )
        
        plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_part_transition_graph(
        self,
        df: pd.DataFrame,
        output_path: str,
        title: str = "視線部位の遷移回数",
        figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """部位間の遷移を可視化する."""
        plt.figure(figsize=figsize)
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
        
        max_parts = df['max_part'].values
        
        # 遷移をカウント
        transitions = {}
        for i in range(1, len(max_parts)):
            prev_part = max_parts[i-1]
            curr_part = max_parts[i]
            
            if prev_part != curr_part:
                transition = f"{prev_part} → {curr_part}"
                transitions[transition] = transitions.get(transition, 0) + 1
        
        # 上位の遷移のみ表示
        sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
        top_transitions = sorted_transitions[:20]  # 上位20の遷移
        
        transition_names = [t[0] for t in top_transitions]
        transition_counts = [t[1] for t in top_transitions]
        
        plt.barh(range(len(transition_names)), transition_counts, color='skyblue', alpha=0.7)
        plt.yticks(range(len(transition_names)), transition_names)
        plt.xlabel('遷移回数', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """視線データの統計情報を作成する."""
        max_parts = df['max_part'].values
        
        # 各部位の注視時間（フレーム数）
        part_counts = df['max_part'].value_counts()
        
        # 部位の変化回数
        changes = 0
        for i in range(1, len(max_parts)):
            if max_parts[i] != max_parts[i-1]:
                changes += 1
        
        # 最長注視区間
        current_part = max_parts[0]
        current_length = 1
        max_length = 1
        max_length_part = current_part
        
        for i in range(1, len(max_parts)):
            if max_parts[i] == current_part:
                current_length += 1
            else:
                if current_length > max_length:
                    max_length = current_length
                    max_length_part = current_part
                current_part = max_parts[i]
                current_length = 1
        
        # 最後の区間もチェック
        if current_length > max_length:
            max_length = current_length
            max_length_part = current_part
        
        return {
            'total_frames': len(df),
            'unique_parts': len(part_counts),
            'total_changes': changes,
            'most_attended_part': part_counts.index[0],
            'most_attended_frames': part_counts.iloc[0],
            'longest_attention_part': max_length_part,
            'longest_attention_frames': max_length,
            'part_distribution': part_counts.to_dict()
        }


def main():
    """メイン関数."""
    # データの読み込み
    csv_path = "/Users/naoya/dev/research-university/smplmodel_sample/results/User11/frame_gaze_analysis.csv"
    output_dir = Path("/Users/naoya/dev/research-university/smplmodel_sample/output")
    output_dir.mkdir(exist_ok=True)
    
    visualizer = GazeTimelineVisualizer()
    
    # データ読み込み
    df = visualizer.load_gaze_data(csv_path)
    print(f"データを読み込みました: {len(df)} フレーム")
    
    # 時系列グラフ作成（秒単位）
    timeline_output = output_dir / "gaze_timeline_seconds.png"
    visualizer.create_timeline_graph(
        df, 
        str(timeline_output),
        title="視線の時系列変化 (User11)",
        show_seconds=True,
        fps=30.0
    )
    
    # 時系列グラフ作成（フレーム単位も作成）
    timeline_frame_output = output_dir / "gaze_timeline_frames.png"
    visualizer.create_timeline_graph(
        df,
        str(timeline_frame_output),
        title="視線の時系列変化 - フレーム単位 (User11)",
        show_seconds=False
    )
    
    # 部位遷移グラフ作成
    transition_output = output_dir / "gaze_part_transitions.png"
    visualizer.create_part_transition_graph(
        df,
        str(transition_output),
        title="視線部位の遷移パターン (User11)"
    )
    
    # 統計情報出力
    stats = visualizer.create_summary_statistics(df)
    print("=== 視線データ統計情報 ===")
    print(f"総フレーム数: {stats['total_frames']}")
    print(f"注視部位数: {stats['unique_parts']}")
    print(f"部位変化回数: {stats['total_changes']}")
    print(f"最も注視された部位: {stats['most_attended_part']} ({stats['most_attended_frames']} フレーム)")
    print(f"最長注視区間: {stats['longest_attention_part']} ({stats['longest_attention_frames']} フレーム)")
    
    print(f"\nグラフを保存しました:")
    print(f"- 時系列グラフ(秒): {timeline_output}")
    print(f"- 時系列グラフ(フレーム): {timeline_frame_output}")
    print(f"- 遷移グラフ: {transition_output}")


if __name__ == "__main__":
    main()