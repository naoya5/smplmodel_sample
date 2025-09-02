#!/usr/bin/env python3
"""
視線データのLLM言語化分析システム
視線データを分析してLLMが理解しやすい形式に変換し、自然言語での解説を生成
"""

import json
import os
import statistics
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple, Optional

import pandas as pd


class GazeLLMAnalyzer:
    """視線データのLLM分析用クラス"""
    
    def __init__(self):
        """初期化"""
        self.frame_rate = 30.0
        self.frame_duration = 1.0 / self.frame_rate
        
    def load_analysis_results(self, results_dir: str) -> Dict[str, Any]:
        """
        分析結果を読み込み
        
        Args:
            results_dir: 分析結果ディレクトリ
            
        Returns:
            統合された分析データ
        """
        data = {}
        
        # 各ファイルを読み込み
        files_to_load = [
            ('durations', 'part_gaze_durations.csv'),
            ('transitions', 'gaze_transitions.csv'),
            ('statistics', 'part_statistics.json'),
            ('timeline', 'timeline_analysis.json')
        ]
        
        for key, filename in files_to_load:
            filepath = os.path.join(results_dir, filename)
            if os.path.exists(filepath):
                if filename.endswith('.csv'):
                    data[key] = pd.read_csv(filepath)
                elif filename.endswith('.json'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data[key] = json.load(f)
                        
        return data
        
    def extract_behavioral_patterns(self, transitions_df: pd.DataFrame, 
                                   timeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        行動パターンを抽出
        
        Args:
            transitions_df: 注視変化データ
            timeline_data: タイムラインデータ
            
        Returns:
            行動パターン分析結果
        """
        patterns = {}
        
        if transitions_df.empty:
            return patterns
            
        # 1. 固視時間パターン
        fixation_durations = []
        for i in range(len(transitions_df) - 1):
            duration = transitions_df.iloc[i + 1]['time_seconds'] - transitions_df.iloc[i]['time_seconds']
            fixation_durations.append(duration)
            
        if fixation_durations:
            patterns['fixation_analysis'] = {
                'average_fixation_duration': statistics.mean(fixation_durations),
                'median_fixation_duration': statistics.median(fixation_durations),
                'shortest_fixation': min(fixation_durations),
                'longest_fixation': max(fixation_durations),
                'fixation_variability': statistics.stdev(fixation_durations) if len(fixation_durations) > 1 else 0
            }
        
        # 2. 注視変化パターン
        transition_types = []
        body_parts_hierarchy = self._get_body_parts_hierarchy()
        
        for _, row in transitions_df.iterrows():
            from_part = row['from_part']
            to_part = row['to_part']
            
            # 部位間の距離を計算
            distance = self._calculate_part_distance(from_part, to_part, body_parts_hierarchy)
            transition_types.append({
                'from': from_part,
                'to': to_part,
                'distance': distance,
                'time': row['time_seconds']
            })
        
        patterns['transition_analysis'] = {
            'total_transitions': len(transition_types),
            'transitions_per_second': len(transition_types) / timeline_data['total_time_seconds'],
            'common_transitions': self._find_common_transitions(transition_types),
            'scanning_behavior': self._detect_scanning_behavior(transition_types),
            'return_behavior': self._detect_return_behavior(transition_types)
        }
        
        # 3. 時間的パターン
        dominant_sequence = timeline_data['dominant_parts_sequence']
        patterns['temporal_analysis'] = {
            'exploration_phase': self._analyze_exploration_phase(dominant_sequence[:10]),
            'stabilization_phase': self._analyze_stabilization_phase(dominant_sequence),
            'attention_clusters': self._find_attention_clusters(transitions_df),
            'periodic_patterns': self._detect_periodic_patterns(dominant_sequence)
        }
        
        return patterns
        
    def _get_body_parts_hierarchy(self) -> Dict[str, Dict[str, Any]]:
        """身体部位の階層構造を定義"""
        return {
            'head': {'group': 'upper', 'proximity': {'neck': 1, 'leftShoulder': 2, 'rightShoulder': 2}},
            'neck': {'group': 'upper', 'proximity': {'head': 1, 'spine1': 1, 'spine2': 2}},
            'leftShoulder': {'group': 'upper', 'proximity': {'leftArm': 1, 'neck': 2, 'spine1': 2}},
            'rightShoulder': {'group': 'upper', 'proximity': {'rightArm': 1, 'neck': 2, 'spine1': 2}},
            'leftArm': {'group': 'upper', 'proximity': {'leftForeArm': 1, 'leftShoulder': 1}},
            'rightArm': {'group': 'upper', 'proximity': {'rightForeArm': 1, 'rightShoulder': 1}},
            'leftForeArm': {'group': 'upper', 'proximity': {'leftHand': 1, 'leftArm': 1}},
            'rightForeArm': {'group': 'upper', 'proximity': {'rightHand': 1, 'rightArm': 1}},
            'leftHand': {'group': 'extremity', 'proximity': {'leftHandIndex1': 1, 'leftForeArm': 1}},
            'rightHand': {'group': 'extremity', 'proximity': {'rightHandIndex1': 1, 'rightForeArm': 1}},
            'spine1': {'group': 'torso', 'proximity': {'spine2': 1, 'neck': 1}},
            'spine2': {'group': 'torso', 'proximity': {'spine': 1, 'spine1': 1}},
            'spine': {'group': 'torso', 'proximity': {'hips': 1, 'spine2': 1}},
            'hips': {'group': 'torso', 'proximity': {'leftUpLeg': 1, 'rightUpLeg': 1, 'spine': 1}},
            'leftUpLeg': {'group': 'lower', 'proximity': {'leftLeg': 1, 'hips': 1}},
            'rightUpLeg': {'group': 'lower', 'proximity': {'rightLeg': 1, 'hips': 1}},
            'leftLeg': {'group': 'lower', 'proximity': {'leftFoot': 1, 'leftUpLeg': 1}},
            'rightLeg': {'group': 'lower', 'proximity': {'rightFoot': 1, 'rightUpLeg': 1}},
            'leftFoot': {'group': 'extremity', 'proximity': {'leftToeBase': 1, 'leftLeg': 1}},
            'rightFoot': {'group': 'extremity', 'proximity': {'rightToeBase': 1, 'rightLeg': 1}}
        }
        
    def _calculate_part_distance(self, from_part: str, to_part: str, hierarchy: Dict[str, Dict[str, Any]]) -> int:
        """部位間の距離を計算（解剖学的近接性）"""
        if from_part == to_part:
            return 0
            
        if from_part in hierarchy and to_part in hierarchy[from_part].get('proximity', {}):
            return hierarchy[from_part]['proximity'][to_part]
        elif to_part in hierarchy and from_part in hierarchy[to_part].get('proximity', {}):
            return hierarchy[to_part]['proximity'][from_part]
        else:
            # 同じグループなら中距離、異なるグループなら長距離
            from_group = hierarchy.get(from_part, {}).get('group', 'unknown')
            to_group = hierarchy.get(to_part, {}).get('group', 'unknown')
            return 3 if from_group == to_group else 5
            
    def _find_common_transitions(self, transition_types: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """よくある注視変化パターンを特定"""
        transition_pairs = [(t['from'], t['to']) for t in transition_types]
        common = Counter(transition_pairs).most_common(5)
        
        return [{'from': from_part, 'to': to_part, 'count': count, 'percentage': count/len(transition_types)*100} 
                for (from_part, to_part), count in common]
                
    def _detect_scanning_behavior(self, transition_types: List[Dict[str, Any]]) -> Dict[str, Any]:
        """スキャニング行動を検出"""
        sequential_distances = [t['distance'] for t in transition_types]
        
        # 連続する長距離移動をスキャニングとして検出
        scanning_episodes = []
        current_episode = []
        
        for i, distance in enumerate(sequential_distances):
            if distance >= 4:  # 長距離移動
                current_episode.append(i)
            else:
                if len(current_episode) >= 3:  # 3回以上の連続長距離移動
                    scanning_episodes.append(current_episode)
                current_episode = []
                
        return {
            'scanning_episodes': len(scanning_episodes),
            'total_scanning_transitions': sum(len(ep) for ep in scanning_episodes),
            'scanning_percentage': sum(len(ep) for ep in scanning_episodes) / len(transition_types) * 100 if transition_types else 0
        }
        
    def _detect_return_behavior(self, transition_types: List[Dict[str, Any]]) -> Dict[str, Any]:
        """リターン行動（同じ部位への回帰）を検出"""
        recent_parts = []
        return_count = 0
        
        for transition in transition_types:
            to_part = transition['to']
            if to_part in recent_parts[-5:]:  # 直近5回以内に注視した部位への回帰
                return_count += 1
            recent_parts.append(to_part)
            
        return {
            'return_transitions': return_count,
            'return_percentage': return_count / len(transition_types) * 100 if transition_types else 0
        }
        
    def _analyze_exploration_phase(self, initial_sequence: List[str]) -> Dict[str, Any]:
        """探索フェーズを分析"""
        unique_parts = set(initial_sequence)
        return {
            'unique_parts_explored': len(unique_parts),
            'exploration_diversity': len(unique_parts) / len(initial_sequence) if initial_sequence else 0,
            'dominant_initial_part': Counter(initial_sequence).most_common(1)[0] if initial_sequence else None
        }
        
    def _analyze_stabilization_phase(self, full_sequence: List[str]) -> Dict[str, Any]:
        """安定化フェーズを分析"""
        if len(full_sequence) < 20:
            return {'insufficient_data': True}
            
        # 後半での部位集中度を計算
        latter_half = full_sequence[len(full_sequence)//2:]
        part_counts = Counter(latter_half)
        most_common = part_counts.most_common(1)[0] if part_counts else None
        
        return {
            'stabilization_detected': most_common[1] / len(latter_half) > 0.3 if most_common else False,
            'dominant_latter_part': most_common[0] if most_common else None,
            'concentration_ratio': most_common[1] / len(latter_half) if most_common else 0
        }
        
    def _find_attention_clusters(self, transitions_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """注意のクラスターを特定"""
        if transitions_df.empty:
            return []
            
        clusters = []
        current_cluster = None
        
        for _, row in transitions_df.iterrows():
            time = row['time_seconds']
            part = row['to_part']
            
            # 新しいクラスターの開始判定（5秒以上の間隔）
            if current_cluster is None or time - current_cluster['end_time'] > 5.0:
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = {
                    'start_time': time,
                    'end_time': time,
                    'parts': [part],
                    'part_counts': Counter([part])
                }
            else:
                current_cluster['end_time'] = time
                current_cluster['parts'].append(part)
                current_cluster['part_counts'][part] += 1
                
        if current_cluster:
            clusters.append(current_cluster)
            
        # クラスター情報を整理
        formatted_clusters = []
        for cluster in clusters:
            duration = cluster['end_time'] - cluster['start_time']
            dominant_part = cluster['part_counts'].most_common(1)[0]
            formatted_clusters.append({
                'duration': duration,
                'dominant_part': dominant_part[0],
                'part_diversity': len(set(cluster['parts'])),
                'transition_count': len(cluster['parts'])
            })
            
        return formatted_clusters
        
    def _detect_periodic_patterns(self, sequence: List[str]) -> Dict[str, Any]:
        """周期的なパターンを検出"""
        if len(sequence) < 10:
            return {'insufficient_data': True}
            
        # 簡単な周期検出：連続する同じパターンを探す
        pattern_lengths = [2, 3, 4, 5]
        detected_patterns = []
        
        for length in pattern_lengths:
            for i in range(len(sequence) - length * 2):
                pattern = sequence[i:i+length]
                next_pattern = sequence[i+length:i+length*2]
                
                if pattern == next_pattern:
                    detected_patterns.append({
                        'pattern': pattern,
                        'length': length,
                        'position': i
                    })
                    
        return {
            'periodic_patterns_found': len(detected_patterns),
            'patterns': detected_patterns[:5]  # 上位5パターン
        }
        
    def create_llm_structured_data(self, data: Dict[str, Any], patterns: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM用の構造化データを作成
        
        Args:
            data: 基本分析データ
            patterns: 行動パターンデータ
            
        Returns:
            LLM用構造化データ
        """
        durations_df = data.get('durations')
        statistics_data = data.get('statistics', {})
        timeline_data = data.get('timeline', {})
        
        # 基本統計情報
        basic_info = {
            'total_duration_seconds': timeline_data.get('total_time_seconds', 0),
            'total_frames': timeline_data.get('total_frames', 0),
            'frame_rate': timeline_data.get('frame_rate', 30)
        }
        
        # 主要部位情報
        top_parts = []
        if durations_df is not None and not durations_df.empty:
            for _, row in durations_df.head(5).iterrows():
                percentage = (row['duration_seconds'] / basic_info['total_duration_seconds']) * 100
                top_parts.append({
                    'part_name': row['part_name'],
                    'duration_seconds': row['duration_seconds'],
                    'percentage': percentage
                })
        
        # 行動特徴
        behavioral_features = {
            'attention_stability': self._assess_attention_stability(patterns),
            'exploration_tendency': self._assess_exploration_tendency(patterns),
            'scanning_behavior': self._assess_scanning_behavior(patterns),
            'focus_consistency': self._assess_focus_consistency(patterns)
        }
        
        # コンテキスト推論
        context_inference = {
            'likely_activity': self._infer_activity_type(top_parts, patterns),
            'attention_strategy': self._classify_attention_strategy(patterns),
            'engagement_level': self._assess_engagement_level(patterns, basic_info)
        }
        
        return {
            'basic_information': basic_info,
            'dominant_parts': top_parts,
            'behavioral_patterns': patterns,
            'behavioral_features': behavioral_features,
            'context_inference': context_inference,
            'data_quality': {
                'sufficient_data': basic_info['total_frames'] > 100,
                'analysis_confidence': self._calculate_confidence_score(basic_info, patterns)
            }
        }
        
    def _assess_attention_stability(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """注意の安定性を評価"""
        fixation_analysis = patterns.get('fixation_analysis', {})
        transition_analysis = patterns.get('transition_analysis', {})
        
        if not fixation_analysis or not transition_analysis:
            return {'assessment': 'insufficient_data'}
            
        avg_fixation = fixation_analysis.get('average_fixation_duration', 0)
        transitions_per_sec = transition_analysis.get('transitions_per_second', 0)
        
        stability_score = min(avg_fixation / 2.0, 1.0) * max(0, 1.0 - transitions_per_sec / 2.0)
        
        if stability_score > 0.7:
            level = 'high'
        elif stability_score > 0.4:
            level = 'moderate'
        else:
            level = 'low'
            
        return {
            'level': level,
            'score': stability_score,
            'average_fixation_duration': avg_fixation,
            'transition_frequency': transitions_per_sec
        }
        
    def _assess_exploration_tendency(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """探索傾向を評価"""
        temporal_analysis = patterns.get('temporal_analysis', {})
        exploration_phase = temporal_analysis.get('exploration_phase', {})
        
        if not exploration_phase or exploration_phase.get('insufficient_data'):
            return {'assessment': 'insufficient_data'}
            
        diversity = exploration_phase.get('exploration_diversity', 0)
        unique_parts = exploration_phase.get('unique_parts_explored', 0)
        
        if diversity > 0.8 and unique_parts > 5:
            tendency = 'high_exploration'
        elif diversity > 0.5 and unique_parts > 3:
            tendency = 'moderate_exploration'
        else:
            tendency = 'focused_attention'
            
        return {
            'tendency': tendency,
            'diversity_score': diversity,
            'unique_parts_explored': unique_parts
        }
        
    def _assess_scanning_behavior(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """スキャニング行動を評価"""
        transition_analysis = patterns.get('transition_analysis', {})
        scanning_data = transition_analysis.get('scanning_behavior', {})
        
        if not scanning_data:
            return {'assessment': 'no_scanning_detected'}
            
        scanning_percentage = scanning_data.get('scanning_percentage', 0)
        
        if scanning_percentage > 30:
            intensity = 'high'
        elif scanning_percentage > 15:
            intensity = 'moderate'
        else:
            intensity = 'low'
            
        return {
            'intensity': intensity,
            'scanning_percentage': scanning_percentage,
            'episodes_count': scanning_data.get('scanning_episodes', 0)
        }
        
    def _assess_focus_consistency(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """フォーカスの一貫性を評価"""
        temporal_analysis = patterns.get('temporal_analysis', {})
        stabilization = temporal_analysis.get('stabilization_phase', {})
        
        if stabilization.get('insufficient_data'):
            return {'assessment': 'insufficient_data'}
            
        concentration_ratio = stabilization.get('concentration_ratio', 0)
        
        if concentration_ratio > 0.5:
            consistency = 'high'
        elif concentration_ratio > 0.3:
            consistency = 'moderate'
        else:
            consistency = 'low'
            
        return {
            'consistency': consistency,
            'concentration_ratio': concentration_ratio,
            'dominant_part': stabilization.get('dominant_latter_part')
        }
        
    def _infer_activity_type(self, top_parts: List[Dict[str, Any]], patterns: Dict[str, Any]) -> str:
        """活動タイプを推論"""
        if not top_parts:
            return 'unknown'
            
        primary_part = top_parts[0]['part_name']
        primary_percentage = top_parts[0]['percentage']
        
        # 頭部中心の場合
        if primary_part in ['head', 'neck'] and primary_percentage > 40:
            return 'face_focused_interaction'
        # 手部中心の場合
        elif any(part in primary_part.lower() for part in ['hand', 'finger']) and primary_percentage > 30:
            return 'manual_task_observation'
        # 胴体中心の場合
        elif primary_part in ['spine1', 'spine2', 'hips'] and primary_percentage > 35:
            return 'body_posture_analysis'
        # 分散的な場合
        elif len([p for p in top_parts[:3] if p['percentage'] > 15]) >= 3:
            return 'general_body_scanning'
        else:
            return 'mixed_attention_pattern'
            
    def _classify_attention_strategy(self, patterns: Dict[str, Any]) -> str:
        """注意戦略を分類"""
        fixation_analysis = patterns.get('fixation_analysis', {})
        transition_analysis = patterns.get('transition_analysis', {})
        
        if not fixation_analysis or not transition_analysis:
            return 'unknown'
            
        avg_fixation = fixation_analysis.get('average_fixation_duration', 0)
        transitions_per_sec = transition_analysis.get('transitions_per_second', 0)
        scanning_percentage = transition_analysis.get('scanning_behavior', {}).get('scanning_percentage', 0)
        
        if avg_fixation > 3.0 and transitions_per_sec < 0.5:
            return 'sustained_attention'
        elif scanning_percentage > 25:
            return 'systematic_scanning'
        elif transitions_per_sec > 1.5:
            return 'rapid_switching'
        else:
            return 'balanced_exploration'
            
    def _assess_engagement_level(self, patterns: Dict[str, Any], basic_info: Dict[str, Any]) -> str:
        """エンゲージメントレベルを評価"""
        duration = basic_info.get('total_duration_seconds', 0)
        transition_analysis = patterns.get('transition_analysis', {})
        
        if duration < 10:
            return 'brief_observation'
        elif duration > 60:
            engagement = 'extended_observation'
        else:
            engagement = 'moderate_observation'
            
        # 注視変化の頻度からエンゲージメントを調整
        transitions_per_sec = transition_analysis.get('transitions_per_second', 0)
        if transitions_per_sec > 2.0:
            engagement += '_highly_active'
        elif transitions_per_sec < 0.3:
            engagement += '_passive'
            
        return engagement
        
    def _calculate_confidence_score(self, basic_info: Dict[str, Any], patterns: Dict[str, Any]) -> float:
        """分析の信頼度スコアを計算"""
        score = 0.0
        
        # データ量による信頼度
        frames = basic_info.get('total_frames', 0)
        if frames > 1000:
            score += 0.4
        elif frames > 300:
            score += 0.3
        elif frames > 100:
            score += 0.2
        else:
            score += 0.1
            
        # パターンの複雑さによる信頼度
        if patterns.get('fixation_analysis'):
            score += 0.2
        if patterns.get('transition_analysis'):
            score += 0.2
        if patterns.get('temporal_analysis'):
            score += 0.2
            
        return min(score, 1.0)
        
    def generate_llm_prompt(self, structured_data: Dict[str, Any]) -> str:
        """
        LLM用のプロンプトを生成
        
        Args:
            structured_data: 構造化された分析データ
            
        Returns:
            LLM用プロンプト
        """
        basic_info = structured_data['basic_information']
        dominant_parts = structured_data['dominant_parts']
        behavioral_features = structured_data['behavioral_features']
        context_inference = structured_data['context_inference']
        
        prompt = f"""以下の視線行動データを専門的かつ分かりやすい日本語で分析・解説してください。

## 基本情報
- 観察時間: {basic_info['total_duration_seconds']:.2f}秒 ({basic_info['total_frames']}フレーム)
- フレームレート: {basic_info['frame_rate']}fps

## 主要注視部位（上位5位）
"""
        
        for i, part in enumerate(dominant_parts, 1):
            prompt += f"{i}. {part['part_name']}: {part['duration_seconds']:.2f}秒 ({part['percentage']:.1f}%)\n"
        
        prompt += f"""
## 行動特徴分析
### 注意の安定性
- レベル: {behavioral_features['attention_stability'].get('level', 'unknown')}
- 平均固視時間: {behavioral_features['attention_stability'].get('average_fixation_duration', 0):.2f}秒
- 注視変化頻度: {behavioral_features['attention_stability'].get('transition_frequency', 0):.2f}回/秒

### 探索行動
- 探索傾向: {behavioral_features['exploration_tendency'].get('tendency', 'unknown')}
- 探索多様性: {behavioral_features['exploration_tendency'].get('diversity_score', 0):.2f}

### スキャニング行動
- 強度: {behavioral_features['scanning_behavior'].get('intensity', 'unknown')}
- スキャニング割合: {behavioral_features['scanning_behavior'].get('scanning_percentage', 0):.1f}%

## コンテキスト推論
- 推定活動タイプ: {context_inference['likely_activity']}
- 注意戦略: {context_inference['attention_strategy']}
- エンゲージメントレベル: {context_inference['engagement_level']}

## 分析タスク
上記のデータに基づいて、以下の観点から視線行動を分析してください：

1. **行動パターンの特徴**: どのような視線行動パターンが観察されるか
2. **注意配分の傾向**: 注意がどのように配分されているか
3. **認知的プロセスの推論**: この視線行動から推測される認知的処理
4. **行動の意図や目的**: 観察者の意図や目的について
5. **専門的な知見**: 視線行動研究の観点からの解釈

分析結果は以下の構成で出力してください：
- **概要**: 全体的な視線行動の特徴
- **詳細分析**: 各指標の専門的解釈
- **行動解釈**: 認知的・心理的側面の考察
- **結論**: 主要な発見と示唆
"""
        
        return prompt
        
    def save_llm_analysis(self, output_dir: str, structured_data: Dict[str, Any], prompt: str) -> None:
        """
        LLM分析用データを保存
        
        Args:
            output_dir: 出力ディレクトリ
            structured_data: 構造化データ
            prompt: 生成されたプロンプト
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 構造化データを保存
        with open(os.path.join(output_dir, 'llm_structured_data.json'), 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)
            
        # プロンプトを保存
        with open(os.path.join(output_dir, 'llm_prompt.txt'), 'w', encoding='utf-8') as f:
            f.write(prompt)
            
        # 簡易サマリーを保存
        self._save_simple_summary(output_dir, structured_data)
        
    def _save_simple_summary(self, output_dir: str, structured_data: Dict[str, Any]) -> None:
        """簡易サマリーを保存"""
        basic_info = structured_data['basic_information']
        dominant_parts = structured_data['dominant_parts']
        context_inference = structured_data['context_inference']
        
        summary_path = os.path.join(output_dir, 'simple_summary.md')
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("# 視線行動分析サマリー\n\n")
            f.write(f"**観察時間**: {basic_info['total_duration_seconds']:.2f}秒\n")
            f.write(f"**主要注視部位**: {dominant_parts[0]['part_name'] if dominant_parts else '不明'}\n")
            f.write(f"**推定活動**: {context_inference['likely_activity']}\n")
            f.write(f"**注意戦略**: {context_inference['attention_strategy']}\n")
            f.write(f"**エンゲージメント**: {context_inference['engagement_level']}\n\n")
            f.write("詳細な分析結果は `llm_prompt.txt` をLLMに入力して取得してください。\n")
        
    def run_full_analysis(self, analysis_results_dir: str, output_dir: str) -> None:
        """
        完全分析を実行
        
        Args:
            analysis_results_dir: 基本分析結果ディレクトリ
            output_dir: LLM分析出力ディレクトリ
        """
        print("LLM言語化分析を開始します...")
        
        # データ読み込み
        data = self.load_analysis_results(analysis_results_dir)
        print("基本分析データ読み込み完了")
        
        # 行動パターン抽出
        patterns = self.extract_behavioral_patterns(
            data.get('transitions', pd.DataFrame()), 
            data.get('timeline', {})
        )
        print("行動パターン抽出完了")
        
        # 構造化データ作成
        structured_data = self.create_llm_structured_data(data, patterns)
        print("LLM用構造化データ作成完了")
        
        # プロンプト生成
        prompt = self.generate_llm_prompt(structured_data)
        print("LLMプロンプト生成完了")
        
        # 結果保存
        self.save_llm_analysis(output_dir, structured_data, prompt)
        print(f"LLM分析データを {output_dir} に保存しました")
        
        return structured_data, prompt


def main():
    """メイン実行関数"""
    import sys
    
    # デフォルトパス
    analysis_results_dir = "use-research-code/analysis_results"
    output_dir = "use-research-code/llm_analysis"
    
    # 引数からパスを取得
    if len(sys.argv) >= 2:
        analysis_results_dir = sys.argv[1]
    if len(sys.argv) >= 3:
        output_dir = sys.argv[2]
    
    # 分析実行
    analyzer = GazeLLMAnalyzer()
    analyzer.run_full_analysis(analysis_results_dir, output_dir)


if __name__ == "__main__":
    main()