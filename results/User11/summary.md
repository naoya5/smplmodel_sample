most_attended_frames は上位 5 個の部位を示す
この項目は「各部位が最も注視されたフレーム数」の
ランキングです：

1. spine2: 438 フレーム（46.3%）
2. hips: 81 フレーム（8.6%）
3. head: 73 フレーム（7.7%）
4. spine1: 66 フレーム（7.0%）
5. rightLeg: 55 フレーム（5.8%）

top_parts_by_probability との違い

- top_parts_by_probability:
  総視線量による上位 10 部位
- most_attended_frames:
  フレーム単位で「最も注視された回数」の上位 5 部位

興味深い点: head は総視線量では 2 位ですが、「最も
注視されたフレーム数」では 3 位になっています。こ
れは、head への視線は強度は高いが、継続的な注視よ
りも瞬間的な注視が多いことを示唆しています。

2. all_parts_frame_analysis: 各部位について以下の詳細カウントを追加：
   - frame_count_with_gaze: 視線値が 0 より大きいフレーム数
   - frame_count_without_gaze: 視線値が 0 のフレーム数
   - percentage_with_gaze: 視線ありフレームの割合
   - percentage_without_gaze: 視線なしフレームの割合

データから読み取れること：

- head: 総視線量 2 位（22.7%）だが、「最も注視されたフ
  レーム」では 0 フレーム
- spine2: 総視線量 1 位（36.7%）で、「最も注視されたフ
  レーム」でも 438 フレーム（46.3%）で 1 位

これが意味すること：

head は「強度の高い瞬間的な視線」を受けているが、どの
フレームでも他の部位（主に spine2）の方が視線値が高い

具体例：

- あるフレームで head = 100、spine2 = 150 の場合
- そのフレームの「最大注視部位」は spine2 になる
- しかし、head も高い視線値（100）を獲得している

つまり、head は常に spine2 に次ぐ"2 番手"として安定して
視線を集めているが、spine2 を上回ることはないという視
線パターンを示しています。

これは人間の視覚的注意が体の中心（spine2）を基準とし
つつ、頭部も継続的に意識していることを表しています。
