# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

SMPL（Skinned Multi-Person Linear Model）を使用した 3D ヒューマンメッシュ処理ライブラリです。4D-Humans の出力データの処理と SMPL メッシュの部位別着色機能を提供します。

# Development Guidelines

This document contains critical information about working with this codebase.
Follow these guidelines precisely.

## Rules

1. Package Management

   - ONLY use uv, NEVER pip
   - Installation: `uv add package`
   - Upgrading: `uv add --dev package --upgrade-package package`
   - FORBIDDEN: `uv pip install`, `@latest` syntax

2. Code Quality

   - Type hints required for all code
   - Follow existing patterns exactly
   - Use Google style for docstring

3. Testing Requirements

   - Framework: `uv run --frozen pytest`
   - Coverage: test edge cases and errors
   - New features require tests
   - Bug fixes require regression tests

4. Git
   - Follow the Conventional Commits style on commit messages.

## Code Formatting and Linting

1. Ruff
   - Format: `uv run --frozen ruff format .`
   - Check: `uv run --frozen ruff check .`
   - Fix: `uv run --frozen ruff check . --fix`
2. Pre-commit
   - Config: `.pre-commit-config.yaml`
   - Runs: on git commit
   - Tools: Ruff (Python)

## 開発コマンド

このプロジェクトは標準的な Python スクリプトとして実行されます。特別なビルドシステムは使用していません。

```bash
# スクリプトの実行例
uv run src/pkl2obj.py
uv run src/obj2obj.py
```

## コード構造

### 主要モジュール

- `src/pkl2obj.py` - 4D-Humans PKL ファイルから SMPL メッシュを生成
- `src/obj2obj.py` - 既存の OBJ ファイルに部位別着色を適用

### 主要関数

**pkl2obj.py:**

- `load_4dhumans_pkl()` - zlib で圧縮された joblib ファイルを読み込み
- `process_4d_humans_pkl_smpl()` - PKL ファイルから SMPL メッシュを生成・着色
- `download_segmentation_file()` - 頂点セグメンテーションファイルをダウンロード

**obj2obj.py:**

- `color_obj_smpl()` - OBJ ファイルを読み込み、部位別着色を適用
- `color_part_on_mesh()` - trimesh メッシュに部位別着色を適用

## 使用例

```python
# PKLファイルからSMPLメッシュを生成
from src.pkl2obj import process_4d_humans_pkl_smpl

mesh = process_4d_humans_pkl_smpl(
    pkl_path="demo.pkl",
    smpl_model_path="SMPL_NEUTRAL.npz",
    frame_idx=0,
    person_idx=0,
    part_name="head",
    color=[1.0, 0.0, 0.0]
)

# 既存OBJファイルに着色
from src.obj2obj import color_obj_smpl

mesh = color_obj_smpl(
    obj_path="model.obj",
    part_name="leftHand",
    color=[0.0, 1.0, 0.0]
)
```

## 注意事項

- SMPL モデルファイル（.pkl または .npz）は別途用意する必要があります
- 頂点セグメンテーションファイルは初回実行時に自動ダウンロードされます
- 対応する部位名は `smpl_vert_segmentation.json` に定義されています
