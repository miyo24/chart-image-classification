#!/bin/bash

# 実行対象のディレクトリ指定
target_directory="./download"

# 並列ジョブ数の制限
MAX_JOBS=5
PYTHON_SCRIPT="generate.py"
source venv/bin/activate
# ディレクトリ内のすべてのファイルに対してPythonスクリプトを並列実行
ls "$target_directory"/* | awk 'NR % 5 == 1' | while read file; do
    if [ -f "$file" ]; then  # ファイルのみを対象
        filename=$(basename "$file" .csv)  # .csvを除いたファイル名を取得
        echo "Processing: $filename"
        echo "python3 "$PYTHON_SCRIPT" "$filename" > logs/"$filename""
    fi
done | parallel -j $MAX_JOBS
