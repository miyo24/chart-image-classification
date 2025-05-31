#!/bin/bash

# 引数でディレクトリを受け取る（例: ./delete_even_images.sh /path/to/images）
TARGET_DIR="$1"

if [ -z "$TARGET_DIR" ]; then
  echo "Usage: $0 /path/to/images"
  exit 1
fi

# 対象の画像拡張子（必要に応じて変更）
IMAGE_EXTENSIONS="png"

# 拡張子ごとのループ
for ext in $IMAGE_EXTENSIONS; do
  # findで対象画像ファイルをすべて取得し、ソートしてインデックス番号をつける
  find "$TARGET_DIR" -type f -iname "*.${ext}" | sort | nl | while read -r index filepath; do
    # インデックスが2の倍数なら削除
    if [ $((index % 2)) -eq 0 ]; then
      echo "Deleting: $filepath"
      rm "$filepath"
    fi
  done
done
