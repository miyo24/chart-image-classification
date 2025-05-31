#!/bin/bash

# start_date="2010-01-10"
# end_date="2025-01-10"
# 
# current_date="$start_date"
# while [[ "$current_date" != "$(date -j -v+1m -f "%Y-%m-%d" "$end_date" +"%Y-%m-%d")" ]]; do
#     echo "$current_date"
#     next_date=$(date -j -v+1d -f "%Y-%m-%d" "$current_date" +"%Y-%m-%d")
# 
#     while :; do
#         NODE_TLS_REJECT_UNAUTHORIZED=0 npx dukascopy-node -i usdjpy \
#             -from "$current_date" \
#             -to "$next_date" \
#             -t tick \
#             -f csv \
#             --date-format "YYYY-MM-DD HH:mm:ss" \
#             -p 'ask' \
#             -fn "${current_date}"
# 
#         # ダウンロードしたファイルサイズが100000byte以下なら日付を1日追加して再試行
#         if [ $(stat -f "%z" "download/${current_date}.csv") -lt 1000000 ]; then
#             echo "ファイルが小さすぎます。翌日の日付で再試行します。"
#             rm "download/${current_date}.csv"
#             current_date=$(date -j -v+1d -f "%Y-%m-%d" "$current_date" +"%Y-%m-%d")
#             next_date=$(date -j -v+1d -f "%Y-%m-%d" "$next_date" +"%Y-%m-%d")
#         else
#             echo "ダウンロード成功: ${current_date}.csv"
#             break
#         fi
#     done
# 
#     current_date=$(date -j -v+1m -f "%Y-%m-%d" "$current_date" +"%Y-%m-%d")
# done

start_date="2025-02-01"
end_date="2025-02-28"

current_date="$start_date"
while [[ "$current_date" != "$(date -j -v+1d -f "%Y-%m-%d" "$end_date" +"%Y-%m-%d")" ]]; do
    echo "$current_date"
    next_date=$(date -j -v+1d -f "%Y-%m-%d" "$current_date" +"%Y-%m-%d")

    NODE_TLS_REJECT_UNAUTHORIZED=0 npx dukascopy-node -i usdjpy \
        -from "$current_date" \
        -to "$next_date" \
        -t tick \
        -f csv \
        --date-format "YYYY-MM-DD HH:mm:ss" \
        -p 'ask' \
        -fn "${current_date}"

    # ダウンロードしたファイルサイズが100000byte以下なら日付を1日追加して再試行
    if [ $(stat -f "%z" "download/${current_date}.csv") -lt 1000000 ]; then
        echo "ファイルが小さすぎます。削除します。"
        rm "download/${current_date}.csv"
    else
        echo "ダウンロード成功: ${current_date}.csv"
    fi

    current_date=$(date -j -v+1d -f "%Y-%m-%d" "$current_date" +"%Y-%m-%d")
done
