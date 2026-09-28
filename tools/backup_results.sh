#!/bin/bash
# 每隔 10 分钟把正在写的生成结果另存一份。
#
#   nohup bash tools/backup_results.sh > /dev/null 2>&1 &
#
# 19:57 那次 rebuttal/mmstar/ 连同所有 cgr-* 文件一起消失，正在跑的 5 小时
# 结果直接归零。生成是贪心的、可复现，但重跑要 5 小时，所以边跑边留一份副本
# 更便宜。只复制 jsonl 生成结果，不复制图片和日志。
set -u
SRC=/home/lizhihao/phd/VRGA/rebuttal
DST=/home/lizhihao/vrga_backup

while true; do
    stamp=$(date +%H%M)
    for model_dir in "$SRC"/*/; do
        dataset=$(basename "$model_dir")
        [ "$dataset" = "logs" ] && continue
        [ "$dataset" = "scores" ] && continue
        [ "$dataset" = "figs" ] && continue
        [ "$dataset" = "_superseded" ] && continue
        for f in "$model_dir"*.jsonl; do
            [ -f "$f" ] || continue
            target="$DST/$dataset/$(basename "$model_dir")"
            mkdir -p "$target"
            cp -f "$f" "$target/" 2>/dev/null
        done
    done
    sleep 600
done
