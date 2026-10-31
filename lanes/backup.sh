#!/bin/zsh
# Every 10 min: outputs, data index/features, cache and a git bundle to gdrive:SurgProof/backup. raw/ excluded (re-downloadable).
while :; do
  T=$TMPDIR/sp_backup; mkdir -p $T
  sqlite3 ~/surgatlas/cache.sqlite ".backup '$T/cache.sqlite'" 2>/dev/null
  git -C ~/surgatlas bundle create $T/surgproof.bundle --all 2>/dev/null
  rclone copy ~/surgatlas/out gdrive:SurgProof/backup/out --exclude '*.tmp' -q
  rclone copy ~/surgatlas/data gdrive:SurgProof/backup/data --exclude 'external/**/frames/**' -q
  rclone copy $T gdrive:SurgProof/backup/snap -q
  date "+%H:%M:%S" | rclone rcat gdrive:SurgProof/backup/LAST_BACKUP
  sleep 600
done
