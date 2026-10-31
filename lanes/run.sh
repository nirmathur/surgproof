#!/bin/zsh
# usage: lanes/run.sh <lane>   architect | data | manual | label | student | demo
lane=$1
[ "$lane" = architect ] && dir=~/surgatlas || dir=~/surgatlas-lanes/$lane
cd "$dir" || exit 1
[ -f ~/surgatlas/.env ] && set -a && source ~/surgatlas/.env && set +a
[ -z "$GEMINI_API_KEY" ] && { echo "No GEMINI_API_KEY"; exit 1; }
mkdir -p ~/surgatlas/out/lanes
[ -f ~/surgatlas/out/lanes/$lane.md ] || printf 'PHASE: A\nSTATUS: running\nARTIFACTS: none\nNUMBERS: none\nBLOCKED ON: none\nREQUESTS: none\n' > ~/surgatlas/out/lanes/$lane.md
echo "lane=$lane dir=$PWD branch=$(git branch --show-current)"
exec ~/.local/bin/agy --model "${AGY_MODEL:-gemini-4-argon-high}" --dangerously-skip-permissions --add-dir ~/surgatlas -i "$(cat ~/surgatlas/lanes/$lane.txt) Read GEMINI.md first. If out/lanes/$lane.md or your branch already shows progress from an earlier worker, continue from it rather than restarting. Read GEMINI.md now, especially FOCUSED BUILD MODE and Keep each other updated. First action: post your plan on GitHub issue #1."
