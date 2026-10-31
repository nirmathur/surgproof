#!/bin/zsh
# usage: lanes/run.sh <lane>   architect | data | manual | label | student | demo
lane=$1
[ "$lane" = architect ] && dir=~/surgatlas || dir=~/surgatlas-lanes/$lane
cd "$dir" || exit 1
[ -f ~/surgatlas/.env ] && set -a && source ~/surgatlas/.env && set +a
[ -z "$GEMINI_API_KEY" ] && { echo "No GEMINI_API_KEY"; exit 1; }
export GEMINI_CLI_TRUST_WORKSPACE=true
mkdir -p ~/surgatlas/out/lanes
[ -f ~/surgatlas/out/lanes/$lane.md ] || printf 'PHASE: A\nSTATUS: running\nARTIFACTS: none\nNUMBERS: none\nBLOCKED ON: none\nREQUESTS: none\n' > ~/surgatlas/out/lanes/$lane.md
echo "lane=$lane dir=$PWD branch=$(git branch --show-current)"
exec gemini -m "${ARGON_MODEL:-gemini-4-argon}" --yolo -i "$(cat ~/surgatlas/lanes/$lane.txt) Read GEMINI.md first, then start Phase A."
