#!/bin/zsh
# Opens one Terminal window per lane.
for lane in architect data manual label student demo promo coach verifier gate director; do
  osascript -e "tell application \"Terminal\" to do script \"~/surgatlas/lanes/run.sh $lane\"" >/dev/null
done
