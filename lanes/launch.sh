#!/bin/zsh
# Opens one Terminal window per lane.
for lane in architect data manual label student demo promo; do
  osascript -e "tell application \"Terminal\" to do script \"~/surgatlas/lanes/run.sh $lane\"" >/dev/null
done
