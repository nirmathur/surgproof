#!/bin/zsh
# Writes the Gemini key from the clipboard into .env, refusing anything that isn't a key.
k=$(pbpaste | tr -d '[:space:]')
case "$k" in
  AIza*) [ ${#k} -ge 35 ] || { echo "Clipboard starts like a key but is too short (${#k} chars). Copy it again."; exit 1; } ;;
  *) echo "Clipboard is not a Gemini key (starts with '${k:0:4}'). Copy the key in AI Studio, then rerun."; exit 1 ;;
esac
cd "${0:A:h}/.." && umask 077 && printf 'GEMINI_API_KEY=%s\n' "$k" > .env && chmod 600 .env
echo "Saved key (${#k} chars, ${k:0:4}...) to .env"
