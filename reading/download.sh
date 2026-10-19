#!/usr/bin/env bash
# Downloads everything in sources.tsv into reading/pdfs/ (arXiv) and reading/web/ (web pages).
# Both folders are gitignored: the files are third-party copyrighted material, so we keep links, not copies.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p pdfs web
grep -v '^#' sources.tsv | while IFS=$'\t' read -r kind src name; do
  [ -z "$kind" ] && continue
  if [ "$kind" = arxiv ]; then
    out="pdfs/${src}_${name}.pdf"
    [ -s "$out" ] || curl -sfL -o "$out" "https://arxiv.org/pdf/$src" || echo "FAILED $src"
  else
    out="web/${name}.html"
    [ -s "$out" ] || curl -sfL -A "Mozilla/5.0" -o "$out" "$src" || echo "FAILED $src"
  fi
  echo "ok  $out"
done
