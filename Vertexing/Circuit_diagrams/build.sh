#!/bin/bash
cd "$(dirname "$0")"
for f in "$@"; do
  pdflatex -interaction=nonstopmode $f.tex > /dev/null 2>&1 || { grep -E "^!" -A3 $f.log | head -8; }
  pdftoppm -png -r 200 -singlefile $f.pdf $f
  pdftoppm -png -r 90 -singlefile $f.pdf ${f}_prev
done
