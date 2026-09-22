#!/usr/bin/env bash
# Build with standard TeX Live / MiKTeX tools. Run from any working directory.
set -euo pipefail
cd "$(dirname "$0")"
python3 scripts/export_tables.py
if command -v bibtex >/dev/null 2>&1; then
  BIBTEX="$(command -v bibtex)"
elif command -v bibtex.original >/dev/null 2>&1; then
  BIBTEX="$(command -v bibtex.original)"
elif command -v bibtex8 >/dev/null 2>&1; then
  BIBTEX="$(command -v bibtex8)"
else
  echo "BibTeX is required. Install the bibliography tools of your TeX distribution." >&2
  exit 1
fi
pdflatex -interaction=nonstopmode -halt-on-error main.tex
"$BIBTEX" main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
printf '\nBuilt main.pdf. Optional local preflight: python3 scripts/check_pdf.py main.pdf\n'
