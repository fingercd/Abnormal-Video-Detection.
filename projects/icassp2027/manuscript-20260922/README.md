# PairSelect: four-system r02 rebuild

This self-contained source package was regenerated from the user-uploaded r02 evidence archive after the previously linked generated files were unavailable. It is an author working draft, not a promise of byte identity with the unavailable files.

## Build

Open `main.tex` in Overleaf (pdfLaTeX), or run locally:

```bash
bash scripts/build.sh
```

Requires Python 3 and a standard TeX Live/MiKTeX installation with BibTeX. The script regenerates all numerical tables from the bundled JSON/CSV snapshots, then compiles the PDF. No models are run.

## Contents

- `main.pdf`: rebuilt four-system manuscript, five pages.
- `sections/`, `figures/`, `references.bib`, `author_config.tex`: editable source.
- `evidence_snapshot/`: 62 source evidence files from the uploaded r02 package.
- `data/four_systems_results.json`: all extracted quality and efficiency records.
- `data/numeric_provenance.csv`: 340 source-field trace records.
- `data/evidence_sha256.csv`: snapshot integrity checks.
- `notes/CLAIM_EVIDENCE_MAP.md`: Chinese claim/evidence map and remaining author checks.
- `notes/REBUILD_HANDOFF.md`: rebuilding details.

Preserves four-system training identities, UCF step AP, distinct XD metric definitions, actual budgets, and encoder-only timing in the manuscript; complete-video timing evidence remains in the bundled records. Full-encoder FLOPs and energy were not measured and are not invented.
