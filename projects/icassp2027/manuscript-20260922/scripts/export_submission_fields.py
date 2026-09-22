"""Extract title / abstract / keywords from main.tex into submissions/*.txt.

The submission form needs plain text, so LaTeX escapes are unwrapped here.
Run from the manuscript root:  python scripts/export_submission_fields.py
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "submissions"


def unwrap(text: str) -> str:
    text = text.replace("\\\\", " ")                # LaTeX line break -> space
    text = re.sub(r"\\(%)", r"\1", text)          # \% -> %
    text = text.replace("--", "–")                 # en dash
    text = re.sub(r"\$([^$]*)\$", r"\1", text)     # inline math -> plain
    text = re.sub(r"\\[A-Za-z]+\s*", "", text)     # leftover macros
    text = re.sub(r"[{}]", "", text)
    return " ".join(text.split())


def main() -> int:
    tex = (ROOT / "main.tex").read_text(encoding="utf-8")

    title = re.search(r"\\title\{([^}]*)\}", tex)
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    keywords = re.search(r"\\begin\{keywords\}(.*?)\\end\{keywords\}", tex, re.S)
    if not (title and abstract and keywords):
        print("could not locate title/abstract/keywords in main.tex", file=sys.stderr)
        return 1

    OUT.mkdir(exist_ok=True)
    (OUT / "title.txt").write_text(unwrap(title.group(1)) + "\n", encoding="utf-8")
    (OUT / "abstract.txt").write_text(unwrap(abstract.group(1)) + "\n", encoding="utf-8")
    (OUT / "keywords.txt").write_text(unwrap(keywords.group(1)) + "\n", encoding="utf-8")

    words = len((OUT / "abstract.txt").read_text(encoding="utf-8").split())
    print(f"wrote submissions/title.txt, abstract.txt ({words} words), keywords.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
