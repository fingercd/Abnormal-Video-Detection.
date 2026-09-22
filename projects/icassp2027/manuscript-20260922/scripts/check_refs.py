"""Verify every \\cite key in the manuscript resolves to an entry in references.bib."""
import re
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parent.parent
bib = root.joinpath('references.bib').read_text(encoding='utf-8')
keys = set(re.findall(r'@\w+\{\s*([^,\s]+)\s*,', bib))

cited = set()
for p in sorted(root.glob('sections/*.tex')) + [root / 'main.tex'] + sorted(root.glob('tables/*.tex')) + sorted(root.glob('figures/*.tex')):
    for m in re.finditer(r'\\cite\{([^}]*)\}', p.read_text(encoding='utf-8')):
        cited.update(k.strip() for k in m.group(1).split(',') if k.strip())

missing = sorted(cited - keys)
unused = sorted(keys - cited)
print('bib entries          :', len(keys))
print('keys cited in text   :', len(cited))
print('MISSING (render as ?):', missing or 'none')
print('uncited entries      :', unused or 'none')
sys.exit(1 if missing or unused else 0)
