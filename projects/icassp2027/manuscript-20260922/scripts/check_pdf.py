#!/usr/bin/env python3
"""Local ICASSP working-draft preflight, not a venue validation service.
Requires PyMuPDF. Poppler's pdffonts provides an additional font embedding check.
"""
from __future__ import annotations
import argparse, json, re, shutil, subprocess
from pathlib import Path

def check(pdf: Path) -> dict:
    try:
        import fitz
    except ImportError as exc:
        raise SystemExit('Optional preflight requires PyMuPDF (python package pymupdf).') from exc
    root=Path(__file__).resolve().parents[1]
    if not pdf.is_file(): raise FileNotFoundError(pdf)
    doc=fitz.open(pdf)
    failures=[]
    pages=[]
    if doc.page_count!=5: failures.append(f'Expected this draft to have 4 technical + 1 declaration/reference pages; found {doc.page_count}')
    if pdf.stat().st_size>5_000_000:failures.append('PDF exceeds 5 MB')
    if doc.is_encrypted:failures.append('PDF is encrypted')
    for index,page in enumerate(doc):
        if abs(page.rect.width-612)>0.5 or abs(page.rect.height-792)>0.5:failures.append(f'Page {index+1} is not Letter')
        spans=[s for b in page.get_text('dict')['blocks'] for line in b.get('lines',[]) for s in line['spans'] if s['text'].strip()]
        prose=[s for s in spans if 'NimbusRom' in s['font'] or 'Times' in s['font']]
        small=[{'text':s['text'],'size_pdf_points':s['size']} for s in prose if s['size']<8.90]
        if small:failures.append(f'Page {index+1} has prose smaller than nominal 9 TeX points')
        box=[min(s['bbox'][0] for s in spans),min(s['bbox'][1] for s in spans),max(s['bbox'][2] for s in spans),max(s['bbox'][3] for s in spans)]
        # Rounded nominal print area, with modest tolerance for font bounding boxes.
        if box[0]<51 or box[2]>561 or box[1]<(90 if index==0 else 68) or box[3]>726:
            failures.append(f'Page {index+1} has text outside the local print-area tolerance: {box}')
        text=page.get_text()
        if re.search(r'\[\?\]|\?\?',text):failures.append(f'Page {index+1} has unresolved markers')
        pages.append({'page':index+1,'size_points':[page.rect.width,page.rect.height],'text_bounds_points':[round(x,3) for x in box],'min_prose_size_pdf_points':round(min(s['size'] for s in prose),4),'prose_smaller_than_9texpt':small})
    if doc.page_count>=5:
        fifth=doc[4].get_text()
        if not all(term in fifth for term in ('COMPLIANCE WITH ETHICAL STANDARDS','ACKNOWLEDGMENTS','REFERENCES')):failures.append('Page 5 missing expected permitted headings')
        if re.search(r'\b(?:Table|Fig\.)\s*\d|\b(?:PAIRSELECT|EXPERIMENTS|CONCLUSION)\b',fifth):failures.append('Possible technical material on page 5')
    aux=root/'main.aux'; last_page=None
    if aux.exists():
        m=re.search(r'\\newlabel\{lasttechnical\}\{\{[^}]*\}\{(\d+)\}',aux.read_text())
        if m:
            last_page=int(m.group(1))
            if last_page>4:failures.append(f'Last technical text is on page {last_page}')
    log=root/'main.log'; warning_lines=[]
    if log.exists():
        warning_lines=[line for line in log.read_text(errors='replace').splitlines() if re.search(r'Overfull|undefined|Float too large|LaTeX Warning:',line)]
        if warning_lines:failures.append('Inspect LaTeX warnings')
    font_report='pdffonts unavailable; inspect embedding separately'; font_ok=None
    if shutil.which('pdffonts'):
        font_report=subprocess.run(['pdffonts',str(pdf)],check=True,capture_output=True,text=True).stdout
        font_rows=font_report.splitlines()[2:]
        font_ok=bool(font_rows) and all(re.search(r'\byes\s+yes\s+(yes|no)\s+\d+\s+\d+\s*$',r) for r in font_rows) and 'Type 3' not in font_report
        if not font_ok:failures.append('Font embedding/subsetting or Type3 check failed')
    main=(root/'main.tex').read_text();abstract=main.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]
    words=len(abstract.split())
    if not 100<=words<=150:failures.append(f'Abstract word count {words} outside 100--150')
    report={'status':'PASS_LOCAL_FORMAT_CHECK' if not failures else 'REVIEW_REQUIRED','scope':'Local technical format preflight only. Not a venue acceptance, research-validity or LLM-policy certification.','pdf':pdf.name,'page_count':doc.page_count,'last_technical_text_page':last_page,'pdf_size_bytes':pdf.stat().st_size,'encrypted':doc.is_encrypted,'abstract_whitespace_word_count':words,'font_embedded_and_subset_no_type3':font_ok,'font_report':font_report,'pages':pages,'latex_warnings':warning_lines,'failures':failures,'expected_pending':'Two quality budgets, all hardware measurements, author declarations, F04 raw observation evidence, substantive author writing.'}
    return report

def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('pdf',nargs='?',default='main.pdf');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    report=check(Path(args.pdf).resolve())
    (root/'notes/PDF_PREFLIGHT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','page_count','last_technical_text_page','pdf_size_bytes','abstract_whitespace_word_count','font_embedded_and_subset_no_type3','failures')},ensure_ascii=False,indent=2))
    if report['failures']:raise SystemExit(1)

if __name__=='__main__':main()
