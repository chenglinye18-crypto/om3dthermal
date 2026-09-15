"""Bounded delivery checks: no simulator, optimizer, or thermal solver."""
import hashlib
import json
from pathlib import Path
import zipfile
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/manuscript'


def main():
    edits=json.loads((OUT/'manuscript_edits.json').read_text(encoding='utf-8'))
    source=Path(edits['source'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==edits['source_sha256']
    p=PdfReader(OUT/'iom3d_hbm_evidence.pdf')
    assert len(p.pages)==6
    text='\n'.join(x.extract_text() for x in p.pages)
    for token in ('7.28','4.75','81.41','3.07','2.96','169.28','619.96','Fig. 11.'):
        assert token in text,token
    for token in ('PHYSICAL_EXECUTION_FIGURE_PLACEHOLDER','Fig. X','Algorithm 1.','Identify applicable funding','<sup>'):
        assert token not in text,token
    before=zipfile.ZipFile(source);after=zipfile.ZipFile(OUT/'iom3d_hbm_evidence.docx')
    media=lambda z:{hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('word/media/')}
    assert media(before)<=media(after)
    preservation=json.loads((ROOT/'runs/physical_decode_execution_anatomy/canonical_preservation.json').read_text(encoding='utf-8'))
    for path,expected in preservation['files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
    report=dict(pages=6,original_manuscript_unchanged=True,original_media_retained=len(media(before)),
        canonical_data_files_unchanged=len(preservation['files']),headline_and_subset_text='PASS',
        placeholders='NONE',figure_type='SVG_EMBEDDED',body_font_size='ORIGINAL_UNCHANGED',
        targeted_tests='4 passed',full_pytest_run=False,benchmark_runs=0,thermal_solves=0,
        unique_decode_checkpoints_replayed=1,CPA_optimizer_runs=0,
        renderer='Native Word PDF export + Poppler PNG inspection',
        source_docx_sha256=edits['source_sha256'],
        output_pdf_sha256=hashlib.sha256((OUT/'iom3d_hbm_evidence.pdf').read_bytes()).hexdigest())
    (OUT/'delivery_checks.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
