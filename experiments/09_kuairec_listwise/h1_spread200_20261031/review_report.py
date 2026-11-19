"""Bind the actual four-figure visual review and quantitative report checks."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image

here=Path(__file__).resolve().parent
out=here/'final_report_r2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
binding=read(out/'REPORT_BINDING.json')
for name,digest in binding['outputs'].items():
    if sha(out/name)!=digest:raise ValueError('Generated report output changed')
analysis=read(here/'final_analysis/ANALYSIS.json')
text=(out/'REPORT.md').read_text(encoding='utf-8')
for metric,m in analysis['comparisons']['fixed_test'].items():
    expected=f'| {metric} | {m["baseline"]:.8f} | {m["treatment"]:.8f} | {m["delta"]:+.8f} |'
    if expected not in text:raise ValueError('Fixed-test report table not bound to accepted statistics')
figures={}
for name in ('validation_curve','T_block_response','tokenizer_stability','temporal_generalization'):
    png=out/(name+'.png');viewed=here/'final_report'/(name+'.png')
    if sha(png)!=sha(viewed):raise ValueError('Final figure differs from image actually viewed')
    with Image.open(png) as image:
        width,height=image.size;image.verify()
    svg=ET.parse(out/(name+'.svg')).getroot()
    if not svg.tag.endswith('svg') or 'viewBox' not in svg.attrib:raise ValueError('Invalid standalone SVG')
    for element in svg.iter():
        for key,value in element.attrib.items():
            if key.endswith('href') and not value.startswith(('#','data:')):
                raise ValueError('SVG depends on an external asset')
    pdf=(out/(name+'.pdf')).read_bytes()
    if not pdf.startswith(b'%PDF-') or not pdf.rstrip().endswith(b'%%EOF'):raise ValueError('Invalid PDF envelope')
    figures[name]=dict(png_sha256=sha(png),viewed_reference_sha256=sha(viewed),width=width,height=height,
        svg_sha256=sha(out/(name+'.svg')),pdf_sha256=sha(out/(name+'.pdf')))
result=dict(status='passed_manual_PNG_visual_review_and_report_table_binding',
    report_sha256=sha(out/'REPORT.md'),generated_binding_sha256=sha(out/'REPORT_BINDING.json'),
    analysis_sha256=sha(here/'final_analysis/ANALYSIS.json'),reviewer_script_sha256=sha(Path(__file__)),
    figures=figures,manual_review=dict(
        actually_viewed_four_PNGs=True,final_PNG_bytes_identical_to_viewed_images=True,
        axes_ticks_legends_and_labels_readable=True,no_visible_clipping_or_overlap=True,
        fixed_endpoint_and_unequal_budget_best_separated=True,
        SID_collision_and_candidate_filling_not_confused_with_recommendation_quality=True,
        temporal_absolute_gap_change_not_claimed_as_better_generalization=True,
        facts_hypotheses_and_single_seed_limits_explicit=True),
    vector_scope='SVG XML and embedded references plus PDF envelope checked; manual visual review is of the exact PNGs exported from the same Matplotlib figures.',
    limitation='Science source audits, archive/cold restoration and resource closure remain separate evidence.')
with (out/'VISUAL_REVIEW_ACCEPTED.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(status=result['status'],figures=len(figures),report_sha256=result['report_sha256'])))
