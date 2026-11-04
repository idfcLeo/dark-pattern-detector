from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether

OUT = Path('output/pdf/response_to_reviewers_icmlde.pdf')
OUT.parent.mkdir(parents=True, exist_ok=True)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Title2', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=16, leading=20, alignment=TA_CENTER, spaceAfter=6))
styles.add(ParagraphStyle(name='Sub', parent=styles['Normal'], fontName='Helvetica', fontSize=9.5, leading=13, alignment=TA_CENTER, textColor=colors.HexColor('#4b5563'), spaceAfter=14))
styles.add(ParagraphStyle(name='H1x', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=12, leading=15, textColor=colors.HexColor('#17365d'), spaceBefore=10, spaceAfter=5))
styles.add(ParagraphStyle(name='H2x', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=10.5, leading=13, textColor=colors.HexColor('#17365d'), spaceBefore=8, spaceAfter=3))
styles.add(ParagraphStyle(name='Bodyx', parent=styles['BodyText'], fontName='Helvetica', fontSize=9.2, leading=13, spaceAfter=5))
styles.add(ParagraphStyle(name='Quote', parent=styles['BodyText'], fontName='Helvetica-Oblique', fontSize=8.9, leading=12.2, leftIndent=10, rightIndent=6, textColor=colors.HexColor('#374151'), spaceBefore=2, spaceAfter=3))
styles.add(ParagraphStyle(name='Small', parent=styles['BodyText'], fontName='Helvetica', fontSize=8.2, leading=10.5, textColor=colors.HexColor('#374151')))

def P(t, style='Bodyx'):
    return Paragraph(t, styles[style])

def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#d1d5db'))
    canvas.line(18*mm, 14*mm, 192*mm, 14*mm)
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#6b7280'))
    canvas.drawString(18*mm, 9.5*mm, 'Response to Reviewers - ICMLDE 5.0')
    canvas.drawRightString(192*mm, 9.5*mm, f'Page {doc.page}')
    canvas.restoreState()

doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=16*mm, bottomMargin=20*mm)
story = []
story += [P('Response to Reviewers', 'Title2'), P('AI-Based Detection and Analysis of Dark Patterns: A Text-Based Baseline with a CCPA-Aware Multimodal Research Framework', 'Sub')]
story += [P('Author statement', 'H1x'), P('We thank the reviewers for their constructive feedback. The revised manuscript will preserve a strict distinction between completed evidence and planned work. Specifically, the current empirical contribution is a TF-IDF + logistic-regression text baseline on the public ec-darkpattern corpus; the multimodal, India-focused, and CCPA-mapping components are a proposed framework until their data collection, annotation, and evaluation are complete. We will not report unrun transformer, cross-site, multimodal, or regulatory-mapping results as completed experiments.')]
story += [P('Revision summary', 'H1x')]
rows = [
    [P('<b>Revision</b>','Small'), P('<b>Action in the next manuscript version</b>','Small')],
    [P('Scope and novelty','Small'), P('Reframe novelty as the proposed India-focused CCPA-aware evaluation framework and transparent baseline, not as novelty of TF-IDF/logistic regression.','Small')],
    [P('Data and annotation','Small'), P('Expand dataset provenance, source, label schema, split policy, annotation protocol, annotator qualifications, and agreement statistic.','Small')],
    [P('Evaluation','Small'), P('Add transformer and prior-system comparisons only after reproducing them under comparable splits; add leave-one-site-out testing and per-category outcomes with uncertainty.','Small')],
    [P('CCPA and ethics','Small'), P('Add a CCPA-category crosswalk, evidence-retention rule, limitations, legal non-advice disclaimer, privacy/consent, and dataset-release governance.','Small')],
    [P('Framework communication','Small'), P('Add a clearly labelled proposed-system diagram and separate “completed baseline” from “future multimodal implementation”.','Small')],
]
tbl = Table(rows, colWidths=[38*mm, 136*mm], repeatRows=1)
tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dbeafe')),('GRID',(0,0),(-1,-1),0.35,colors.HexColor('#cbd5e1')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
story += [tbl, Spacer(1,5)]

def response(reviewer, comment, answer):
    story.append(P(reviewer, 'H1x'))
    story.append(P('<b>Comment.</b> ' + comment, 'Quote'))
    story.append(P('<b>Response and revision.</b> ' + answer))

response('Reviewer 1', 'The novelty of using TF-IDF logistic regression as baseline is low.', 'Agreed. We will remove any implication that the classifier itself is novel. The revised contribution statement will describe it as a transparent, reproducible reference baseline. The research contribution is the evaluation design: an India-focused collection plan, a CCPA-aware interpretation layer, and a proposed multimodal extension. The title and abstract will make the baseline/proposed-framework distinction explicit.')
response('Reviewer 1', 'The source and labeling of the 2,356 samples should be specified.', 'The manuscript already identifies the source as the ec-darkpattern text dataset of Yada et al. (2022), derived from texts reported by Mathur et al. (2019) plus non-dark-pattern texts from the same e-commerce sites. We will expand this into a reproducibility subsection: source/version, the four released fields, binary-label definition, pattern-category provenance, preprocessing, duplicate handling, random seed, and the stratified 80/20 split. We will also state that the reported baseline uses this public corpus rather than the future India-collected corpus.')
response('Reviewer 1', 'Comparison with transformer baselines and with existing dark pattern detectors is required.', 'Agreed. We will add a comparative evaluation with a transformer text baseline (for example, DistilBERT or RoBERTa) trained on the identical split, together with a related-work comparison that separates incomparable screenshot/app systems from directly comparable text classifiers. We will report accuracy, macro-F1, class-wise precision/recall/F1, training setup, and confidence intervals. Until these runs are complete, the claim that the baseline is competitive with transformers will be removed.')
response('Reviewer 1', 'Multimodal framework should be implemented prior to making claims about it.', 'Agreed. We will constrain claims to the completed text baseline. The framework figure will be labelled “proposed” and all future-tense components will be moved to a Future Work subsection. A later revision will report visual, fusion, and CCPA-mapping results only after implementation and held-out evaluation.')

response('Reviewer 2', 'Error analysis is useful and can be expanded with examples.', 'We will expand the error analysis into a table of anonymized held-out examples: text, gold label, prediction, confidence, error type, and interpretation. The present two failure modes - formatted numeric urgency false negatives and neutral stock-status false positives - will be retained, with counts and category/site breakdowns added where supported by the data.')
response('Reviewer 2', 'Connection to CCPA categories should be shown.', 'We will add a CCPA crosswalk table that links each annotation category to the relevant 2023 Guideline category and specifies the retained evidence (text span and/or screenshot region). We will state that this is an interpretive mapping, not a legal determination, and report category-level agreement only after independent annotation is completed.')
response('Reviewer 2', 'Testing on cross-sites generalisation should be conducted.', 'Agreed. In addition to the stratified random split, we will conduct leave-one-site-out evaluation where site identifiers are available, reporting the mean and variation across held-out sites. This will test whether the lexical baseline transfers beyond site-specific wording.')
response('Reviewer 2', 'Using balanced dataset does not reflect reality; this should be mentioned.', 'Agreed. The limitations section will explicitly state that the 50/50 binary balance is useful for controlled benchmarking but does not estimate prevalence in the wild. We will add prevalence-sensitive metrics (precision-recall curves, average precision, calibration, and thresholds) and evaluate a naturally sampled Indian collection when available.')
response('Reviewer 2', 'Per category results should be provided.', 'We will report per-category support and macro metrics only for categories with adequate test support, and mark sparse categories as exploratory rather than draw unreliable conclusions. The current paper already documents the severe imbalance (e.g., Forced Action has four samples); this constraint will be carried into the results table.')
response('Reviewer 2', 'Ethical and legal issues should be discussed.', 'We will add an ethics and legal section covering public-page collection, platform terms and rate limiting, minimization of personal data, redaction of screenshots, secure storage, annotator well-being and instructions, bias across sites/languages, false-positive harm, and the non-legal-advice status of CCPA mappings.')

response('Reviewer 3', 'Proposed framework should be shown on a diagram.', 'Agreed. We will add a single architecture diagram with two visually distinct lanes: completed baseline (public text corpus -> TF-IDF + logistic regression -> binary output/error analysis) and proposed system (Indian/foreign collection -> text and screenshot encoders -> fusion -> detection -> CCPA evidence mapping).')
response('Reviewer 3', 'Annotation agreement should be provided.', 'Agreed. For the India-focused collection, two independent annotators will label a pilot and overlapping production subset using a written codebook. We will report Cohen’s kappa (or Krippendorff’s alpha if appropriate), raw agreement, disagreement resolution, and agreement by category. No agreement statistic will be claimed for public labels that we did not create.')
response('Reviewer 3', 'Release of the labeled dark pattern dataset would be helpful for community.', 'We intend to release a documentation-rich dataset where licensing, platform terms, and privacy review permit: de-identified text, permitted screenshots or derived annotations, site/source metadata, label definitions, CCPA crosswalk, splits, codebook, and data statement. Restricted materials will be excluded or distributed as reproducible collection instructions rather than republished.')
response('Reviewer 3', 'Recent literature can be expanded.', 'Agreed. We will expand related work with recent dark-pattern, UI-understanding, domain-generalization, and regulatory/consumer-protection studies, while ensuring that every comparison identifies modality, data domain, task definition, and evaluation protocol.')
response('Reviewer 3', 'Conclusion section should be provided.', 'The current manuscript includes a conclusion. We will strengthen it by restating only validated findings, the scope limitation of the balanced public text corpus, and the exact evidence required before advancing the multimodal and CCPA-mapping claims.')

story += [P('Commitment to evidence-based revision', 'H1x'), P('The revision will be staged. Textual clarifications, framework diagram, CCPA crosswalk, ethics discussion, and a more complete reproducibility description can be added immediately. Claims requiring new data or experiments - transformer comparison, cross-site testing, per-category analysis, annotation agreement, and multimodal performance - will be added only with the corresponding methods, results, and artifacts. This preserves the paper’s transparency while giving the next version a credible and testable path forward.')]
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT.resolve())
