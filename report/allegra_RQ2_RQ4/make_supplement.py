"""Render the contribution supplement from the measured result tables."""
from pathlib import Path
from xml.sax.saxutils import escape
import json
import pandas as pd
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
ROOT=Path(__file__).resolve().parents[2]
FONT='Helvetica'
if Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf').exists():
    pdfmetrics.registerFont(TTFont('DejaVu','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    pdfmetrics.registerFont(TTFont('DejaVu-Bold','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
    pdfmetrics.registerFontFamily('DejaVu',normal='DejaVu',bold='DejaVu-Bold',italic='DejaVu',boldItalic='DejaVu-Bold');FONT='DejaVu'
scores=pd.read_csv(ROOT/'rq2/results/seed42/summary.csv')
gains=pd.read_csv(ROOT/'rq2/results/seed42/prompt_benefit.csv')
ablations=pd.read_csv(ROOT/'rq4/results/seed42/ablation_summary.csv')
features=pd.read_csv(ROOT/'rq4/results/seed42/feature_summary.csv',header=[0,1],index_col=0)
styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='Body2',fontName=FONT,fontSize=9.2,leading=12,spaceAfter=7));styles.add(ParagraphStyle(name='Small2',fontName=FONT,fontSize=8,leading=10))
for name in ['Title','Heading1','Heading2']:styles[name].fontName=FONT+'-Bold';styles[name].textColor=colors.HexColor('#183f68')
flow=[]
def text(s):flow.append(Paragraph(s,styles['Body2']))
def heading(s):flow.append(Paragraph(s,styles['Heading1']))
def table(rows,widths):
    t=Table([[Paragraph(escape(str(x)),styles['Small2']) for x in row] for row in rows],colWidths=widths,repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4edf6')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7fa')])]))
    flow.extend([t,Spacer(1,12)])
def pct(x):return f'{100*x:.1f}%'
flow.append(Paragraph('LLM Fingerprinting: RQ2 and RQ4',styles['Title']))
text('CMPSC 448 | Allegra Francis contribution supplement | October 11, 2026')
text('Uses Amal\'s current dataset and shared model/training implementation. Combine with the team\'s RQ1/RQ3 work and final leader/member identification. This supplement is not the complete team submission.')
heading('Exact shared dataset')
text('The existing data/processed/raid_balanced.csv contains 1,600 responses from chatgpt, llama-chat, mistral-chat and cohere-chat, representing four distinct families. Domains are news, poetry, recipes and reviews. Each domain has 100 matched original generation prompts, each answered by all four models. The team builder uses RAID-train, attack=none, sampling decoding and no repetition penalty. We reused the CSV unchanged rather than collecting or resampling data.')
table([['Domain','Train prompts','Validation','Test prompts','Responses'],*[[d,80,10,10,400] for d in ['news','poetry','recipes','reviews']]],[100,96,96,96,100])
text('The saved 80/10/10 source split gives 1,280 training, 160 validation and 160 test responses. Matched sources and their identical prompts stay in one split. The file hash matches the partner manifest: f95ce226d32562d0f8b7e18946931b20be61ecdeab47d13af32a15c580685ca4. No dataset, source builder, common model code or existing RQ1/RQ3 files were changed.')
heading('Reused CNN/LSTM and controlled RQ2 protocol')
text('Import common.train.TrainConfig and train_experiment. Reuse TextCNN with 128-dimensional embeddings, 128 channels per window 3/4/5 and masked max pooling. Reuse the one-layer 128-unit TextLSTM with masked mean pooling. Both use dropout 0.3 and a four-class linear head. AdamW lr 0.001, weight_decay 0.0001, batch 32, clipping 1.0, seed 42, up to 15 epochs and patience 4. Select maximum validation macro-F1, breaking ties with lower validation loss; then evaluate test.')
text('On copies of the rows, preserve the first 128 prompt tokens and first 256 response tokens using the shared tokenizer. Output-only and combined conditions see exactly the same response tokens. Combined adds the prompt and shared newline separator, with maximum 385 tokens. Fit vocabularies separately on visible training text only (max 20,000, min frequency 2). Metadata and labels are excluded from input.')
text('The original RQ1 fits its vocabulary on full training outputs before encoding truncation. Our output-only baseline refits vocabulary on visible output windows for the controlled comparison. Therefore, RQ2 scores are independently measured, not copied from RQ1. Reuse refers to code/architecture, not previously trained weights. Runs here use CPU; numerical results can differ across devices.')
flow.append(PageBreak())
heading('RQ2: Does the prompt help overall?')
table([['Model','Input condition','Accuracy','Macro-F1'],*[[r.model.upper(),r.input_mode,pct(r.accuracy),f'{r.macro_f1:.3f}'] for r in scores.itertuples()]],[85,170,115,118])
for model in ['cnn','lstm']:
    r=gains[(gains.model==model)&(gains.domain=='overall')].iloc[0]
    text(f'{model.upper()}: adding the prompt changes accuracy from {pct(r.output_accuracy)} to {pct(r.both_accuracy)} ({100*r.accuracy_gain:+.1f} percentage points), and macro-F1 from {r.output_macro_f1:.3f} to {r.both_macro_f1:.3f}. The paired accuracy-gain interval is [{100*r.accuracy_gain_ci_low:+.1f}, {100*r.accuracy_gain_ci_high:+.1f}] percentage points.')
text('There is no clear overall prompt benefit in this single-seed run: both point estimates decrease, and both overall accuracy-gain intervals include zero. This does not establish that prompts can never help.')
text('Input-only accuracy is exactly 25% because every prompt is paired with all four labels. This is a controlled-design property, not failed training. Macro-F1 need not be 25%. This dataset does not measure users\' deliberate platform choices.')
heading('Does prompt benefit differ by domain?')
rows=[['Domain','CNN accuracy gain [95% CI], pp','LSTM accuracy gain [95% CI], pp']]
for domain in ['news','poetry','recipes','reviews']:
    row=[domain]
    for model in ['cnn','lstm']:
        r=gains[(gains.model==model)&(gains.domain==domain)].iloc[0]
        row.append(f'{100*r.accuracy_gain:+.1f} [{100*r.accuracy_gain_ci_low:+.1f}, {100*r.accuracy_gain_ci_high:+.1f}]')
    rows.append(row)
table(rows,[90,199,199])
text('Poetry improves in both point estimates; CNN recipes also improve. CNN news falls by 22.5 percentage points, with a negative paired interval. Other changes are uncertain. These are exploratory within-run observations, not established universal domain effects.')
flow.append(Image(str(ROOT/'rq2/results/seed42/prompt_benefit_by_domain.png'),width=320,height=160))
text('Gain means input+output minus output-only. Intervals use 2,000 paired bootstrap resamples of entire matched source groups, seed 42. There are 40 test responses but only 10 independent prompts per domain. Intervals are conditional on fixed trained models and exclude training-seed variability. We did not formally test differences between domain gains or adjust multiple comparisons. Domain slices are not RQ3 domain-held-out transfer.')
flow.append(PageBreak())
heading('RQ4: Linguistic and structural differences')
text('Descriptive analysis uses full original training responses only, covering length, vocabulary diversity, punctuation, newlines, bullets, headings, code fences and uppercase fraction. Sentence splitting is heuristic. Frequent-word tables are descriptive frequency counts, not neural feature attribution.')
rows=[['Model','Mean words','Raw TTR','First100 TTR','Newlines /100 words']]
for model in features.index:
    r=features.loc[model];rows.append([model,f'{r[("words","mean")]:.1f}',f'{r[("type_token_ratio","mean")]:.3f}',f'{r[("first100_type_token_ratio","mean")]:.3f}',f'{r[("newlines_per100_words","mean")]:.2f}'])
table(rows,[100,85,80,100,123])
text('Raw type-token ratio depends on length. First100 fixes its word budget but excludes shorter responses; usable counts appear in feature_summary.csv. Domain-conditioned summaries help inspect task effects. These full-output characteristics need not match the classifier\'s truncated window.')
heading('Retraining ablations with shared models')
rows=[['Model','Condition','Accuracy','Macro-F1','F1 drop']]
for r in ablations.itertuples():rows.append([r.model.upper(),r.condition,pct(r.accuracy),f'{r.macro_f1:.3f}',f'{r.macro_f1_drop_from_baseline:+.3f}'])
table(rows,[60,175,85,85,83])
text('Shuffling lowers CNN accuracy from 73.1% to 51.9%, but increases LSTM accuracy from 53.8% to 60.6%. Removing formatting lowers CNN accuracy to 69.4% and increases LSTM accuracy to 55.6%. Ordered patterns appear useful for the CNN under this protocol; these results do not support order as essential for the retrained mean-pooling LSTM. Optimization and single-seed variability remain possible explanations.')
text('First select the baseline\'s visible 256 response tokens. Formatting removal preserves the visible word content and capitalization while removing punctuation/newlines; it does not introduce additional response content. Token shuffling preserves the visible token multiset, with a deterministic per-response seed. Each condition refits training vocabulary and retrains both models using the original split and shared hyperparameters. Negative drops mean improvement.')
text('A performance decrease suggests the altered signals were useful under this training protocol, but these are combined interventions: shuffling changes syntax, local phrases and line layout; formatting removal changes token counts. Descriptive differences do not establish classifier reliance. Single-seed ablation differences are exploratory rather than causal or statistically established.')
heading('Verification and scope')
text('All 21 existing and added tests passed. Added checks verify token-budget equivalence, metadata/split preservation and ablation token/word preservation. The setup checker validates the unchanged partner dataset/shared code against the completed run. Performance remains limited to one representative model per family, four domains, fixed generation settings, truncated text and one training seed.')
flow.append(PageBreak())
heading('Reproduce or add the completed contribution')
text('The delivery ZIP contains only rq2, rq4, tests/test_rq2_rq4.py, report/allegra_RQ2_RQ4 and START_HERE_ALLEGRA.md. It does not replace data, src/common, scripts, rq1, rq3, root README or requirements. Completed result tables/plots are included, so rerunning is optional. Checkpoints are excluded from the ZIP and ignored by the team\'s .gitignore.')
text('From the cloned repository root:<br/><font name="Courier">python3 rq2/check_setup.py</font><br/>This uses only standard Python libraries to compare dataset and shared-code hashes. See START_HERE_ALLEGRA.md for exact cloning, branching, unzip, commit and pull-request commands.')
text('Optional reproduction after installing the repository requirements:<br/><font name="Courier" size="8">python rq2/run_rq2.py --device cpu --output-dir rq2/results/reproduction<br/>python rq4/run_rq4.py --device cpu --ablations --output-dir rq4/results/reproduction</font>')
text('Main files: rq2/results/seed42/summary.csv, prompt_benefit.csv, domain_scores.csv; rq4/results/seed42/feature_summary.csv, features_by_domain.csv, ablation_summary.csv. Per-run directories follow Amal\'s test_metrics.json, test_predictions.csv, config.json, vocabulary.json, history.json, confusion_matrix.png and learning_curves.png format.')
heading('Lessons supported by these experiments')
text('Matched sampling makes the input-only control interpretable. Source-group splitting prevents repeated-prompt leakage. Fixed component budgets avoid confounding prompt addition with lost or extra response evidence. Code reuse improves team consistency, but vocabulary protocol/device differences still require disclosure. Small domain samples yield uncertain estimates, and ablations can change several signals at once. Add or confirm personal lessons using the team\'s actual experience.')
heading('Final team report and disclosure')
text('The assignment requires RQ1 and RQ2, with RQ4 optional extra credit. Combine this contribution with the team\'s RQ1/RQ3 analysis, full member names, leader designation and verified lessons. The final PDF should cover dataset curation, CNN/RNN and training, results, in-depth analyses and lessons/experience. This supplement does not replace the final team PDF.')
text('OpenAI Codex and ChatGPT assisted with implementation, execution, testing and documentation. Team members must understand, verify and explain all submitted work.')
heading('Sources')
text('Team repository: https://github.com/amalsony/llm-fingerprinting<br/>Base commit: 2850ea1e9503695d19fc63780b996a40198356d4<br/>RAID dataset: https://huggingface.co/datasets/liamdugan/raid<br/>Dugan et al. (2024), RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors. https://aclanthology.org/2024.acl-long.674/<br/>Retain the team\'s dataset provenance and applicable license notices. No human reference documents or private chats were added.')
def footer(canvas,doc):
    canvas.setFont(FONT,8);canvas.setFillColor(colors.HexColor('#65768b'));canvas.drawString(54,28,'CMPSC448 | Partner-compatible RQ2/RQ4 supplement');canvas.drawRightString(558,28,str(doc.page))
out=Path(__file__).parent/'RQ2_RQ4_partner_supplement.pdf'
SimpleDocTemplate(str(out),pagesize=(612,792),leftMargin=54,rightMargin=54,topMargin=42,bottomMargin=45).build(flow,onFirstPage=footer,onLaterPages=footer)
print(out)
