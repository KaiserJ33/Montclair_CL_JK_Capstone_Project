import argparse,base64,itertools,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def holm(values):
 p=np.asarray(values);order=np.argsort(p);adjusted=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))));result=np.empty(len(p));result[order]=adjusted;return result

def wilson(k,n):
 z=stats.norm.ppf(.975);p=k/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return center-half,center+half

parser=argparse.ArgumentParser();parser.add_argument('--reference-dir',default='reference_outputs');parser.add_argument('--output-dir',default='reproduced_outputs');args=parser.parse_args();root=Path(__file__).resolve().parent;ref=Path(args.reference_dir);ref=ref if ref.is_absolute() else root/ref;out=Path(args.output_dir);out=out if out.is_absolute() else root/out;out.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(ref/'ner_class_detection_summary.csv');platform=pd.read_csv(ref/'ner_class_b_platform_detection_summary.csv')
for frame in [d,platform]:
 frame['entity_rate']=frame.usernames_with_entity/frame.n_usernames
 intervals=[wilson(int(row.usernames_with_entity),int(row.n_usernames)) for row in frame.itertuples()];frame['ci95_low']=[v[0] for v in intervals];frame['ci95_high']=[v[1] for v in intervals]
d.to_csv(out/'ner_class_detection_summary.csv',index=False);platform.to_csv(out/'ner_class_b_platform_detection_summary.csv',index=False)
tests=[]
for variant in ['raw','normalized']:
 g=d[d.variant==variant].set_index('class')
 for a,b in itertools.combinations(['A','B','C'],2):
  first=g.loc[a];second=g.loc[b];n1=int(first.n_usernames);n2=int(second.n_usernames);k1=int(first.usernames_with_entity);k2=int(second.usernames_with_entity);test=stats.fisher_exact([[k1,n1-k1],[k2,n2-k2]]);p1=k1/n1;p2=k2/n2
  tests.append(dict(variant=variant,class_1=a,class_2=b,n_1=n1,n_2=n2,rate_1=p1,rate_2=p2,rate_difference_1_minus_2=p1-p2,odds_ratio=test.statistic,cohen_h=2*(np.arcsin(np.sqrt(p1))-np.arcsin(np.sqrt(p2))),p_value_fisher=test.pvalue))
t=pd.DataFrame(tests);t['p_value_holm']=holm(t.p_value_fisher);t.to_csv(out/'ner_pairwise_fisher_tests.csv',index=False)
m=pd.read_csv(ref/'ner_raw_vs_normalized_mcnemar_tests.csv');m['statistic']=m[['raw_no_normalized_yes','raw_yes_normalized_no']].min(axis=1);m['p_value']=[stats.binomtest(int(row.raw_no_normalized_yes),int(row.raw_no_normalized_yes+row.raw_yes_normalized_no),p=.5).pvalue if row.raw_no_normalized_yes+row.raw_yes_normalized_no else 1 for row in m.itertuples()];m['p_value_holm']=holm(m.p_value);m.to_csv(out/'ner_raw_vs_normalized_mcnemar_tests.csv',index=False)
for name in ['ner_class_label_summary.csv','ner_class_b_platform_label_summary.csv','input_audit.csv']:(out/name).write_bytes((ref/name).read_bytes())
for name,new,pcol in [('ner_pairwise_fisher_tests.csv',t,'p_value_holm'),('ner_raw_vs_normalized_mcnemar_tests.csv',m,'p_value_holm')]:
 old=pd.read_csv(ref/name);assert np.allclose(old[pcol],new[pcol],rtol=1e-10,atol=1e-14)
plt.rcParams.update({'font.size':11})
fig,ax=plt.subplots(figsize=(9,4.5));positions=np.arange(3)
for j,variant in enumerate(['raw','normalized']):
 g=d[d.variant==variant].set_index('class').loc[['A','B','C']];x=positions+(j-.5)*.35;y=g.entity_rate.to_numpy()*100;ax.bar(x,y,width=.35,label=variant.capitalize());ax.errorbar(x,y,yerr=np.stack([(g.entity_rate-g.ci95_low)*100,(g.ci95_high-g.entity_rate)*100]),fmt='none',ecolor='black',capsize=3)
ax.set(xticks=positions,xticklabels=['Class A','Class B','Class C'],ylabel='Usernames with a detected entity (%)',title='NER detection rates: raw and normalized input');ax.legend();fig.tight_layout();fig.savefig(out/'ner_detection_rates.png',dpi=160);plt.close(fig)
labels=pd.read_csv(ref/'ner_class_label_summary.csv');g=labels[labels.variant=='normalized'].pivot(index='entity_label',columns='class',values='pct_of_usernames').fillna(0).reindex(columns=['A','B','C']);fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(g.values,aspect='auto',cmap='Blues');ax.set(xticks=np.arange(3),xticklabels=['Class A','Class B','Class C'],yticks=np.arange(len(g)),yticklabels=g.index,title='Normalized NER labels');fig.colorbar(im,ax=ax,label='Usernames with label (%)');fig.tight_layout();fig.savefig(out/'ner_label_heatmap.png',dpi=160);plt.close(fig)
def table(frame):return frame.to_html(index=False,float_format=lambda v:f'{v:.5g}',border=0)
def image(name):return '<img style="width:100%;max-width:1000px" src="data:image/png;base64,'+base64.b64encode((out/name).read_bytes()).decode()+'">'
disclaimer='Not all usernames, account links, or source details could be independently verified. NER labels are model predictions, not verified identities or facts about account holders. No completed human-labelled reference set was available to measure recognition accuracy.'
report='<html><head><meta charset="utf-8"><title>Username NER results</title><style>body{font:16px/1.5 Arial;max-width:1100px;margin:40px auto;padding:0 24px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf3f8}.note{background:#fff2d8;padding:16px}</style></head><body><h1>Named entity recognition in usernames</h1><p class="note">'+disclaimer+'</p>'
report+='<h2>What was measured</h2><p>The saved analysis used spaCy 3.8.14 with the English transformer model en_core_web_trf. It examined 234 distinct Class A strings, 19,957 distinct platform–handle records from Class B, and a sample of 19,957 Class C records. Class C was sampled from 25,865,739 Reddit records using seed 42. The saved description specifies random-priority sampling. The exact sample and original model-running script were not available in the recovered files.</p><p>Each username was analysed as supplied and after light normalization. The saved settings describe removal of a leading @, splitting of CamelCase, delimiters, and letter–digit boundaries, and punctuation cleanup. No title-casing was added. The configured analytic labels are listed in ner_configuration.json. Detection rates count usernames with at least one of those labels.</p><h2>Main results</h2><p>After normalization, detection rates were about 54.70% for Class A, 74.65% for Class B, and 54.16% for Class C. Class B had more detections than A or C. The A–C difference was small and was not statistically significant after correction. Detection is not the same as correct recognition.</p>'+image('ner_detection_rates.png')+table(d)
report+='<h2>Class B platforms</h2><p>X and YouTube have separate results. Normalized detection was about 80.75% on X and 68.55% on YouTube. This describes model behaviour on these samples; it does not establish why the platforms differ.</p>'+table(platform)
report+='<h2>Entity labels</h2><p>The graph shows the percentage of usernames with each label after normalization. A username can receive more than one label, so percentages need not add to 100. A PERSON prediction does not prove that a handle contains its owner’s name. Other labels can also reflect mistaken readings of short strings.</p>'+image('ner_label_heatmap.png')+table(labels)
report+='<h2>Between-class tests</h2><p>Fisher exact tests compare detection rates. Cohen’s h describes the size and direction of a rate difference. Holm correction covers all six class comparisons across raw and normalized input. Differences can reflect platform conventions, collection methods, and model behaviour.</p>'+table(t)
report+='<h2>Raw versus normalized input</h2><p>The paired McNemar test checks whether normalization changes detection within the same usernames. The two disagreement columns count cases gaining or losing a detection. This package uses the exact binomial form of the test and applies Holm correction across the three classes. A change in detections is not proof that normalization improved accuracy.</p>'+table(m)
report+='<h2>Reading the columns</h2><p>n_usernames is the number of records analysed. usernames_with_entity counts those with at least one analytic label. entity_rate is that count divided by n_usernames. Multiply rates by 100 for percentages. ci95_low and ci95_high are the ends of a Wilson 95% confidence interval. mean_analytic_entities is the average number of detected analytic entities per record. Small intervals show precision under the statistical assumptions, not independent source verification.</p><h2>What can be reproduced</h2><p>The supplied code rebuilds detection rates, confidence intervals, Fisher tests, McNemar tests, and graphs from the saved aggregate counts. These calculations match the saved corrected p-values. Entity-label tables and average entity counts are retained from the earlier outputs. This is not a full rerun of spaCy on the original samples. That would require the original inference code, the exact sampled usernames, and matching model assets. No model accuracy, precision, recall, or F1 is claimed here.</p><h2>Rerunning the tables and report</h2><p>Run python reproduce_ner_statistics.py from this package. The original aggregate tables are in reference_outputs; rebuilt outputs are in reproduced_outputs. No raw usernames or private review sheets are included.</p></body></html>'
(root/'NER_Report.html').write_text(report);(root/'Verification_Disclaimer.txt').write_text(disclaimer+'\n');print('NER statistics reproduced and checked against saved tests. Two graphs and report created.')
