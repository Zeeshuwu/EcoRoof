"""FASE 0d - brute-force search for the origin of '92.08 % Precision / 80.61 % IoU'.
Every subset of site x class rows of every source, mean and pooled, all five metrics.
Output printed and written to results/phase0_trace_9208.txt"""
import pandas as pd, itertools, numpy as np
import os
from common import metrics_from_counts, PUBLISHED, RESULTS
ps=pd.read_csv(os.path.join(RESULTS,'phase0_all_sources_per_site.csv'))
hits=[]
M=['IoU','Precision','Recall','F1','Accuracy']
for src,d in ps.groupby('source'):
    rows=d.to_dict('records')
    for r in range(1,len(rows)+1):
        for sub in itertools.combinations(rows,r):
            sd=pd.DataFrame(sub)
            pooled=metrics_from_counts(sd.TP.sum(),sd.FP.sum(),sd.FN.sum(),sd.TN.sum())
            for k in M:
                for agg,val in (('mean',sd[k].mean()),('pooled',100*pooled[k])):
                    for tgt in (92.08,80.61):
                        if abs(val-tgt)<0.03: hits.append((tgt,src,k,agg,round(val,3),';'.join(sd.site+'/'+sd.cls)))
for h in sorted(hits): print(h)
# from published numbers
P=PUBLISHED
vals={f'{c}.{k}':v for c in P for k,v in P[c].items()}
for (a,va),(b,vb) in itertools.combinations(vals.items(),2):
    for w in (0.5,):
        m=(va+vb)/2
        if abs(m-92.08)<0.05 or abs(m-80.61)<0.05: print('published mean',a,b,m)
print('F1 from P_roof & P_pv? harmonic', 2*96.6*87.06/(96.6+87.06))
