"""Per-site PAPER_CONFIG metrics in every MC test fold (checks folds without GT).
Output: results/mc_per_site_paperconfig.csv"""
import numpy as np, pandas as pd
from montecarlo import load, mc_split, GRID_IDS, PAPER_ID, BASE_SEED, N_MC
import os
from common import vec_metrics, RESULTS
_,_,data=load()
pid=GRID_IDS.index(PAPER_ID)
for cls in ('pv','roof'):
    sites=data[cls]; nogt=np.zeros(len(sites),int); ious=[]
    for it in range(N_MC):
        sp=mc_split([len(s['blocks']) for s in sites],BASE_SEED+it,cls)
        row=[]
        for k,(s,(ca,te)) in enumerate(zip(sites,sp)):
            c=s['C'][pid][te].sum(0)
            if c[0]+c[2]==0: nogt[k]+=1
            row.append(vec_metrics(c)['IoU']*100)
        ious.append(row)
    ious=np.array(ious)
    print(cls,[s['site'] for s in sites],'iters with no GT in test:',nogt,'per-site IoU mean',ious.mean(0).round(1),'sd',ious.std(0,ddof=1).round(1))
rows=[]
for cls in ('pv','roof'):
    sites=data[cls]
    for it in range(N_MC):
        sp=mc_split([len(s['blocks']) for s in sites],BASE_SEED+it,cls)
        for s,(ca,te) in zip(sites,sp):
            c=s['C'][pid][te].sum(0); m=vec_metrics(c)
            rows.append(dict(cls=cls,iteration=it,site=s['site'],n_test_blocks=len(te),gt_px_test=int(c[0]+c[2]),IoU_paperconfig=float(m['IoU'])*100,F1_paperconfig=float(m['F1'])*100))
pd.DataFrame(rows).to_csv(os.path.join(RESULTS,'mc_per_site_paperconfig.csv'),index=False)
