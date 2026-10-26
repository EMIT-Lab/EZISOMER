# -*- coding: utf-8 -*-
"""Permutation importance for the final ANN trained on a single CSV."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rgb_common import FEATURES, LABELS, TARGETS, load_dataset, de_from_concentrations, score

class RGBAnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(
            nn.Linear(6,32),nn.ReLU(),nn.Dropout(.10),
            nn.Linear(32,16),nn.ReLU(),nn.Dropout(.10),
            nn.Linear(16,8),nn.ReLU(),nn.Linear(8,2))
    def forward(self,x):return self.network(x)

def main(args):
    root=Path(__file__).resolve().parent
    dat=Path(args.data)
    if not dat.is_absolute() and not dat.is_file():dat=root/dat
    results=Path(args.results)
    if not results.exists() and not results.is_absolute():results=root/results
    df=load_dataset(dat)
    try:
        ck=torch.load(results/'final_ann_model.pth',map_location='cpu',weights_only=True)
    except TypeError:
        ck=torch.load(results/'final_ann_model.pth',map_location='cpu')
    if ck.get('feature_columns')!=FEATURES:raise ValueError('Model and CSV feature order do not match')
    model=RGBAnn();model.load_state_dict(ck['model_state_dict']);model.eval()
    sx=joblib.load(results/'final_scaler_X.pkl')
    sy=joblib.load(results/'final_scaler_y.pkl')
    X=df[FEATURES].to_numpy(float)
    Y=df[LABELS].to_numpy(float)
    true={'E_concentration':Y[:,0],'Z_concentration':Y[:,1],
          'total_concentration':Y.sum(axis=1),
          'de_from_concentration':de_from_concentrations(Y[:,0],Y[:,1])}
    def predict(x):
        with torch.no_grad():
            y=sy.inverse_transform(model(torch.as_tensor(sx.transform(x),dtype=torch.float32)).numpy())
        return {'E_concentration':y[:,0],'Z_concentration':y[:,1],
                'total_concentration':y.sum(axis=1),
                'de_from_concentration':de_from_concentrations(y[:,0],y[:,1])}
    base=predict(X)
    rng=np.random.default_rng(args.seed)
    rows=[]
    for name in FEATURES:
        j=FEATURES.index(name)
        for iteration in range(1,args.repeats+1):
            perm=X.copy()
            perm[:,j]=rng.permutation(perm[:,j])
            altered=predict(perm)
            for target in true:
                initial=score(true[target],base[target])['MAE']
                shuffled=score(true[target],altered[target])['MAE']
                rows.append({'feature':name,'repeat':iteration,'target':target,
                             'baseline_MAE':initial,'permuted_MAE':shuffled,
                             'importance_delta_MAE':shuffled-initial})
    raw=pd.DataFrame(rows)
    out=Path(args.out) if args.out else results/'feature_importance'
    out.mkdir(parents=True,exist_ok=True)
    raw.to_csv(out/'permutation_repeats.csv',index=False,encoding='utf-8-sig')
    summary=raw.groupby(['target','feature'],as_index=False).agg(
        importance_mean=('importance_delta_MAE','mean'),
        importance_sd=('importance_delta_MAE','std'),
        baseline_MAE=('baseline_MAE','mean'))
    # Relative shares are illustrative and not additive causal contributions.
    summary['relative_share_pct']=np.nan
    for target,ids in summary.groupby('target').groups.items():
        positive=summary.loc[ids,'importance_mean'].clip(lower=0)
        total=float(positive.sum())
        if total>0:summary.loc[ids,'relative_share_pct']=100*positive/total
    summary.to_csv(out/'permutation_importance_summary.csv',index=False,encoding='utf-8-sig')
    for target in true:
        block=summary[summary.target==target].sort_values('importance_mean',ascending=False)
        fig,ax=plt.subplots(figsize=(7,4.5))
        ax.bar(block.feature,block.importance_mean,yerr=block.importance_sd,capsize=3)
        ax.axhline(0,color='gray',linewidth=.8)
        ax.set(ylabel='Change in MAE (mean ± SD)',title=f'ANN permutation importance | {target}')
        fig.tight_layout()
        dest=out/f'importance_{target}'
        fig.savefig(str(dest)+'.png',dpi=150)
        if args.pdf:fig.savefig(str(dest)+'.pdf')
        plt.close(fig)
    (out/'method.json').write_text(json.dumps({
        'model':'final trained ANN','data':str(dat),'repeats':args.repeats,
        'metric':'permuted MAE - baseline MAE','evaluation':'in-sample final-model importance'},
        indent=2,ensure_ascii=False),encoding='utf-8')
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('Output:',out.resolve())

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Permutation importance using final ANN and one CSV')
    p.add_argument('--data',default='data.csv')
    p.add_argument('--results',default='results_ann_loao')
    p.add_argument('--out',default=None)
    p.add_argument('--repeats',type=int,default=50)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--pdf',action='store_true')
    main(p.parse_args())
