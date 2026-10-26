# -*- coding: utf-8 -*-
"""Traditional RGB regression models trained on ONE input CSV; no row filtering."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cross_decomposition import PLSRegression
from rgb_common import FEATURES, LABELS, TARGETS, load_dataset, predicted_rows, score, metrics_table, plot_predictions

SEED = 42


def model_zoo():
    return {
        'KNN': KNeighborsRegressor(n_neighbors=5),
        'RandomForest': RandomForestRegressor(n_estimators=500,random_state=SEED,n_jobs=-1),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=500,random_state=SEED,n_jobs=-1),
        'DecisionTree': DecisionTreeRegressor(min_samples_leaf=2,random_state=SEED),
        'GradientBoosting': MultiOutputRegressor(GradientBoostingRegressor(
             n_estimators=150,learning_rate=.05,max_depth=2,random_state=SEED)),
        'SVR_RBF': MultiOutputRegressor(SVR(kernel='rbf',C=10,epsilon=.05)),
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=.01,max_iter=20000),
        'PLS': PLSRegression(n_components=2,scale=False),
    }


def evaluate(args):
    root=Path(__file__).resolve().parent
    data=Path(args.data)
    if not data.is_absolute() and not data.is_file(): data=root/data
    out=Path(args.out) if args.out else root/f'results_traditional_{args.cv}'
    out.mkdir(parents=True,exist_ok=True)
    df=load_dataset(data)
    groups=None
    if args.cv=='loao':
        if args.group_column not in df.columns:
            raise ValueError(f'For loao, CSV must contain group column: {args.group_column}')
        if df[args.group_column].isna().any(): raise ValueError('Group column contains missing values')
        groups=df[args.group_column].astype(str).to_numpy()
        if len(set(groups))<3:raise ValueError('Need >=3 unique groups for loao')
        splits=list(LeaveOneGroupOut().split(df,groups=groups))
    else:
        if args.folds<2 or args.folds>len(df):raise ValueError('Invalid number of folds')
        splits=list(KFold(n_splits=args.folds,shuffle=True,random_state=SEED).split(df))
    print(f'Dataset: {data}; records: {len(df)}; CV: {args.cv}; folds: {len(splits)}')
    df.to_csv(out/'input_records.csv',index=False,encoding='utf-8-sig')
    X=df[FEATURES].to_numpy(float)
    Y=df[LABELS].to_numpy(float)
    zoo=model_zoo()
    invalid=set(args.models)-set(zoo)
    if invalid:raise ValueError(f'Unknown models: {sorted(invalid)}')
    fold_rows, oof_parts, train_parts = [], [], []
    for name in args.models:
        print(f'\n{name}',flush=True)
        for fold,(itr,ite) in enumerate(splits,1):
            estimator=TransformedTargetRegressor(
                regressor=Pipeline([('scaler_X',StandardScaler()),('regressor',clone(zoo[name]))]),
                transformer=StandardScaler())
            estimator.fit(X[itr],Y[itr])
            for role,idx in [('train',itr),('test',ite)]:
                pred=np.asarray(estimator.predict(X[idx]),float).reshape(-1,2)
                block=predicted_rows(df.iloc[idx],pred,fold,role,name)
                if role=='train':train_parts.append(block)
                else:oof_parts.append(block)
                for _,r in metrics_table(block).iterrows():
                    fold_rows.append({'model':name,'fold':fold,'role':role,**r.to_dict()})
            print(f'  fold {fold}/{len(splits)} complete',flush=True)
    oof=pd.concat(oof_parts,ignore_index=True).sort_values(['model','record_id'])
    trains=pd.concat(train_parts,ignore_index=True)
    fm=pd.DataFrame(fold_rows)
    rows=[]
    for name in args.models:
        sample=oof.loc[oof.model==name]
        if len(sample)!=len(df) or sample.record_id.duplicated().any():
            raise RuntimeError(f'OOF coverage is not one prediction per input row: {name}')
        for _,m in metrics_table(sample).iterrows():
            target=m['target']
            tr=fm[(fm.model==name)&(fm.role=='train')&(fm.target==target)]
            te=fm[(fm.model==name)&(fm.role=='test')&(fm.target==target)]
            rows.append({'model':name,'dataset':'all_data','target':target,
                         'Test_n_unique_records':int(m['n']),
                         'Train_R2_fold_mean':tr.R2.mean(),
                         'Test_R2_fold_mean':te.R2.mean(),
                         'Test_R2_OOF_pooled':m.R2,'Test_MAE_OOF':m.MAE,
                         'Test_RMSE_OOF':m.RMSE,'Test_Bias_OOF':m.Bias})
    summary=pd.DataFrame(rows)
    summary.to_csv(out/'SUMMARY_Train_vs_Test_R2.csv',index=False,encoding='utf-8-sig')
    fm.to_csv(out/'fold_train_test_metrics.csv',index=False,encoding='utf-8-sig')
    oof.to_csv(out/'test_OOF_predictions_all_models.csv',index=False,encoding='utf-8-sig')
    trains.to_csv(out/'fold_training_predictions_all_models.csv',index=False,encoding='utf-8-sig')
    print('\n',summary[['model','target','Test_R2_OOF_pooled','Test_MAE_OOF']].to_string(index=False))
    for name in args.models:
        model_dir=out/'plots'/name
        plot_predictions(oof.loc[oof.model==name],model_dir,args.pdf)
        final=TransformedTargetRegressor(
            regressor=Pipeline([('scaler_X',StandardScaler()),('regressor',clone(zoo[name]))]),
            transformer=StandardScaler())
        final.fit(X,Y)
        joblib.dump({'estimator':final,'features':FEATURES,'targets':LABELS,'model':name},out/f'final_{name}.joblib')
    for target in TARGETS:
        sub=summary.loc[summary.target==target].sort_values('Test_R2_OOF_pooled',ascending=False)
        fig,ax=plt.subplots(figsize=(9,4.5))
        ax.bar(sub.model,sub.Test_R2_OOF_pooled)
        ax.set_title(f'Model comparison: {target}');ax.set_ylabel('OOF R2')
        ax.tick_params(axis='x',rotation=38);ax.axhline(0,color='gray',linewidth=.8)
        fig.tight_layout()
        dest=out/'plots'/f'comparison_{target}'
        fig.savefig(str(dest)+'.png',dpi=150)
        if args.pdf:fig.savefig(str(dest)+'.pdf')
        plt.close(fig)
    (out/'run_config.json').write_text(json.dumps({'dataset':str(data),'records':len(df),'cv':args.cv,
        'folds':len(splits),'models':args.models,'features':FEATURES,'targets':LABELS,'seed':SEED},
        indent=2,ensure_ascii=False),encoding='utf-8')
    print('Output:',out.resolve())

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Compare classical ML models using a single input CSV')
    p.add_argument('--data',default='data.csv',help='One input CSV for every model')
    p.add_argument('--cv',choices=['kfold','loao'],default='kfold')
    p.add_argument('--folds',type=int,default=5)
    p.add_argument('--group-column',default='图片名称')
    p.add_argument('--out',default=None)
    p.add_argument('--pdf',action='store_true')
    p.add_argument('--models',nargs='+',default=list(model_zoo()))
    evaluate(p.parse_args())
