# -*- coding: utf-8 -*-
"""ANN concentration regression on one CSV, with LOAO or random K-fold CV."""
from __future__ import annotations
import argparse, copy, json, random
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneGroupOut, KFold, GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from rgb_common import FEATURES, LABELS, TARGETS, load_dataset, predicted_rows, score, metrics_table, plot_predictions

SEED=42
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RGBAnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(
            nn.Linear(6,32),nn.ReLU(),nn.Dropout(0.10),
            nn.Linear(32,16),nn.ReLU(),nn.Dropout(0.10),
            nn.Linear(16,8),nn.ReLU(),nn.Linear(8,2))
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    def forward(self,x):
        return self.network(x)

def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def train_model(x_scaled,y_scaled,epochs,batch_size,lr,seed,val=None,patience=200):
    set_seed(seed)
    mdl=RGBAnn().to(DEVICE)
    optim=torch.optim.Adam(mdl.parameters(),lr=lr,weight_decay=1e-5)
    loader=DataLoader(TensorDataset(torch.as_tensor(x_scaled,dtype=torch.float32),
                                    torch.as_tensor(y_scaled,dtype=torch.float32)),
                      batch_size=min(batch_size,len(x_scaled)),shuffle=True,
                      generator=torch.Generator().manual_seed(seed))
    lossfn=nn.MSELoss()
    if val is not None:
        valx,valy=val
        valx=torch.as_tensor(valx,dtype=torch.float32,device=DEVICE)
        valy=torch.as_tensor(valy,dtype=torch.float32,device=DEVICE)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optim,mode='min',factor=.5,patience=min(50,max(3,patience//4)),min_lr=1e-6)
    bestloss=float('inf'); bestepoch=1; beststate=None;stale=0
    history=[]
    for ep in range(1,epochs+1):
        mdl.train(); batch_losses=[]
        for xb,yb in loader:
            xb=xb.to(DEVICE);yb=yb.to(DEVICE)
            optim.zero_grad(set_to_none=True)
            loss=lossfn(mdl(xb),yb)
            loss.backward();nn.utils.clip_grad_norm_(mdl.parameters(),max_norm=1.)
            optim.step();batch_losses.append(float(loss.item()))
        row={'epoch':ep,'train_loss_scaled':float(np.mean(batch_losses))}
        if val is not None:
            mdl.eval()
            with torch.no_grad():vloss=float(lossfn(mdl(valx),valy).item())
            scheduler.step(vloss)
            row['validation_loss_scaled']=vloss
            if vloss<bestloss-1e-10:
                bestloss=vloss;bestepoch=ep;beststate=copy.deepcopy(mdl.state_dict());stale=0
            else:stale+=1
        history.append(row)
        if ep==1 or ep%200==0:
            status=f' epoch {ep}: train={row["train_loss_scaled"]:.5f}'
            if val is not None:status+=f' val={row["validation_loss_scaled"]:.5f}'
            print(status,flush=True)
        if val is not None and stale>=patience:break
    if val is not None:
        mdl.load_state_dict(beststate)
        return mdl,bestepoch,history
    return mdl,epochs,history

def predict(mdl,sx,sy,x):
    mdl.eval()
    with torch.no_grad():
        yp=mdl(torch.as_tensor(sx.transform(x),dtype=torch.float32,device=DEVICE)).cpu().numpy()
    return sy.inverse_transform(yp)

def choose_epochs(X,Y,outer_tr,groups,fold,args):
    if args.cv=='loao':
        subgroups=groups[outer_tr]
        if len(set(subgroups))<2:raise ValueError('Not enough groups for an internal validation split')
        inner_tr,inner_v=next(GroupShuffleSplit(n_splits=1,test_size=.20,random_state=SEED+fold).split(
            X[outer_tr],groups=subgroups))
        itr,iv=outer_tr[inner_tr],outer_tr[inner_v]
    else:
        itr,iv=train_test_split(outer_tr,test_size=.2,random_state=SEED+fold,shuffle=True)
    sx=StandardScaler().fit(X[itr]);sy=StandardScaler().fit(Y[itr])
    _,best,history=train_model(sx.transform(X[itr]),sy.transform(Y[itr]),
        args.epochs,args.batch_size,args.lr,SEED+fold,
        val=(sx.transform(X[iv]),sy.transform(Y[iv])),patience=args.patience)
    return best,history

def evaluate(args):
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parent
    data=Path(args.data)
    if not data.is_absolute() and not data.is_file():data=root/data
    df=load_dataset(data)
    out=Path(args.out) if args.out else root/f'results_ann_{args.cv}'
    out.mkdir(parents=True,exist_ok=True)
    if args.quick:
        args.epochs=12;args.patience=4
        print('QUICK TEST ONLY: reported R2 is not a final scientific estimate')
    if args.cv=='loao':
        if args.group_column not in df.columns:
            raise ValueError(f'CSV must contain group column: {args.group_column}')
        if df[args.group_column].isna().any():raise ValueError('Group column contains missing values')
        groups=df[args.group_column].astype(str).to_numpy()
        if len(set(groups))<3:raise ValueError('Need at least 3 unique groups for LOAO')
        splits=list(LeaveOneGroupOut().split(df,groups=groups))
    else:
        groups=np.array(['']*len(df))
        if not 2<=args.folds<=len(df):raise ValueError('Invalid folds')
        splits=list(KFold(n_splits=args.folds,shuffle=True,random_state=SEED).split(df))
    df.to_csv(out/'input_records.csv',index=False,encoding='utf-8-sig')
    X=df[FEATURES].to_numpy(np.float32);Y=df[LABELS].to_numpy(np.float32)
    all_train,all_test,foldrows,foldinfo,histrows=[],[],[],[],[]
    print(f'One dataset: {data}; n={len(df)}; CV={args.cv}; folds={len(splits)}; device={DEVICE}',flush=True)
    for fold,(itr,ite) in enumerate(splits,1):
        held=str(groups[ite[0]]) if args.cv=='loao' else str(fold)
        if args.cv=='loao' and (set(groups[itr]) & set(groups[ite])):
            raise AssertionError('Group overlap between train and test')
        print(f'\nFOLD {fold}/{len(splits)} group={held} train={len(itr)} test={len(ite)}',flush=True)
        best,history=choose_epochs(X,Y,itr,groups,fold,args)
        sx=StandardScaler().fit(X[itr]);sy=StandardScaler().fit(Y[itr])
        mdl,_,_=train_model(sx.transform(X[itr]),sy.transform(Y[itr]),best,args.batch_size,args.lr,SEED+fold)
        a=predicted_rows(df.iloc[itr],predict(mdl,sx,sy,X[itr]),fold,'train')
        b=predicted_rows(df.iloc[ite],predict(mdl,sx,sy,X[ite]),fold,'test')
        if args.cv=='loao':
            a['held_out_group']=held;b['held_out_group']=held
        all_train.append(a);all_test.append(b)
        for h in history: histrows.append({'fold':fold,**h})
        foldinfo.append({'fold':fold,'held_out_group':held,'train_n':len(itr),'test_n':len(ite),'selected_epochs':best})
        for role,part in [('train',a),('test',b)]:
            for _,r in metrics_table(part).iterrows():foldrows.append({'fold':fold,'role':role,**r.to_dict()})
    oof=pd.concat(all_test,ignore_index=True).sort_values('record_id')
    train=pd.concat(all_train,ignore_index=True)
    if len(oof)!=len(df) or oof.record_id.duplicated().any():raise AssertionError('OOF is not one-to-one')
    fm=pd.DataFrame(foldrows)
    summary=[]
    for _,r in metrics_table(oof).iterrows():
        target=r['target']; tr=fm[(fm.target==target)&(fm.role=='train')];te=fm[(fm.target==target)&(fm.role=='test')]
        summary.append({'dataset':'all_data','target':target,'Test_n_unique_records':int(r.n),
                        'Train_R2_fold_mean':tr.R2.mean(),'Test_R2_fold_mean':te.R2.mean(),
                        'Test_R2_OOF_pooled':r.R2,'Test_MAE_OOF':r.MAE,
                        'Test_RMSE_OOF':r.RMSE,'Test_Bias_OOF':r.Bias})
    summary=pd.DataFrame(summary)
    summary.to_csv(out/'SUMMARY_Train_vs_Test_R2.csv',index=False,encoding='utf-8-sig')
    fm.to_csv(out/'fold_train_test_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(foldinfo).to_csv(out/'fold_details.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(histrows).to_csv(out/'training_history.csv',index=False,encoding='utf-8-sig')
    train.to_csv(out/'fold_training_predictions.csv',index=False,encoding='utf-8-sig')
    oof.to_csv(out/'test_OOF_predictions.csv',index=False,encoding='utf-8-sig')
    if args.cv=='loao':
        gr=[]
        for group,sub in oof.groupby(args.group_column,sort=True):
            for _,r in metrics_table(sub).iterrows():gr.append({'group':group,**r.to_dict()})
        pd.DataFrame(gr).to_csv(out/'LOAO_metrics_BY_GROUP.csv',index=False,encoding='utf-8-sig')
    print('\n',summary.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    if not args.skip_plots:plot_predictions(oof,out/'plots',args.pdf)
    if not args.skip_final:
        finalepochs=max(1,int(round(np.median([f['selected_epochs'] for f in foldinfo]))))
        sx=StandardScaler().fit(X);sy=StandardScaler().fit(Y)
        model,_,_=train_model(sx.transform(X),sy.transform(Y),finalepochs,args.batch_size,args.lr,SEED+1000)
        torch.save({'model_state_dict':model.state_dict(),'feature_columns':FEATURES,
                    'target_columns':LABELS,'architecture':'6-32-16-8-2',
                    'final_epochs':finalepochs},out/'final_ann_model.pth')
        joblib.dump(sx,out/'final_scaler_X.pkl')
        joblib.dump(sy,out/'final_scaler_y.pkl')
    (out/'run_config.json').write_text(json.dumps({
        'data':str(data),'records':len(df),'cv':args.cv,'folds':len(splits),
        'group_column':args.group_column if args.cv=='loao' else None,
        'seed':SEED,'epochs_max':args.epochs,'patience':args.patience,
        'quick_test':args.quick,'architecture':'6-32-16-8-2'},indent=2,ensure_ascii=False),encoding='utf-8')
    print('Output:',out.resolve())

if __name__=='__main__':
    p=argparse.ArgumentParser(description='ANN regression with one input CSV')
    p.add_argument('--data',default='data.csv')
    p.add_argument('--cv',choices=['loao','kfold'],default='loao')
    p.add_argument('--group-column',default='图片名称')
    p.add_argument('--folds',type=int,default=5)
    p.add_argument('--out',default=None)
    p.add_argument('--epochs',type=int,default=4000)
    p.add_argument('--patience',type=int,default=200)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--quick',action='store_true')
    p.add_argument('--pdf',action='store_true')
    p.add_argument('--skip-plots',action='store_true')
    p.add_argument('--skip-final',action='store_true')
    evaluate(p.parse_args())
