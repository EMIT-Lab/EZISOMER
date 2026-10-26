"""Shared I/O, concentration-derived de and metrics for a single RGB CSV."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = ['A_B', 'A_G', 'A_R', 'B_B', 'B_G', 'B_R']
LABELS = ['Y1', 'Y2']
TARGETS = {
    'E_concentration': ('Y1_true', 'Y1_pred'),
    'Z_concentration': ('Y2_true', 'Y2_pred'),
    'total_concentration': ('total_true', 'total_pred'),
    'de_from_concentration': ('de_true', 'de_pred'),
}


def load_dataset(csv_path):
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f'Input CSV not found: {path.resolve()}')
    df = pd.read_csv(path, encoding='utf-8-sig')
    missing = [x for x in FEATURES + LABELS if x not in df.columns]
    if missing:
        raise ValueError(f'Missing CSV columns: {missing}')
    for key in FEATURES + LABELS:
        df[key] = pd.to_numeric(df[key], errors='raise')
    if len(df) < 4:
        raise ValueError('Need at least four records')
    if not np.isfinite(df[FEATURES+LABELS].to_numpy(dtype=float)).all():
        raise ValueError('Found NaN or infinite RGB/target values; correct the input CSV')
    if (df[LABELS].to_numpy(float) < 0).any():
        raise ValueError('Target concentrations must be nonnegative')
    df.insert(0, 'record_id', np.arange(len(df), dtype=int))
    return df


def de_from_concentrations(e, z):
    """Negative predicted concentrations are clipped; zero total -> undefined de."""
    e = np.maximum(np.asarray(e, dtype=float), 0)
    z = np.maximum(np.asarray(z, dtype=float), 0)
    denom = e + z
    result = np.full(np.shape(denom), np.nan, dtype=float)
    np.divide(e-z, denom, out=result, where=denom>0)
    return result


def predicted_rows(records, pred_y, fold, role, model=None):
    out = records.copy()
    out['fold'] = fold
    out['role'] = role
    if model is not None:
        out['model'] = model
    out['Y1_true'] = out['Y1']
    out['Y2_true'] = out['Y2']
    out['Y1_pred'] = pred_y[:,0]
    out['Y2_pred'] = pred_y[:,1]
    out['total_true'] = out['Y1_true'] + out['Y2_true']
    out['total_pred'] = out['Y1_pred'] + out['Y2_pred']
    out['de_true'] = de_from_concentrations(out['Y1_true'], out['Y2_true'])
    out['de_pred'] = de_from_concentrations(out['Y1_pred'], out['Y2_pred'])
    return out


def score(true, pred):
    t, p = np.asarray(true, float), np.asarray(pred, float)
    valid = np.isfinite(t) & np.isfinite(p)
    t, p = t[valid], p[valid]
    n = len(t)
    return dict(n=n,
                R2=(float(r2_score(t,p)) if n>=2 and np.var(t)>1e-12 else np.nan),
                MAE=(float(mean_absolute_error(t,p)) if n else np.nan),
                RMSE=(float(np.sqrt(mean_squared_error(t,p))) if n else np.nan),
                Bias=(float(np.mean(p-t)) if n else np.nan))


def metrics_table(predictions):
    return pd.DataFrame([{'target':k, **score(predictions[truth],predictions[pred])}
                         for k,(truth,pred) in TARGETS.items()])


def plot_predictions(predictions, output_dir, pdf=False, prefix=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': False})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for key,(tcol,pcol) in TARGETS.items():
        true = predictions[tcol].to_numpy(float)
        pred = predictions[pcol].to_numpy(float)
        valid = np.isfinite(true) & np.isfinite(pred)
        if not valid.any():
            continue
        m=score(true,pred)
        a, b = true[valid], pred[valid]
        fig, ax = plt.subplots(figsize=(5.3,5.2))
        ax.scatter(a,b,s=20,alpha=.7)
        lo,hi=min(a.min(),b.min()),max(a.max(),b.max())
        if hi<=lo:lo-=1;hi+=1
        pad=(hi-lo)*.06
        ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],linestyle='--',color='gray',linewidth=1)
        r='N/A' if np.isnan(m['R2']) else f"{m['R2']:.3f}"
        ax.text(.03,.97,f"R2={r}  MAE={m['MAE']:.3f}\nBias={m['Bias']:.3f}  n={m['n']}",
                transform=ax.transAxes,va='top',fontsize=9,
                bbox={'boxstyle':'round','facecolor':'white','alpha':.85})
        ax.set_xlabel('True');ax.set_ylabel('Out-of-fold prediction');ax.set_title(key)
        ax.grid(alpha=.2)
        fig.tight_layout()
        name=output_dir/f'{prefix}{key}'
        fig.savefig(str(name)+'.png',dpi=150)
        if pdf:fig.savefig(str(name)+'.pdf')
        plt.close(fig)
