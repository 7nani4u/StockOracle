import sys
sys.path.insert(0,'C:/Users/Administrator/Documents/trae_projects/StockOracle')
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import lightgbm as lgb
p='C:/Users/Administrator/Documents/trae_projects/StockOracle/datasets/training_features.parquet'
df=pd.read_parquet(p).replace([np.inf,-np.inf],np.nan).dropna(subset=['label'])
from market_briefing.ml_features import FORWARD_DAYS, FEATURE_COLS
# use 8feats
feats=['BB_lower','SMA_50','EMA_12','India_VIX','ADX','volatility_20d','relative_strength','trend_spread_20_50']
df_sorted=df.sort_values('date')
split_date=df_sorted.iloc[int(len(df_sorted)*0.8)]['date']
train_full=df[(df.date <= split_date - pd.offsets.BDay(FORWARD_DAYS)) & (df.label!=2)]
test=df[(df.date > split_date) & (df.label!=2)]
params={'n_estimators':400,'learning_rate':0.02,'max_depth':4,'num_leaves':15,'min_child_samples':60,'subsample':0.8,'colsample_bytree':0.75,'reg_lambda':4.0}
m=lgb.LGBMClassifier(**params, verbose=-1, random_state=42)
m.fit(train_full[feats], train_full.label)
pred=m.predict_proba(test[feats])[:,1]
print('raw hold auc', roc_auc_score(test.label, pred))
print('raw balanced', balanced_accuracy_score(test.label, (pred>=0.5).astype(int)))
print('raw pred stats', pred.min(), pred.max(), pred.mean())
# calibrate
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score as auc2
# OOF for calibration
def oof():
    data=train_full.sort_values('date')
    dates=pd.Index(data['date'].drop_duplicates().sort_values())
    first_start=int(len(dates)*0.50)
    fold_width=max(20, int((len(dates)-first_start)/3))
    probs=[]
    labels=[]
    for fold in range(3):
        s_idx=first_start+fold*fold_width
        e_idx=min(len(dates), s_idx+fold_width)
        if e_idx-s_idx<10: continue
        test_start,test_end=dates[s_idx],dates[e_idx-1]
        embargo=test_start - pd.offsets.BDay(FORWARD_DAYS)
        tr=data[data.date <= embargo]
        te=data[(data.date>=test_start)&(data.date<=test_end)]
        tr=tr[tr.label!=2]
        te=te[te.label!=2]
        if len(tr)<100 or len(te)<20: continue
        mm=lgb.LGBMClassifier(**params, verbose=-1, random_state=42)
        mm.fit(tr[feats], tr.label)
        pr=mm.predict_proba(te[feats])[:,1]
        probs.extend(pr.tolist())
        labels.extend(te.label.tolist())
    return np.array(probs), np.array(labels)
probs, labels = oof()
print('oof auc', roc_auc_score(labels, probs))
print('oof balanced 0.5', balanced_accuracy_score(labels, (probs>=0.5).astype(int)))
# try threshold sweep
for thr in [0.5,0.52,0.55,0.58,0.6]:
    print(f"oof thr {thr} bal {balanced_accuracy_score(labels, (probs>=thr).astype(int)):.4f}")
# calibrate
def calibrate(y_proba, y_true):
    from scipy.optimize import minimize
    def obj(p):
        a,b=p
        eps=1e-8
        pp=np.clip(y_proba,eps,1-eps)
        logit=np.log(pp/(1-pp))
        pcal=1/(1+np.exp(a*logit+b))
        pcal=np.clip(pcal,eps,1-eps)
        return -np.mean(y_true*np.log(pcal)+(1-y_true)*np.log(1-pcal))
    res=minimize(obj, x0=[-1,0], method='L-BFGS-B', bounds=[(-5,-1e-4),(None,None)])
    return res.x
a,b=calibrate(probs, labels)
print(f'calib a {a:.4f} b {b:.4f}')
eps=1e-8
logit=np.log(np.clip(pred,eps,1-eps)/(1-np.clip(pred,eps,1-eps)))
pcal=1/(1+np.exp(a*logit+b))
print('cal hold auc', roc_auc_score(test.label, pcal))
print('cal balanced 0.5', balanced_accuracy_score(test.label, (pcal>=0.5).astype(int)))
for thr in [0.5,0.55,0.6]:
    print(f"cal thr {thr} bal {balanced_accuracy_score(test.label, (pcal>=thr).astype(int)):.4f}")
print('cal pcal stats', pcal.min(), pcal.max(), pcal.mean())
