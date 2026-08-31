import sys
sys.path.insert(0,'C:/Users/Administrator/Documents/trae_projects/StockOracle')
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
p='C:/Users/Administrator/Documents/trae_projects/StockOracle/datasets/training_features.parquet'
df=pd.read_parquet(p).replace([np.inf,-np.inf],np.nan).dropna(subset=['label'])
from market_briefing.ml_features import FORWARD_DAYS
feats=['BB_lower','SMA_50','EMA_12','India_VIX','ADX','volatility_20d','relative_strength','trend_spread_20_50']
params={'n_estimators':400,'learning_rate':0.02,'max_depth':4,'num_leaves':15,'min_child_samples':60,'subsample':0.8,'colsample_bytree':0.75,'reg_lambda':4.0}
def walk(df, feats, params, multiclass=False):
    data=df.sort_values('date')
    dates=pd.Index(data['date'].drop_duplicates().sort_values())
    # restrict to train_full range? use full df for walk like train_ml_model does on train_df only
    # we simulate train_full = first 80% with embargo
    df_sorted=data.sort_values('date')
    split_date=df_sorted.iloc[int(len(df_sorted)*0.8)]['date']
    train_full=data[data.date <= split_date - pd.offsets.BDay(FORWARD_DAYS)]
    # now walk within train_full
    data=train_full
    dates=pd.Index(data['date'].drop_duplicates().sort_values())
    first_start=int(len(dates)*0.50)
    fold_width=max(20, int((len(dates)-first_start)/3))
    aucs=[]
    for fold in range(3):
        s_idx=first_start+fold*fold_width
        e_idx=min(len(dates), s_idx+fold_width)
        if e_idx-s_idx<10: continue
        test_start,test_end=dates[s_idx],dates[e_idx-1]
        embargo=test_start - pd.offsets.BDay(FORWARD_DAYS)
        tr=data[data.date <= embargo]
        te=data[(data.date>=test_start)&(data.date<=test_end)]
        if not multiclass:
            tr=tr[tr.label!=2]
            te=te[te.label!=2]
            if len(tr)<100 or len(te)<20 or tr.label.nunique()<2 or te.label.nunique()<2: continue
            m=lgb.LGBMClassifier(**params, verbose=-1, random_state=42)
            m.fit(tr[feats], tr.label)
            pred=m.predict_proba(te[feats])[:,1]
            aucs.append(roc_auc_score(te.label, pred))
        else:
            # multiclass
            te_dir=te[te.label!=2]
            if len(tr)<100 or len(te_dir)<20 or tr.label.nunique()<2 or te_dir.label.nunique()<2: continue
            m=lgb.LGBMClassifier(**params, verbose=-1, random_state=42, objective='multiclass', num_class=3, class_weight='balanced')
            m.fit(tr[feats], tr.label)
            proba=m.predict_proba(te_dir[feats])
            cond=proba[:,1]/np.clip(proba[:,0]+proba[:,1],1e-8,None)
            aucs.append(roc_auc_score(te_dir.label, cond))
    return aucs, float(np.mean(aucs)) if aucs else 0
df_full=pd.read_parquet(p).replace([np.inf,-np.inf],np.nan).dropna(subset=['label'])
for multi in [False, True]:
    aucs, mean=walk(df_full, feats, params, multiclass=multi)
    print(f"multiclass={multi} aucs {aucs} mean {mean:.4f}")
