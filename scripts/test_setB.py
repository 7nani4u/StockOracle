import sys
sys.path.insert(0,'C:/Users/Administrator/Documents/trae_projects/StockOracle')
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from market_briefing.ml_features import FORWARD_DAYS
p='C:/Users/Administrator/Documents/trae_projects/StockOracle/datasets/training_features.parquet'
df=pd.read_parquet(p).replace([np.inf,-np.inf],np.nan).dropna(subset=['label'])
df_sorted=df.sort_values('date')
split_date=df_sorted.iloc[int(len(df_sorted)*0.8)]['date']
train_full=df[(df.date <= split_date - pd.offsets.BDay(FORWARD_DAYS)) & (df.label!=2)]
test=df[(df.date > split_date) & (df.label!=2)]
feats=['BB_lower','SMA_50','EMA_12','India_VIX','ADX','volatility_20d','relative_strength','trend_spread_20_50']
def walk(feats, params):
    data=train_full.sort_values('date')
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
        tr=tr[tr.label!=2]
        te=te[te.label!=2]
        if len(tr)<100 or len(te)<20 or tr.label.nunique()<2 or te.label.nunique()<2: continue
        m=lgb.LGBMClassifier(**params, verbose=-1, random_state=42)
        m.fit(tr[feats], tr.label)
        pred=m.predict_proba(te[feats])[:,1]
        aucs.append(roc_auc_score(te.label, pred))
    return float(np.mean(aucs)), aucs
def hold(feats, params):
    m=lgb.LGBMClassifier(**params, verbose=-1, random_state=42)
    m.fit(train_full[feats], train_full.label)
    pred=m.predict_proba(test[feats])[:,1]
    return roc_auc_score(test.label, pred)

cands=[
    {'n_estimators':300,'max_depth':5,'learning_rate':0.03,'num_leaves':31,'subsample':0.8,'colsample_bytree':0.8},
    {'n_estimators':400,'learning_rate':0.02,'max_depth':4,'num_leaves':15,'min_child_samples':60,'subsample':0.8,'colsample_bytree':0.75,'reg_lambda':4.0},
    {'n_estimators':500,'learning_rate':0.02,'max_depth':5,'num_leaves':16,'min_child_samples':70,'subsample':0.80,'colsample_bytree':0.80,'reg_lambda':6.0},
    {'n_estimators':800,'learning_rate':0.03,'max_depth':6,'num_leaves':31,'min_child_samples':80,'subsample':0.85,'colsample_bytree':0.85,'bagging_freq':5},
    {'n_estimators':260,'learning_rate':0.02,'max_depth':3,'num_leaves':5,'min_child_samples':160,'subsample':0.80,'colsample_bytree':0.65,'reg_lambda':12.0,'reg_alpha':1.0},
]
for c in cands:
    w,aucs=walk(feats,c)
    h=hold(feats,c)
    print(f"params depth {c.get('max_depth')} leaves {c.get('num_leaves')} walk {w:.4f} {aucs} hold {h:.4f}")
# ensemble
c=cands[0]
m1=lgb.LGBMClassifier(**c, verbose=-1, random_state=42)
m2=lgb.LGBMClassifier(**c, verbose=-1, random_state=123)
m1.fit(train_full[feats], train_full.label)
m2.fit(train_full[feats], train_full.label)
p1=m1.predict_proba(test[feats])[:,1]
p2=m2.predict_proba(test[feats])[:,1]
print('ensemble hold', roc_auc_score(test.label, (p1+p2)/2))
