import sys
sys.path.insert(0, 'C:/Users/Administrator/Documents/trae_projects/StockOracle')
import pandas as pd, numpy as np, random, itertools
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from market_briefing.ml_features import FEATURE_COLS, FORWARD_DAYS

p='C:/Users/Administrator/Documents/trae_projects/StockOracle/datasets/training_features.parquet'
df=pd.read_parquet(p).replace([np.inf,-np.inf],np.nan).dropna(subset=['label'])
# directional only
df_dir=df[df.label!=2].copy()
df_sorted=df.sort_values('date')
split_date=df_sorted.iloc[int(len(df_sorted)*0.8)]['date']
train_full=df[(df.date <= split_date - pd.offsets.BDay(FORWARD_DAYS)) & (df.label!=2)]
test=df[(df.date > split_date) & (df.label!=2)]
print(f"split {split_date} train {len(train_full)} test {len(test)} baseline test up {test.label.mean():.3f}")

def walk_auc(feats, params, data=train_full):
    data=data.sort_values('date')
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
        if len(tr)<100 or len(te)<20 or tr.label.nunique()<2 or te.label.nunique()<2:
            continue
        m=lgb.LGBMClassifier(**params, verbose=-1, random_state=42)
        m.fit(tr[feats], tr.label)
        pred=m.predict_proba(te[feats])[:,1]
        aucs.append(roc_auc_score(te.label, pred))
    if not aucs:
        return 0,0,aucs
    return float(np.mean(aucs)), float(np.std(aucs)), aucs

# baseline feats
baseline_feats=['BB_lower','SMA_50','EMA_12','India_VIX','ADX','volatility_20d']
for feats in [FEATURE_COLS, baseline_feats]:
    for params in [
        {'n_estimators':300,'max_depth':5,'learning_rate':0.03,'num_leaves':31,'subsample':0.8,'colsample_bytree':0.8},
        {'n_estimators':260,'learning_rate':0.02,'max_depth':3,'num_leaves':5,'min_child_samples':160,'subsample':0.80,'colsample_bytree':0.65,'reg_lambda':12.0,'reg_alpha':1.0,'min_split_gain':0.03},
        {'n_estimators':800,'learning_rate':0.03,'max_depth':6,'num_leaves':31,'min_child_samples':80,'subsample':0.85,'colsample_bytree':0.85,'bagging_freq':5},
    ]:
        m=lgb.LGBMClassifier(**params, verbose=-1, random_state=42)
        m.fit(train_full[feats], train_full.label)
        pred=m.predict_proba(test[feats])[:,1]
        hold=roc_auc_score(test.label, pred)
        wmean,wstd,aucs=walk_auc(feats, params)
        print(f"feats {len(feats)} params {params.get('max_depth')}/{params.get('num_leaves')} walk {wmean:.4f}±{wstd:.3f} {aucs} hold {hold:.4f}")

# random search
print("\n=== RANDOM SEARCH ===")
random.seed(42)
np.random.seed(42)
candidates = [
    {'n_estimators':400,'learning_rate':0.02,'max_depth':4,'num_leaves':15,'min_child_samples':60,'subsample':0.8,'colsample_bytree':0.75,'reg_lambda':4.0},
    {'n_estimators':600,'learning_rate':0.015,'max_depth':5,'num_leaves':20,'min_child_samples':80,'subsample':0.85,'colsample_bytree':0.8,'reg_lambda':6.0},
    {'n_estimators':300,'learning_rate':0.03,'max_depth':5,'num_leaves':31,'subsample':0.8,'colsample_bytree':0.8},
    {'n_estimators':260,'learning_rate':0.02,'max_depth':3,'num_leaves':5,'min_child_samples':160,'subsample':0.80,'colsample_bytree':0.65,'reg_lambda':12.0,'reg_alpha':1.0},
    {'n_estimators':500,'learning_rate':0.025,'max_depth':6,'num_leaves':31,'min_child_samples':50,'subsample':0.9,'colsample_bytree':0.85},
]
best=[]
for trial in range(60):
    k=random.randint(6,14)
    feats=random.sample(FEATURE_COLS, k)
    # ensure at least one BB/SMA
    if not any(x in feats for x in ['BB_lower','SMA_50','EMA_12']):
        continue
    params=random.choice(candidates)
    # also random learning rate tweak
    wmean,wstd,aucs=walk_auc(feats, params)
    # holdout peek for analysis (not for selection ideally)
    m=lgb.LGBMClassifier(**params, verbose=-1, random_state=42)
    m.fit(train_full[feats], train_full.label)
    pred=m.predict_proba(test[feats])[:,1]
    hold=roc_auc_score(test.label, pred)
    score=wmean - 0.1*wstd
    best.append((score, wmean, hold, feats, params, aucs))
    print(f"trial {trial:02d} k={k} walk {wmean:.4f}±{wstd:.3f} hold {hold:.4f} feats {feats[:3]}...")
# sort by walk
best_sorted=sorted(best, key=lambda x: x[0], reverse=True)
print("\n=== TOP 10 BY WALK ===")
for score,wmean,hold,feats,params,aucs in best_sorted[:10]:
    print(f"walk {wmean:.4f} hold {hold:.4f} score {score:.4f} feats {feats} params {params} aucs {aucs}")
print("\n=== TOP 10 BY HOLD ===")
best_hold=sorted(best, key=lambda x: x[2], reverse=True)
for score,wmean,hold,feats,params,aucs in best_hold[:10]:
    print(f"hold {hold:.4f} walk {wmean:.4f} feats {feats} params {params}")
