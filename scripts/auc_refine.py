import sys
sys.path.insert(0,'C:/Users/Administrator/Documents/trae_projects/StockOracle')
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from market_briefing.ml_features import FEATURE_COLS, FORWARD_DAYS

p='C:/Users/Administrator/Documents/trae_projects/StockOracle/datasets/training_features.parquet'
df=pd.read_parquet(p).replace([np.inf,-np.inf],np.nan).dropna(subset=['label'])
df_sorted=df.sort_values('date')
split_date=df_sorted.iloc[int(len(df_sorted)*0.8)]['date']
train_full=df[(df.date <= split_date - pd.offsets.BDay(FORWARD_DAYS)) & (df.label!=2)]
test=df[(df.date > split_date) & (df.label!=2)]
print(f"split {split_date} train {len(train_full)} test {len(test)}")

base=['BB_lower','SMA_50','EMA_12','India_VIX','ADX','volatility_20d']
candidates=['BB_middle','SMA_20','EMA_26','BB_width','SMA_cross','price_momentum_5d','price_momentum_10d','price_momentum_20d','high_low_range','close_to_high_ratio','volume_ratio_20d','ATR_14','stochastic_k','stochastic_d','VWAP_distance','OBV_ratio','NIFTY_return','BANKNIFTY_return','relative_strength','market_is_krx','market_return_20d','trend_spread_20_50','price_return_1d','volatility_rank_60','price_momentum_60d','price_momentum_120d','price_position_120d','trend_hit_rate_60','RSI_14','RSI_21','MACD','MACD_signal','MACD_hist','volume_momentum_5d']

def walk(feats, params={'n_estimators':300,'max_depth':5,'learning_rate':0.03,'num_leaves':31,'subsample':0.8,'colsample_bytree':0.8}):
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
    return float(np.mean(aucs)) if aucs else 0, aucs

def hold(feats, params={'n_estimators':300,'max_depth':5,'learning_rate':0.03,'num_leaves':31,'subsample':0.8,'colsample_bytree':0.8}):
    m=lgb.LGBMClassifier(**params, verbose=-1, random_state=42)
    m.fit(train_full[feats], train_full.label)
    pred=m.predict_proba(test[feats])[:,1]
    return roc_auc_score(test.label, pred)

params={'n_estimators':300,'max_depth':5,'learning_rate':0.03,'num_leaves':31,'subsample':0.8,'colsample_bytree':0.8}
w0,_=walk(base, params)
h0=hold(base, params)
print(f"BASE walk {w0:.4f} hold {h0:.4f} feats {base}")

# try adding one feature
results=[]
for feat in candidates:
    if feat in base: continue
    feats=base+[feat]
    w,_=walk(feats, params)
    h=hold(feats, params)
    results.append((w,h,feat))
    print(f"add {feat:25s} walk {w:.4f} hold {h:.4f} delta_w {w-w0:.4f} delta_h {h-h0:.4f}")

# sort by walk
results_sorted=sorted(results, key=lambda x: x[0], reverse=True)
print("\n=== TOP BY WALK ===")
for w,h,feat in results_sorted[:10]:
    print(f"{feat:25s} walk {w:.4f} hold {h:.4f}")
print("\n=== TOP BY HOLD ===")
for w,h,feat in sorted(results, key=lambda x: x[1], reverse=True)[:10]:
    print(f"{feat:25s} walk {w:.4f} hold {h:.4f}")

# try best combo: add top walk feature to base and test 2-feature additions
best_walk_feat=results_sorted[0][2]
print(f"\nBest walk add: {best_walk_feat}")
base2=base+[best_walk_feat]
w2,_=walk(base2, params)
h2=hold(base2, params)
print(f"base2 walk {w2:.4f} hold {h2:.4f}")

# try adding second feature to base2
results2=[]
for feat in candidates:
    if feat in base2: continue
    feats=base2+[feat]
    w,_=walk(feats, params)
    h=hold(feats, params)
    results2.append((w,h,feat))
    # print only if both high
    if w>0.535 and h>0.545:
        print(f"  candidate 2add {feat:25s} walk {w:.4f} hold {h:.4f} ***")

results2_sorted=sorted(results2, key=lambda x: x[0], reverse=True)
print("\n=== TOP 2-ADD BY WALK ===")
for w,h,feat in results2_sorted[:10]:
    print(f"{feat:25s} walk {w:.4f} hold {h:.4f}")

# also try ultra params for base
print("\n=== BASE WITH ULTRA ===")
ultra={'n_estimators':260,'learning_rate':0.02,'max_depth':3,'num_leaves':5,'min_child_samples':160,'subsample':0.80,'colsample_bytree':0.65,'reg_lambda':12.0,'reg_alpha':1.0,'min_split_gain':0.03}
w_u,_=walk(base, ultra)
h_u=hold(base, ultra)
print(f"ultra walk {w_u:.4f} hold {h_u:.4f}")

# try ensemble of 2 models
print("\n=== ENSEMBLE TEST ===")
import itertools
feats_list=[base, base+[best_walk_feat]]
params_list=[params, ultra]
# simple average of 2 models
for feats in [base, base+[best_walk_feat]]:
    # train 2 seeds
    m1=lgb.LGBMClassifier(**params, verbose=-1, random_state=42)
    m2=lgb.LGBMClassifier(**params, verbose=-1, random_state=123)
    m1.fit(train_full[feats], train_full.label)
    m2.fit(train_full[feats], train_full.label)
    p1=m1.predict_proba(test[feats])[:,1]
    p2=m2.predict_proba(test[feats])[:,1]
    avg=(p1+p2)/2
    hold_avg=roc_auc_score(test.label, avg)
    print(f"ensemble 2 seeds feats {feats[:3]}... hold {hold_avg:.4f} vs single {hold(feats, params):.4f}")
