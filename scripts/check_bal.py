import sys
sys.path.insert(0,'C:/Users/Administrator/Documents/trae_projects/StockOracle')
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
import lightgbm as lgb
p='C:/Users/Administrator/Documents/trae_projects/StockOracle/datasets/training_features.parquet'
df=pd.read_parquet(p).replace([np.inf,-np.inf],np.nan).dropna(subset=['label'])
df_sorted=df.sort_values('date')
split_date=df_sorted.iloc[int(len(df_sorted)*0.8)]['date']
from market_briefing.ml_features import FORWARD_DAYS
train_full=df[(df.date <= split_date - pd.offsets.BDay(FORWARD_DAYS)) & (df.label!=2)]
test=df[(df.date > split_date) & (df.label!=2)]
feats=['BB_lower','SMA_50','EMA_12','India_VIX','ADX','volatility_20d','relative_strength','trend_spread_20_50']
params={'n_estimators':400,'learning_rate':0.02,'max_depth':4,'num_leaves':15,'min_child_samples':60,'subsample':0.8,'colsample_bytree':0.75,'reg_lambda':4.0}
m=lgb.LGBMClassifier(**params, verbose=-1, random_state=42)
m.fit(train_full[feats], train_full.label)
pred=m.predict_proba(test[feats])[:,1]
pred_label=(pred>=0.5).astype(int)
print('auc',roc_auc_score(test.label,pred))
print('bal',balanced_accuracy_score(test.label,pred_label))
print('acc',accuracy_score(test.label,pred_label))
# also test with multiclass
