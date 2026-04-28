import numpy as np
import pandas as pd
import json

np.random.seed(2718)
N = 252
markets = {
    "KRX": {
        "assets": 8,
        "daily_mu": 0.00042,
        "daily_vol": 0.0165,
        "jump_prob": 0.035,
        "jump_sigma": 0.025,
        "momentum_phi": 0.04,
        "target_prob_adj": -0.06,
        "band_width_adj": 1.18,
    },
    "US": {
        "assets": 8,
        "daily_mu": 0.00055,
        "daily_vol": 0.0128,
        "jump_prob": 0.018,
        "jump_sigma": 0.016,
        "momentum_phi": 0.12,
        "target_prob_adj": 0.05,
        "band_width_adj": 0.92,
    },
}

regime = np.zeros(N, dtype=int)
# 0 normal, 1 rally, -1 correction
for t in range(N):
    if 35 <= t < 72 or 150 <= t < 188:
        regime[t] = 1
    elif 92 <= t < 120 or 210 <= t < 232:
        regime[t] = -1

summaries = {}
all_port_rets = []
for m, p in markets.items():
    asset_rets = []
    atr_pcts = []
    for a in range(p["assets"]):
        r = np.zeros(N)
        prev = 0.0
        beta = np.random.uniform(0.85, 1.15)
        for t in range(N):
            mu = p["daily_mu"]
            vol = p["daily_vol"]
            if regime[t] == 1:
                mu += 0.00055 if m == "US" else 0.00035
                vol *= 0.92
            elif regime[t] == -1:
                mu -= 0.00105 if m == "US" else 0.00125
                vol *= 1.35 if m == "KRX" else 1.25
            market_factor = np.random.normal(mu, vol * 0.60) * beta
            idio = np.random.normal(0, vol * 0.75)
            jump = np.random.normal(0, p["jump_sigma"]) if np.random.rand() < p["jump_prob"] else 0
            r[t] = market_factor + idio + p["momentum_phi"] * prev + jump
            r[t] = np.clip(r[t], -0.095 if m == "KRX" else -0.075, 0.095 if m == "KRX" else 0.08)
            prev = r[t]
        price = 100 * np.cumprod(1 + r)
        # OHLC range proxy: intraday range scales with daily vol and shock magnitude.
        range_pct = np.abs(np.random.normal(p["daily_vol"] * 1.15, p["daily_vol"] * 0.35, N)) + np.abs(r) * 0.25
        tr_pct = np.maximum(range_pct, np.abs(r))
        atr = pd.Series(tr_pct).rolling(14).mean().bfill().values
        atr_pcts.append(atr)
        asset_rets.append(r)
    asset_rets = np.array(asset_rets)
    atr_pcts = np.array(atr_pcts)
    port = asset_rets.mean(axis=0)
    all_port_rets.append(port)
    equity = np.cumprod(1 + port)
    dd = equity / np.maximum.accumulate(equity) - 1
    weekly = pd.Series(port).groupby(np.arange(N)//5).apply(lambda x: (1+x).prod()-1).values
    # run lengths
    signs = np.where(port >= 0, 1, -1)
    runs = []
    cur = signs[0]; length=1
    for s in signs[1:]:
        if s == cur:
            length += 1
        else:
            runs.append((cur, length)); cur=s; length=1
    runs.append((cur,length))
    up_runs = [l for s,l in runs if s==1]
    down_runs = [l for s,l in runs if s==-1]
    atr_avg = atr_pcts.mean()
    daily_quant = np.quantile(port, [0.05,0.25,0.5,0.75,0.95])
    weekly_quant = np.quantile(weekly, [0.05,0.25,0.5,0.75,0.95])
    # entry zone simulation: enter at current price - k*ATR; measure next 20d high TP and min loss
    zone_rows=[]
    ks = [("A",0.25,0.65),("B",0.65,1.10),("C",1.10,1.70)]
    horizon=20
    avg_atr = atr_pcts.mean(axis=0)
    for name,k1,k2 in ks:
        wins=[]; rets=[]; holds=[]; losses=[]
        k=(k1+k2)/2
        for t in range(30,N-horizon):
            entry = 1 - k*avg_atr[t]
            fut = np.cumprod(1+port[t+1:t+1+horizon])
            tp = 1 + (1.45 + 0.35*k) * avg_atr[t]
            sl = 1 - (0.95 + 0.25*k) * avg_atr[t]
            hit_tp_idx = np.where(fut/entry >= tp)[0]
            hit_sl_idx = np.where(fut/entry <= sl)[0]
            win = len(hit_tp_idx)>0 and (len(hit_sl_idx)==0 or hit_tp_idx[0] <= hit_sl_idx[0])
            wins.append(win)
            if win:
                holds.append(int(hit_tp_idx[0]+1))
                rets.append(tp-1)
            else:
                holds.append(horizon)
                rets.append(fut[-1]/entry-1)
                losses.append(min(0, fut.min()/entry-1))
        zone_rows.append({
            "zone": name,
            "atr_band": [k1,k2],
            "win_prob": round(float(np.mean(wins))*100,1),
            "expected_return_pct": round(float(np.mean(rets))*100,2),
            "loss_prob": round((1-float(np.mean(wins)))*100,1),
            "avg_hold_days": round(float(np.mean(holds)),1),
            "avg_failed_loss_pct": round(float(np.mean(losses))*100 if losses else -0.0,2),
            "sharpe_proxy": round(float(np.mean(rets))/(float(np.std(rets))+1e-9)*np.sqrt(252/max(np.mean(holds),1)),2),
        })
    summaries[m] = {
        "annual_return_pct": round((equity[-1]-1)*100,2),
        "daily_mean_pct": round(float(np.mean(port))*100,3),
        "daily_std_pct": round(float(np.std(port))*100,3),
        "annualized_vol_pct": round(float(np.std(port))*np.sqrt(252)*100,2),
        "mdd_pct": round(float(dd.min())*100,2),
        "daily_return_quantiles_pct": [round(float(x)*100,2) for x in daily_quant],
        "weekly_return_quantiles_pct": [round(float(x)*100,2) for x in weekly_quant],
        "up_run_avg_days": round(float(np.mean(up_runs)),2),
        "up_run_p75_days": round(float(np.quantile(up_runs,0.75)),2),
        "down_run_avg_days": round(float(np.mean(down_runs)),2),
        "down_run_p75_days": round(float(np.quantile(down_runs,0.75)),2),
        "atr_avg_pct": round(float(atr_avg)*100,2),
        "atr_p75_pct": round(float(np.quantile(atr_pcts,0.75))*100,2),
        "entry_zones": zone_rows,
        "market_adjustments": {
            "band_width_adj": p["band_width_adj"],
            "target_prob_adj_pp": int(p["target_prob_adj"]*100),
        }
    }

mixed = np.vstack(all_port_rets).mean(axis=0)
equity = np.cumprod(1+mixed)
dd = equity/np.maximum.accumulate(equity)-1
weekly = pd.Series(mixed).groupby(np.arange(N)//5).apply(lambda x:(1+x).prod()-1).values
summaries["GLOBAL_EQUAL_WEIGHT"] = {
    "annual_return_pct": round((equity[-1]-1)*100,2),
    "daily_mean_pct": round(float(np.mean(mixed))*100,3),
    "daily_std_pct": round(float(np.std(mixed))*100,3),
    "annualized_vol_pct": round(float(np.std(mixed))*np.sqrt(252)*100,2),
    "mdd_pct": round(float(dd.min())*100,2),
    "daily_return_quantiles_pct": [round(float(x)*100,2) for x in np.quantile(mixed,[0.05,0.25,0.5,0.75,0.95])],
    "weekly_return_quantiles_pct": [round(float(x)*100,2) for x in np.quantile(weekly,[0.05,0.25,0.5,0.75,0.95])],
}

print(json.dumps(summaries, ensure_ascii=False, indent=2))
