"""Before/after audit of volume confirmation inside the original box plan.

Price geometry, targets, stops and next-session fills use the existing service.
Policies are compared on chronological development/validation periods with an
entire forward window purged at the boundary, one open position per symbol.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from market_briefing.box_plan import build_execution_plan
from market_briefing.box_sequence import analyze_box_sequence
from market_briefing.box_volume import assess_box_volume, allows_box_entry, POLICIES, VERSION
from scripts.audit_box_plan import HORIZON, simulate, wilson
from scripts.audit_box_tactics import as_dd, load_prices


def target_first(dd, i, sim, entry_high, stop, target):
    f = sim["fill_index"]
    at_open = dd["Open"][f] <= entry_high
    for j in range(f, sim["exit_index"]+1):
        op, hi, lo = (dd[k][j] for k in ("Open", "High", "Low"))
        if j > f or at_open:
            if op <= stop:
                return False
            if op >= target:
                return True
        if lo <= stop:
            return False
        if j == f and not at_open:
            continue
        if hi >= target:
            return True
    return False


def collect(files, stride=1):
    rows, source, snapshots = [], {}, 0
    started=time.perf_counter()
    for path in files:
        frame=load_prices(str(path))
        if frame is None:
            continue
        trailing=0
        while not frame.empty and frame.iloc[-1].isna().any():
            frame=frame.iloc[:-1]
            trailing+=1
        # Historical audit cache has aligned rows. Do not compress missing sessions.
        if frame.isna().any().any() or not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
            source[path.stem] = {"rejected": "missing/unsorted/duplicate observations"}
            continue
        dd=as_dd(frame)
        market="KRX" if path.stem.endswith(("_KS","_KQ")) else "US"
        source[path.stem] = {"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"trailing_incomplete_rows":trailing,
                             "first":dd["Date"][0],"last":dd["Date"][-1],"bars":len(dd["Close"])}
        before=len(rows)
        for i in range(260,len(dd["Close"])-HORIZON,stride):
            prefix={k:v[:i+1] for k,v in dd.items()}
            seq=analyze_box_sequence(prefix)
            plan=build_execution_plan(seq, current=dd["Close"][i],market=market,
                                      now=dd["Date"][i],params={"volume_policy":"off"})
            snapshots+=1
            if plan["verdict"]["key"]!="buy":
                continue
            prices=plan.get("prices") or {}
            entry=(prices.get("entry") or {}).get("high")
            stop=(prices.get("stop") or {}).get("price")
            targets=[t["price"] for t in prices.get("targets") or []]
            if entry is None or stop is None or not targets:
                continue
            profile=seq.get("volume_evidence")
            passes={p:allows_box_entry(assess_box_volume(profile,plan["strategy"]["key"],p)) for p in POLICIES}
            checked=build_execution_plan(seq,current=dd["Close"][i],market=market,now=dd["Date"][i])
            if (checked["verdict"]["key"]=="buy")!=passes["breakout"]:
                raise AssertionError(f"service/replay volume mismatch: {path.stem} {dd['Date'][i]}")
            sim=simulate(dd,i,entry,stop,targets,.0025 if market=="KRX" else .001)
            record={"ticker":path.stem,"market":market,"date":dd["Date"][i],"index":i,
                    "quote":dd["Close"][i],
                    "horizon_end":dd["Date"][i+HORIZON],"strategy":plan["strategy"]["key"],
                    "volume":profile,"passes":passes,"simulation":sim}
            if sim["status"]=="filled":
                record["target1_before_stop"] = target_first(dd,i,sim,entry,stop,targets[0])
            rows.append(record)
        print(f"{path.stem}: {len(rows)-before} opportunities; {snapshots} prefixes; {time.perf_counter()-started:.1f}s",flush=True)
    return {"feature_version":VERSION,"snapshots":snapshots,"stride":stride,"sources":source,"rows":rows}


def portfolio_rows(rows,policy,split,period):
    busy={}; trades=[]; orders=Counter()
    for row in sorted(rows,key=lambda r:(r["ticker"],r["index"])):
        if period=="development" and not row["horizon_end"] < split:
            continue
        if period=="validation" and row["date"] < split:
            continue
        if not row["passes"].get(policy) or row["index"]<=busy.get(row["ticker"],-1):
            continue
        sim=row["simulation"]
        orders[sim["status"]]+=1
        if sim["status"]!="filled":
            continue
        busy[row["ticker"]]=sim["exit_index"]
        trades.append(row)
    return trades,dict(orders)


def summary(rows):
    if not rows:
        return {"n":0}
    n=len(rows)
    wins=sum(row["simulation"]["r"]>0 for row in rows)
    hit=sum(bool(row["target1_before_stop"]) for row in rows)
    return {"n":n,"win_rate":wins/n,"win_ci":wilson(wins,n),
            "target1_before_stop_rate":hit/n,"target1_ci":wilson(hit,n),
            "mean_r":mean(row["simulation"]["r"] for row in rows),
            "mean_return_pct":mean(row["simulation"]["return_pct"] for row in rows),
            "stop_rate":sum(row["simulation"]["exit"]=="stop" for row in rows)/n,
            "market":{m:summary([r for r in rows if r["market"]==m]) for m in ("KRX","US")
                      if len({r["market"] for r in rows})>1},
            "strategy":{s:summary([r for r in rows if r["strategy"]==s]) for s in ("support","retest")
                        if len({r["strategy"] for r in rows})>1}}


def evaluate(data,split="2025-01-01",policies=POLICIES,scan_price_cap=False):
    result={"feature_version":data["feature_version"],"sources":data["sources"],"snapshots":data["snapshots"],
            "stride":data["stride"],"split":split,"horizon":HORIZON,"policies":{}}
    records=data["rows"]
    if scan_price_cap:
        records=[r for r in records if r["market"]!="US" or r.get("quote",float('inf'))<=23]
    result["scan_us_price_cap"]=23 if scan_price_cap else None
    for policy in policies:
        result["policies"][policy]={}
        for period in ("development","validation"):
            rows,orders=portfolio_rows(records,policy,split,period)
            result["policies"][policy][period]={**summary(rows),"orders":orders}
    base=result["policies"]["off"]["development"]
    eligible=[p for p in policies if p!="off" and
        result["policies"][p]["development"].get("n",0)>=max(40,.5*base.get("n",0)) and
        result["policies"][p]["development"].get("mean_r",-100)>base.get("mean_r",0) and
        result["policies"][p]["development"].get("target1_before_stop_rate",0)>=base.get("target1_before_stop_rate",0)]
    chosen=max(eligible,key=lambda p:result["policies"][p]["development"]["mean_r"]) if eligible else "off"
    result["development_selected_policy"]=chosen
    result["production_policy"]="breakout"
    result["production_policy_basis"]="fixed event-only confirmation; stronger support-volume gates rejected after validation"
    result["service_replay_parity_checked"]=True
    result["selection_basis"]="development only; retained >=50% and >=40 fills; higher mean R and no lower first-target-before-stop rate"
    result["matched_baseline"]={}
    for period in ("development","validation"):
        original,_=portfolio_rows(records,"off",split,period)
        result["matched_baseline"][period]={
            "confirmed":summary([r for r in original if r["passes"].get("breakout")]),
            "not_confirmed":summary([r for r in original if not r["passes"].get("breakout")])}
    result["limits"]=["current large-cap cache: survivorship bias", "daily bars; conservative same-bar stop priority",
                      "volume filter changes entry selection, not box geometry or targets", "validation is not a guaranteed future edge"]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stride',type=int,default=1)
    parser.add_argument('--tickers',default='')
    parser.add_argument('--split',default='2025-01-01')
    parser.add_argument('--cache-dir',type=Path,default=ROOT/'datasets/audit_cache')
    parser.add_argument('--policies',default=','.join(POLICIES))
    parser.add_argument('--scan-price-cap',action='store_true')
    parser.add_argument('--rows',type=Path,default=ROOT/'tmp/box_volume_candidates_20261111.json')
    parser.add_argument('--reuse',action='store_true')
    parser.add_argument('--out',type=Path,default=ROOT/'tmp/box_volume_comparison_20261111.json')
    args=parser.parse_args()
    if args.reuse:
        data=json.loads(args.rows.read_text(encoding='utf-8'))
        if data['feature_version']!=VERSION:
            parser.error('volume feature version mismatch; rebuild rows')
    else:
        files=sorted(args.cache_dir.glob('*.csv'))
        if args.tickers:
            files=[p for p in files if p.stem in args.tickers.split(',')]
        data=collect(files,args.stride)
        args.rows.parent.mkdir(parents=True,exist_ok=True)
        args.rows.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    policies=tuple(args.policies.split(','))
    if 'off' not in policies or any(p not in POLICIES for p in policies):
        parser.error('known policies including off required')
    result=evaluate(data,args.split,policies,args.scan_price_cap)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({"selected":result["development_selected_policy"],
        "comparison":{p:{period:{k:r.get(k) for k in ('n','win_rate','target1_before_stop_rate','mean_r','mean_return_pct')}
                    for period,r in result['policies'][p].items()} for p in policies}},indent=2))


if __name__=='__main__':
    main()
