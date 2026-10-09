"""Chronological, frozen-condition validation of cost-adjusted trade events.

Returns describe event returns, never a portfolio return. Final observations are
used once to evaluate the frozen decision, not to search for exclusions.
"""
from datetime import date, datetime
import math


def _date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _conditions(row):
    value = row.get("conditions", [])
    if isinstance(value, dict):
        return {(str(key), str(item)) for key, item in value.items()}
    if isinstance(value, str):
        return {value}
    return {str(item) for item in value}


def trade_metrics(rows):
    """Wilson interval and net event statistics; zero returns are nonwins."""
    values = [float(row["net_return"]) for row in rows]
    n = len(values)
    wins = sum(value > 0 for value in values)
    losses = sum(value < 0 for value in values)
    positive = sum(value for value in values if value > 0)
    negative = -sum(value for value in values if value < 0)
    if n:
        p, z = wins / n, 1.959963984540054
        denominator = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator
        margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
        interval = [max(0.0, center - margin), min(1.0, center + margin)]
    else:
        interval = [None, None]
    return {"count": n, "wins": wins, "losses": losses,
            "win_rate": wins / n if n else None,
            "loss_rate": losses / n if n else None,
            "win_rate_wilson_95": interval,
            "net_expectancy": sum(values) / n if n else None,
            "event_net_return_sum": sum(values),
            "profit_factor": positive / negative if negative else None,
            "profit_factor_unbounded": bool(positive and not negative)}


def _improves(before, after, strict=True):
    if not before["count"] or not after["count"]:
        return False
    expectancy = after["net_expectancy"] - before["net_expectancy"]
    loss_change = before["loss_rate"] - after["loss_rate"]
    return expectancy > 1e-12 and loss_change > 1e-12 if strict else (
        expectancy >= -1e-12 and loss_change >= -1e-12)


def evaluate_technique(trades, train_end, validation_end, min_trades=20):
    """Evaluate explicit shared date boundaries, purging boundary-crossing trades.

    entry_date <= train_end is development; the next interval is calibration;
    later entries are final. An exit on a cutoff belongs to the earlier split.
    Every exclusion needs min_trades affected events in both development and
    calibration. Selection never reads final returns.
    """
    if min_trades < 20:
        raise ValueError("min_trades must be at least 20")
    train_end, validation_end = _date(train_end), _date(validation_end)
    if train_end >= validation_end:
        raise ValueError("train_end must precede validation_end")
    splits = {name: [] for name in ("train", "validation", "final")}
    invalid, purged = [], []
    for index, source in enumerate(trades):
        try:
            row = dict(source)
            entry, exit_date = _date(row["entry_date"]), _date(row["exit_date"])
            value = float(row["net_return"])
            if not math.isfinite(value) or exit_date < entry:
                raise ValueError("invalid return or exit before entry")
            row["net_return"] = value
            row["_conditions"] = _conditions(row)
        except (KeyError, TypeError, ValueError) as exc:
            invalid.append({"index": index, "reason": str(exc)})
            continue
        split = "train" if entry <= train_end else "validation" if entry <= validation_end else "final"
        cutoff = train_end if split == "train" else validation_end if split == "validation" else None
        if cutoff and exit_date > cutoff:
            purged.append({"index": index, "split": split, "reason": "exit_overlaps_next_split"})
        else:
            splits[split].append(row)
    baseline = {key: trade_metrics(rows) for key, rows in splits.items()}
    selected, rejected, steps = [], [], []
    development = list(splits["train"])
    tags = set().union(*(row["_conditions"] for row in development)) if development else set()
    # Greedy discovery: highest observed loss count first, recomputed after each removal.
    while tags:
        ordered = sorted(tags, key=lambda tag: (-sum(row["net_return"] < 0 for row in development if tag in row["_conditions"]), tag))
        accepted = False
        for tag in ordered:
            tags.remove(tag)
            affected = [row for row in development if tag in row["_conditions"]]
            retained = [row for row in development if tag not in row["_conditions"]]
            before, after, affected_metrics = trade_metrics(development), trade_metrics(retained), trade_metrics(affected)
            reason = ("insufficient_affected_train" if len(affected) < min_trades else
                      "insufficient_retained_train" if len(retained) < min_trades else
                      "condition_not_negative" if affected_metrics["net_expectancy"] >= 0 else
                      "no_train_improvement" if not _improves(before, after) else None)
            step = {"condition": tag, "phase": "train", "before": before, "after": after, "affected": affected_metrics, "accepted": reason is None, "reason": reason}
            steps.append(step)
            if reason:
                rejected.append(step)
            else:
                selected.append(tag)
                development = retained
                accepted = True
                break
        if not accepted:
            break
    calibration = list(splits["validation"])
    exclusions = []
    for tag in selected:
        affected = [row for row in calibration if tag in row["_conditions"]]
        retained = [row for row in calibration if tag not in row["_conditions"]]
        before, after, affected_metrics = trade_metrics(calibration), trade_metrics(retained), trade_metrics(affected)
        reason = ("insufficient_affected_validation" if len(affected) < min_trades else
                  "insufficient_retained_validation" if len(retained) < min_trades else
                  "validation_condition_not_negative" if affected_metrics["net_expectancy"] >= 0 else
                  "no_validation_improvement" if not _improves(before, after) else None)
        step = {"condition": tag, "phase": "validation", "before": before, "after": after, "affected": affected_metrics, "accepted": reason is None, "reason": reason}
        steps.append(step)
        if reason:
            rejected.append(step)
        else:
            exclusions.append(tag)
            calibration = retained
    retained = {key: trade_metrics([row for row in rows if not set(exclusions).intersection(row["_conditions"])]) for key, rows in splits.items()}
    sufficient = all(value["count"] >= min_trades for value in baseline.values())
    retained_sufficient = all(value["count"] >= min_trades for value in retained.values())
    positive = all(value["net_expectancy"] is not None and value["net_expectancy"] > 0 for value in retained.values())
    final_pass = _improves(baseline["final"], retained["final"], strict=False)
    frozen_development_pass = not exclusions or all(
        _improves(baseline[key], retained[key]) for key in ("train", "validation"))
    final_affected_support = {str(tag): sum(tag in row["_conditions"] for row in splits["final"]) for tag in exclusions}
    exclusion_support = all(count >= min_trades for count in final_affected_support.values())
    exclusion_objects = [{"condition": tag[0], "value": tag[1]} if isinstance(tag, tuple) else {"condition": tag, "value": True} for tag in exclusions]
    if not sufficient or not retained_sufficient or not exclusion_support:
        status, reason = "INSUFFICIENT", "Each baseline and retained split requires at least min_trades completed events."
    elif positive and final_pass and frozen_development_pass:
        status, reason = ("PRUNED_KEEP" if exclusions else "KEEP"), "Positive net expectancy on each split; frozen final exclusions do not worsen expectancy or loss rate."
    elif all(value["net_expectancy"] < 0 for value in baseline.values()):
        status, reason = "DROP", "Baseline net expectancy is negative in all three adequately supported splits; no pruning passes final validation."
    else:
        status, reason = "INCONCLUSIVE", "Adequate event counts, but positive performance or frozen final validation is not established."
    return {"status": status, "reason": reason, "min_trades": min_trades,
            "split_cutoffs": {"train_end": train_end.isoformat(), "validation_end": validation_end.isoformat()},
            "baseline": baseline, "retained": retained, "development_candidates": selected,
            "excluded_conditions": exclusion_objects, "exclusions": exclusion_objects, "steps": steps, "rejections": rejected,
            "final_affected_support": final_affected_support,
            "validated": status in ("KEEP", "PRUNED_KEEP", "DROP"),
            "purged_trades": purged, "invalid_trades": invalid,
            "return_basis": "cost_adjusted_event_returns_not_portfolio_return"}
