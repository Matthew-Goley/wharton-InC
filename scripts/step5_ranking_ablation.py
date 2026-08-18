"""Step 5: the three-way ablation this whole sector-expert build was for --
does expert disagreement (uncertainty_adjusted) or the agreement bonus
actually help, tested together rather than assumed.

Reuses the already-trained models (data/processed/all_sector_experts_models.joblib)
-- materials is excluded via config.EXPERTS (baseline_system_architecture.md
open question 9/11), not by retraining; combining and ranking don't depend
on the GA search at all, so nothing here needs to retrain anything.

DESIGN, pre-registered before this ran (open question 11):

- 6 arms: ranking_score in {raw, vol_adjusted, uncertainty_adjusted} x
  agreement_bonus in {0.0 (off), 0.1 (on, the existing default)}. Materials
  in/out was cut as a third axis -- already decided independently, cutting
  it from 12 arms to 6 is most of this ablation's power.
- No frequency cap here -- that's a different axis (concentration), tested
  and recorded separately; mixing it in would confound which effect any
  difference is coming from.
- Split-sample by year (2024/2025) with the standard hold_days-wide
  embargo at the boundary.
- Every arm-vs-baseline comparison goes through decompose_selection_change
  (common/only-a/only-b, block-effective SE) -- the same tool used for
  every other scoring comparison in this project, not a headline mean.
- Bonferroni-corrected significance threshold computed from the actual
  number of comparisons run, not assumed. The near-certain outcome (see
  open question 11: highest |t| observed anywhere in this project so far
  is ~1.7, corrected threshold is ~2.6-2.7 for 6 arms) is that nothing
  clears it.
- PRE-REGISTERED NULL RESPONSE: if nothing clears, the report says so
  explicitly and defaults to the simplest configuration (raw,
  agreement_bonus off) -- added complexity has to earn its place. No arm
  is named a "winner" on an uncorrected or eyeballed basis.

Usage:
    python -m scripts.step5_ranking_ablation
"""
from __future__ import annotations

import itertools
import json
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import norm

from scripts.backtest_broad_real import TEST_START, TEST_END, _rank_bucket_stats, _split_sample_validation
from src.combiner.overall_model import OverallModel
from src.config import BaselineConfig, EXPERTS
from src.data_layer import RealDataProvider
from src.data_layer.real_provider import RAW_TARGET_COLUMN
from src.evaluation.selection_diff import decompose_selection_change
from src.experts.registry import build_expert_data
from src.selection.trade_selector import TradeSelector

DATA_PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
MODEL_PATH = DATA_PROCESSED / "all_sector_experts_models.joblib"
RESULT_PATH = DATA_PROCESSED / "step5_ranking_ablation.json"

RANKING_SCORES = ["raw", "vol_adjusted", "uncertainty_adjusted"]
AGREEMENT_BONUSES = [0.0, 0.1]
BASELINE_ARM = ("raw", 0.0)  # the pre-registered null-response default

FAMILY_ALPHA = 0.05


def block_effective_mean_ci(returns, all_days, hold_days, alpha=0.05):
    """Mean + a block-effective (hold_days-wide, non-overlapping-block)
    confidence interval -- same independence unit as decompose_selection_change,
    applied to a single sample instead of a two-sample comparison."""
    if len(returns) == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None, "blocks": 0}
    idx = returns.index
    day_pos = {d: i for i, d in enumerate(np.sort(all_days))}
    blocks = len({day_pos[t] // hold_days for t in idx.get_level_values("timestamp")})
    se = returns.std() / np.sqrt(max(blocks, 1))
    z = norm.ppf(1 - alpha / 2)
    mean = float(returns.mean())
    return {
        "n": int(len(returns)), "mean": mean, "blocks": int(blocks),
        "ci_low": float(mean - z * se), "ci_high": float(mean + z * se), "se": float(se),
    }


def run_arm(ranking_score, agreement_bonus, expert_predictions, broad_spec, sector_specs, vol_risk, config,
            n_periods, budget):
    combiner = OverallModel(tuple([broad_spec] + sector_specs), agreement_bonus=agreement_bonus)
    combined = combiner.combine(expert_predictions)
    kwargs = {"risk": vol_risk} if ranking_score == "vol_adjusted" else {}
    ranked = combiner.rank(combined, ranking_score=ranking_score, **kwargs)
    trading_config = replace(config.trading, window_trading_days=n_periods, trade_budget=budget)
    trades = TradeSelector(trading_config).run(ranked)
    entries = trades[trades["action"] == "entry"]
    return entries


def main() -> None:
    config = BaselineConfig()
    provider = RealDataProvider()
    broad_spec = next(s for s in EXPERTS if s.sector is None)
    sector_specs = [s for s in EXPERTS if s.sector is not None]
    models = joblib.load(MODEL_PATH)

    test_table = provider.get_feature_table(TEST_START, TEST_END, hold_days=config.trading.hold_days)
    broad_test = build_expert_data(test_table, broad_spec)
    trained_specs = [s for s in sector_specs if s.name in models]

    print("Predicting once (predictions don't depend on ranking_score or agreement_bonus)...")
    expert_predictions = {"broad": models["broad"].predict(broad_test.X)}
    for spec in trained_specs:
        edata = build_expert_data(test_table, spec)
        if not edata.frame.empty:
            expert_predictions[spec.name] = models[spec.name].predict(edata.X)

    vol_risk = broad_test.frame["volatility_20"]
    vol_risk = vol_risk.fillna(vol_risk.max())
    raw = broad_test.frame[RAW_TARGET_COLUMN]
    n_periods = broad_test.frame.index.get_level_values("timestamp").nunique()
    budget = int(round((200 / 50) * n_periods))
    all_days = broad_test.frame.index.get_level_values("timestamp").unique().to_numpy()
    hold_days = config.trading.hold_days

    arms = list(itertools.product(RANKING_SCORES, AGREEMENT_BONUSES))
    print(f"\nRunning {len(arms)} arms: {arms}\n")

    results = {}
    for ranking_score, bonus in arms:
        label = f"{ranking_score}|bonus={bonus}"
        entries = run_arm(ranking_score, bonus, expert_predictions, broad_spec, trained_specs, vol_risk, config,
                           n_periods, budget)
        entry_raw = raw.loc[entries.index]
        overall = block_effective_mean_ci(entry_raw, all_days, hold_days)
        rank1 = entries[entries["rank"] == 1]
        rank1_stats = block_effective_mean_ci(raw.loc[rank1.index], all_days, hold_days)
        buckets = _rank_bucket_stats(entries, entry_raw)
        split = _split_sample_validation(entries, entry_raw, all_days, hold_days)
        results[label] = {
            "ranking_score": ranking_score, "agreement_bonus": bonus,
            "entries": entries, "overall": overall, "rank1": rank1_stats,
            "rank_buckets": buckets, "split_sample": split,
        }
        print(f"[{label}] n={overall['n']} mean={overall['mean']:+.4%} "
              f"CI=[{overall['ci_low']:+.4%}, {overall['ci_high']:+.4%}]  "
              f"rank1: n={rank1_stats['n']} mean={rank1_stats['mean']:+.4%}")

    # Pairwise comparisons: every arm vs. the pre-registered baseline
    # (raw, bonus off). Bonferroni over the actual number of comparisons run.
    baseline_label = f"{BASELINE_ARM[0]}|bonus={BASELINE_ARM[1]}"
    comparison_arms = [a for a in results if a != baseline_label]
    n_comparisons = len(comparison_arms)
    corrected_alpha = FAMILY_ALPHA / n_comparisons
    corrected_t = float(norm.ppf(1 - corrected_alpha / 2))
    print(f"\nBonferroni: {n_comparisons} comparisons vs. baseline '{baseline_label}', "
          f"family alpha={FAMILY_ALPHA} -> per-comparison alpha={corrected_alpha:.5f} -> "
          f"corrected |t| threshold ~= {corrected_t:.2f}")

    comparisons = {}
    any_cleared = False
    for label in comparison_arms:
        baseline_entries = results[baseline_label]["entries"]
        arm_entries = results[label]["entries"]
        diff = decompose_selection_change(baseline_entries, arm_entries, raw, all_days, hold_days)
        diff.pop("common", None)
        t = None
        if diff["significance_only_a_vs_only_b"]:
            t = diff["significance_only_a_vs_only_b"]["block_effective_t"]
        cleared = t is not None and abs(t) >= corrected_t
        any_cleared = any_cleared or cleared
        comparisons[label] = {
            "vs_baseline": baseline_label, "block_effective_t": t, "cleared_corrected_threshold": cleared,
            "only_baseline_n": diff["only_a"]["n"], "only_arm_n": diff["only_b"]["n"],
        }
        flag = "CLEARED" if cleared else "not significant"
        print(f"  {label} vs {baseline_label}: t={t}  [{flag}]")

    print("\n" + "=" * 72)
    if any_cleared:
        print("At least one arm cleared the Bonferroni-corrected threshold -- see comparisons for which.")
        decision = "at_least_one_arm_cleared"
    else:
        print(f"PRE-REGISTERED NULL RESPONSE: no arm cleared the corrected threshold "
              f"(|t| >= {corrected_t:.2f}). Defaulting to the simplest configuration: {baseline_label}.")
        print("This does not mean uncertainty_adjusted or agreement_bonus=on are wrong -- see the n=2")
        print("confound registered in OverallModel.rank()'s docstring and open question 8. It means")
        print("this ablation, at this sample size, could not distinguish any arm from the baseline.")
        decision = "null_default_to_baseline"

    output = {
        "arms": {k: {kk: vv for kk, vv in v.items() if kk != "entries"} for k, v in results.items()},
        "baseline_arm": baseline_label,
        "n_comparisons": n_comparisons,
        "family_alpha": FAMILY_ALPHA,
        "corrected_alpha": corrected_alpha,
        "corrected_t_threshold": corrected_t,
        "comparisons_vs_baseline": comparisons,
        "decision": decision,
    }
    RESULT_PATH.write_text(json.dumps(output, indent=2, default=str) + "\n")
    print(f"\nSaved: {RESULT_PATH}")


if __name__ == "__main__":
    main()
