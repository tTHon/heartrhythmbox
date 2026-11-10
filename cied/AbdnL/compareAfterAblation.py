#!/usr/bin/env python
"""
compare_oof_stageI.py
=====================
Paired comparison of initialisation arms for abandoned-lead detection (Stage I),
using the per-fold out-of-fold files written by finetuneCV_ablation.py
(<arm_dir>/fold_N/oof_predictions.csv).

The FIRST --arm is the reference (e.g. busch_enc); every other arm is compared to it.
Differences are always  reference - other  (positive = reference better).

Run (no paths needed - folders are built from the same naming rule as finetuneCV_ablation.py):
  python compare_oof_stageI.py
  -> compares busch_enc (reference) vs imagenet at train_frac 1, seed 42, final-epoch weights,
     reading <root>/<init>_frac1_seed42/fold_N/oof_predictions.csv

Other cases (still no paths):
  python compare_oof_stageI.py --inits busch_enc imagenet scratch
  python compare_oof_stageI.py --train_frac 0.25
  python compare_oof_stageI.py --ckpt_select          (arms trained with --ckpt_select)
  Override the default root only if the models moved:  --root D:/other/models
  --arm NAME=DIR remains available as a manual override (rarely needed).

What it reports
  PRIMARY (pre-specified): case-level AUC of --score (default area_p0.8), pooled over folds;
      paired difference with DeLong z-test and paired case-level bootstrap CI.
  SENSITIVITY: same for max_prob_abdn and soft_area_abdn (exploratory, no multiplicity control).
  PER-FOLD AUC: consistency across folds.
  OPERATING POINT: specificity at a fixed sensitivity (--op_sens, default 0.975) from pooled scores.
  DICE: per-case abandoned-lead Dice (positives only) and generator Dice (all cases);
      paired bootstrap CI for the mean difference and Wilcoxon signed-rank p.

Caveats (also printed):
  * Pooling scores from 5 different fold models assumes comparable score scales; per-fold AUCs
    are shown for that reason.
  * Cases are resampled as independent units; patient-level clustering is not modelled.
  * Pixel areas are at patch_size resolution (see oof_predictions.csv) - AUC does not depend on
    this, fixed area cut-offs do.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, wilcoxon
from sklearn.metrics import roc_curve


# ──────────────────────────────────────────────────────────────
# AUC helpers
# ──────────────────────────────────────────────────────────────
def auc_fast(y, s):
    """Rank-based AUC with average ranks for ties."""
    y = np.asarray(y)
    m = int(y.sum())
    n = len(y) - m
    if m == 0 or n == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - m * (m + 1) / 2.0) / (m * n)


def _midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong_paired(y, s1, s2):
    """DeLong test for two correlated AUCs on the same cases.
    Returns auc1, auc2, diff, se_diff, z, p."""
    y = np.asarray(y).astype(int)
    pos = y == 1
    m, n = int(pos.sum()), int((~pos).sum())
    scores = np.vstack([s1, s2]).astype(float)
    tx = np.empty((2, m))
    ty = np.empty((2, n))
    tz = np.empty((2, m + n))
    for r in range(2):
        ps, ns = scores[r][pos], scores[r][~pos]
        tz[r] = _midrank(np.concatenate([ps, ns]))
        tx[r] = _midrank(ps)
        ty[r] = _midrank(ns)
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    diff = aucs[0] - aucs[1]
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return aucs[0], aucs[1], diff, 0.0, 0.0, 1.0
    se = float(np.sqrt(var))
    z = diff / se
    return aucs[0], aucs[1], diff, se, z, float(2 * norm.sf(abs(z)))


def spec_at_sens(y, s, target):
    """Specificity at the first (highest-threshold) ROC point with sensitivity >= target."""
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    fpr, tpr, _ = roc_curve(y, s)
    idx = np.where(tpr >= target)[0]
    return float(1.0 - fpr[idx[0]]) if len(idx) else np.nan


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def pct_ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return (np.nan, np.nan) if len(x) == 0 else tuple(np.percentile(x, [2.5, 97.5]))


# ──────────────────────────────────────────────────────────────
# Loading / alignment
# ──────────────────────────────────────────────────────────────
def load_arm(name, base, folds):
    frames = []
    for f in folds:
        p = Path(base) / f"fold_{f}" / "oof_predictions.csv"
        if not p.exists():
            sys.exit(f"[{name}] missing file: {p}")
        d = pd.read_csv(p)
        d["fold"] = f
        d["key"] = d["image"].apply(lambda s: Path(str(s)).name)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    if d["key"].duplicated().any():
        sys.exit(f"[{name}] duplicated image keys across folds (a case appears in >1 validation fold)")
    return d.set_index("key")


def align(arms):
    names = list(arms)
    common = set(arms[names[0]].index)
    for n in names[1:]:
        common &= set(arms[n].index)
    for n in names:
        extra = len(arms[n]) - len(common)
        if extra:
            print(f"WARNING [{n}]: {extra} case(s) not present in every arm - dropped")
    keys = sorted(common)
    out = {n: arms[n].loc[keys].copy() for n in names}
    ref = out[names[0]]
    for n in names[1:]:
        if not (out[n]["has_abandoned"].values == ref["has_abandoned"].values).all():
            sys.exit(f"[{n}] has_abandoned labels differ from the reference arm - not the same cases")
        if not (out[n]["fold"].values == ref["fold"].values).all():
            print(f"WARNING [{n}]: fold assignment differs from the reference arm - "
                  f"the comparison is NOT valid as a paired, same-split comparison")
    return out


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="C:/CIEDID_data/AbdnL/models",
                    help="same as --output_dir of finetuneCV_ablation.py")
    ap.add_argument("--inits", nargs="+", default=["busch_enc", "imagenet"],
                    help="arms to compare; the FIRST is the reference")
    ap.add_argument("--train_frac", type=float, default=0.25, help="fraction of the training fold used (stratified, nested across fractions)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt_select", action="store_true",
                    help="arms were trained with --ckpt_select (folder suffix _ckptsel)")
    ap.add_argument("--arm", action="append", default=None, metavar="NAME=DIR",
                    help="manual override: arm name and the directory that contains fold_N/. "
                         "Repeat; first = reference. When used, --root/--inits are ignored.")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--score", default="area_p0.8", help="PRIMARY score column (default area_p0.8)")
    ap.add_argument("--extra_scores", nargs="*", default=["max_prob_abdn", "soft_area_abdn"])
    ap.add_argument("--op_sens", type=float, default=0.975)
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--boot_seed", type=int, default=0, help="seed of the bootstrap resampling only")
    ap.add_argument("--out", default=None, help="default: <root>/comparison_<arms>_frac<f>_seed<s>")
    args = ap.parse_args()

    spec = {}
    if args.arm:
        for a in args.arm:
            if "=" not in a:
                sys.exit(f"--arm must be NAME=DIR, got: {a}")
            k, v = a.split("=", 1)
            spec[k] = v
    else:
        suffix = "_ckptsel" if args.ckpt_select else ""
        for init in args.inits:
            tag = f"{init}_frac{args.train_frac:g}_seed{args.seed}{suffix}"
            d = Path(args.root) / tag
            if not d.is_dir():
                avail = sorted(x.name for x in Path(args.root).glob("*_frac*_seed*")) if Path(args.root).is_dir() else []
                sys.exit(f"Folder not found: {d}\n"
                         f"Available run folders under {args.root}:\n  " +
                         ("\n  ".join(avail) if avail else "(none - check --root)"))
            spec[init] = str(d)
    if len(spec) < 2:
        sys.exit("need at least two arms")
    for k, v in spec.items():
        print(f"  arm {k:<12} <- {v}")
    if args.out is None:
        if args.arm:
            args.out = "comparison_stageI"
        else:
            name = (f"comparison_{'_vs_'.join(args.inits)}_frac{args.train_frac:g}_seed{args.seed}"
                    + ("_ckptsel" if args.ckpt_select else ""))
            args.out = str(Path(args.root) / name)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = align({k: load_arm(k, v, args.folds) for k, v in spec.items()})
    names = list(arms)
    ref = names[0]
    others = names[1:]
    y = arms[ref]["has_abandoned"].values.astype(int)
    folds = arms[ref]["fold"].values
    N = len(y)
    print(f"\nCases: {N}  (positives {int(y.sum())}, negatives {int((1 - y).sum())})  "
          f"folds: {sorted(int(f) for f in set(folds))}")
    print(f"Reference arm: {ref}   Others: {others}")
    print(f"Primary score: {args.score}   bootstrap resamples: {args.n_boot}")

    rows = []

    def add(metric, arm, est, lo=np.nan, hi=np.nan, p=np.nan, note=""):
        rows.append(dict(metric=metric, arm_or_comparison=arm, estimate=est,
                         ci_low=lo, ci_high=hi, p=p, note=note))

    rng = np.random.RandomState(args.boot_seed)
    idx = rng.randint(0, N, size=(args.n_boot, N))

    # ── PRIMARY + SENSITIVITY: pooled AUC ─────────────────────
    all_scores = [args.score] + [s for s in args.extra_scores if s != args.score]
    primary_p = []
    for sc in all_scores:
        tag = "PRIMARY" if sc == args.score else "sensitivity"
        print(f"\n{'=' * 70}\n{tag}: pooled case-level AUC of '{sc}'\n{'=' * 70}")
        S = {n: arms[n][sc].values.astype(float) for n in names}
        boot = {n: np.array([auc_fast(y[i], S[n][i]) for i in idx]) for n in names}
        for n in names:
            a = auc_fast(y, S[n])
            lo, hi = pct_ci(boot[n])
            print(f"  {n:<16} AUC {a:.3f}  (95% bootstrap CI {lo:.3f}-{hi:.3f})")
            add(f"AUC[{sc}]", n, a, lo, hi, note=tag)
        for o in others:
            a1, a2, d, se, z, p = delong_paired(y, S[ref], S[o])
            bd = boot[ref] - boot[o]
            lo, hi = pct_ci(bd)
            print(f"  {ref} - {o}: dAUC {d:+.3f}  DeLong 95% CI {d - 1.96 * se:+.3f} to {d + 1.96 * se:+.3f}, "
                  f"p={p:.4f};  bootstrap CI {lo:+.3f} to {hi:+.3f}")
            add(f"dAUC[{sc}]", f"{ref}-{o}", d, lo, hi, p, note=f"{tag}; DeLong p; CI=bootstrap")
            if sc == args.score:
                primary_p.append((o, p))
    if len(primary_p) > 1:
        adj = holm([p for _, p in primary_p])
        print("\n  Holm-adjusted primary p-values:")
        for (o, p), pa in zip(primary_p, adj):
            print(f"    {ref} vs {o}: p={p:.4f}  Holm p={pa:.4f}")
            add(f"dAUC[{args.score}] Holm", f"{ref}-{o}", np.nan, p=pa)

    # ── PER-FOLD AUC ──────────────────────────────────────────
    print(f"\n{'=' * 70}\nPer-fold AUC of '{args.score}'\n{'=' * 70}")
    pf = []
    for f in sorted(set(folds)):
        m = folds == f
        rec = {"fold": f, "n": int(m.sum()), "pos": int(y[m].sum())}
        for n in names:
            rec[n] = auc_fast(y[m], arms[n][args.score].values[m])
        pf.append(rec)
    pf = pd.DataFrame(pf)
    print(pf.round(3).to_string(index=False))
    for o in others:
        wins = int((pf[ref] > pf[o]).sum())
        ties = int((pf[ref] == pf[o]).sum())
        print(f"  {ref} higher than {o} in {wins}/{len(pf)} folds (ties: {ties}); "
              f"mean fold dAUC {np.nanmean(pf[ref] - pf[o]):+.3f}")
    pf.to_csv(out_dir / "per_fold_auc.csv", index=False)

    # ── OPERATING POINT ───────────────────────────────────────
    print(f"\n{'=' * 70}\nSpecificity at sensitivity >= {args.op_sens} (pooled scores, exploratory)\n{'=' * 70}")
    S = {n: arms[n][args.score].values.astype(float) for n in names}
    boot = {n: np.array([spec_at_sens(y[i], S[n][i], args.op_sens) for i in idx]) for n in names}
    for n in names:
        v = spec_at_sens(y, S[n], args.op_sens)
        lo, hi = pct_ci(boot[n])
        print(f"  {n:<16} specificity {v:.3f}  (95% bootstrap CI {lo:.3f}-{hi:.3f})")
        add(f"spec@sens{args.op_sens}", n, v, lo, hi)
    for o in others:
        bd = boot[ref] - boot[o]
        d = spec_at_sens(y, S[ref], args.op_sens) - spec_at_sens(y, S[o], args.op_sens)
        lo, hi = pct_ci(bd)
        print(f"  {ref} - {o}: {d:+.3f}  (bootstrap CI {lo:+.3f} to {hi:+.3f})")
        add(f"dSpec@sens{args.op_sens}", f"{ref}-{o}", d, lo, hi)

    # ── DICE ──────────────────────────────────────────────────
    print(f"\n{'=' * 70}\nPer-case Dice\n{'=' * 70}")
    for col, mask, label in [("dice_abdn", y == 1, "abandoned lead (positives only)"),
                             ("dice_gen", np.ones(N, bool), "generator (all cases)")]:
        if col not in arms[ref].columns:
            continue
        print(f"\n  {label}")
        vals = {n: arms[n][col].values.astype(float)[mask] for n in names}
        for n in names:
            v = vals[n][~np.isnan(vals[n])]
            print(f"    {n:<16} mean {v.mean():.3f}  median {np.median(v):.3f}  (n={len(v)})")
            add(f"mean {col}", n, float(v.mean()), note=f"median={np.median(v):.3f}")
        for o in others:
            d = vals[ref] - vals[o]
            d = d[~np.isnan(d)]
            bm = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(args.n_boot)])
            lo, hi = pct_ci(bm)
            try:
                p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            except ValueError:
                p = np.nan
            print(f"    {ref} - {o}: mean diff {d.mean():+.3f} (bootstrap CI {lo:+.3f} to {hi:+.3f}), "
                  f"Wilcoxon p={p:.4f}")
            add(f"d mean {col}", f"{ref}-{o}", float(d.mean()), lo, hi, p)

    # ── ROC plot ──────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        for n in names:
            fpr, tpr, _ = roc_curve(y, arms[n][args.score].values)
            ax.plot(fpr, tpr, label=f"{n} (AUC {auc_fast(y, arms[n][args.score].values):.3f})")
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_xlabel("1 - specificity")
        ax.set_ylabel("Sensitivity")
        ax.set_title(f"Pooled OOF ROC ({args.score})")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "roc_pooled.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # plotting is optional
        print(f"(ROC plot skipped: {e})")

    pd.DataFrame(rows).to_csv(out_dir / "comparison_summary.csv", index=False)
    merged = arms[ref][["image", "fold", "has_abandoned"]].copy()
    for n in names:
        for sc in all_scores + ["dice_abdn", "dice_gen"]:
            if sc in arms[n].columns:
                merged[f"{n}__{sc}"] = arms[n][sc].values
    merged.to_csv(out_dir / "pooled_oof_merged.csv")
    print(f"\nSaved: comparison_summary.csv, per_fold_auc.csv, pooled_oof_merged.csv, roc_pooled.png -> {out_dir}")
    print("\nNOTES: scores from different fold models are pooled; cases are treated as independent;"
          "\n       only the primary score is pre-specified - the rest is exploratory.")


if __name__ == "__main__":
    main()