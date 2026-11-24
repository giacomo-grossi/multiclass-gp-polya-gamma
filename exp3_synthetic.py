"""Experiment 3: does the advantage of OVE-GP on Palmer Penguins survive longer chains on data
simulated from a softmax model?

Inputs are drawn around C = 3 class centres; each class gets a GP draw (RBF kernel, lengthscale
1) plus a bump 2 exp(-|x - centre_c|^2 / (2 spread^2)) around its centre; labels are drawn from
the softmax of these logits. The fitted models have zero-mean GP priors, so the data come from
a softmax likelihood but not exactly from the fitted prior. Three degrees of class overlap,
200 Gibbs steps, 3 chains, 10 replications.

    python exp3_synthetic.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gpclass import (accuracy_fn, brier_fn, ece_fn, rbf_kernel, run_ove_gp,  # noqa: E402
                     run_polson_gp, safe_chol, sample_mvn, softmax_probs)


def generate_from_softmax_gp(N_total, C, centers, spread, ls, seed=42):
    """
    Generate data from the TRUE softmax GP model:
    1. Sample X uniformly
    2. Sample f^c(X) from GP prior for each class
    3. Compute softmax probabilities
    4. Sample labels from softmax
    """
    np.random.seed(seed)

    # Generate inputs around class centers
    npc = N_total // C
    X = np.vstack([centers[c] + spread * np.random.randn(npc, 2) for c in range(C)])

    # Sample GP functions for each class
    Kb = rbf_kernel(X, ls=ls) + 1e-6 * np.eye(len(X))
    LK = safe_chol(Kb)
    f_true = np.zeros((C, len(X)))
    for c in range(C):
        f_true[c] = sample_mvn(np.zeros(len(X)), LK)
        # Shift mean so correct class has higher logit near its center
        for i in range(len(X)):
            dist_to_c = np.linalg.norm(X[i] - centers[c])
            f_true[c, i] += 2.0 * np.exp(-dist_to_c**2 / (2 * spread**2))

    # Compute true softmax probabilities and sample labels
    f_mat = f_true.T  # (N, C)
    p_true = softmax_probs(f_mat)
    y = np.array([np.random.choice(C, p=p_true[i]) for i in range(len(X))])

    return X, y, p_true


def experiment_3():
    print("=" * 80)
    print("  ROBUST SIMULATION: Softmax GP vs OVE GP")
    print("  Data generated from TRUE softmax GP")
    print("=" * 80)

    C = 3
    STEPS = 200  # generous iterations
    DRAWS = 3
    N_REPS = 10  # replications
    ls = 1.0

    # Two scenarios
    scenarios = [
        {
            "name": "Well-separated (spread=0.3)",
            "centers": np.array([[0, 0], [3, 0], [1.5, 2.6]]),
            "spread": 0.3,
            "npc_train": 10,
            "npc_test": 30,
        },
        {
            "name": "Overlapping (spread=1.0)",
            "centers": np.array([[0, 0], [1.5, 0], [0.75, 1.3]]),
            "spread": 1.0,
            "npc_train": 10,
            "npc_test": 30,
        },
        {
            "name": "Heavily overlapping (spread=1.5)",
            "centers": np.array([[0, 0], [1.0, 0], [0.5, 0.87]]),
            "spread": 1.5,
            "npc_train": 10,
            "npc_test": 30,
        },
    ]

    all_results = {}

    for sc in scenarios:
        print(f"\n{'─'*80}")
        print(f"  Scenario: {sc['name']}")
        print(f"  C={C}, {sc['npc_train']} train/class, {sc['npc_test']} test/class")
        print(f"  {STEPS} Gibbs steps, {DRAWS} chains, {N_REPS} replications")
        print(f"{'─'*80}")

        met = {"SM": {"acc":[], "bri":[], "ece":[]},
               "OVE": {"acc":[], "bri":[], "ece":[]}}

        for rep in range(N_REPS):
            # Generate fresh data each replication
            N_total = (sc["npc_train"] + sc["npc_test"]) * C
            X, y, _ = generate_from_softmax_gp(
                N_total, C, sc["centers"], sc["spread"], ls, seed=rep*1000+7)

            # Split train/test
            itr, ite = [], []
            for c in range(C):
                ci = np.where(y == c)[0]
                np.random.seed(rep*100 + c)
                perm = np.random.permutation(ci)
                na = min(sc["npc_train"], len(perm) - 5)
                itr.extend(perm[:na])
                ite.extend(perm[na:na + sc["npc_test"]])
            Xtr, ytr = X[itr], y[itr]
            Xte, yte = X[ite], y[ite]

            if len(Xte) == 0 or len(Xtr) == 0:
                continue

            try:
                p_sm = run_polson_gp(Xtr, ytr, Xte, ls, C, STEPS, DRAWS)
                met["SM"]["acc"].append(accuracy_fn(p_sm, yte))
                met["SM"]["bri"].append(brier_fn(p_sm, yte, C))
                met["SM"]["ece"].append(ece_fn(p_sm, yte))
            except np.linalg.LinAlgError:
                met["SM"]["acc"].append(np.nan)
                met["SM"]["bri"].append(np.nan)
                met["SM"]["ece"].append(np.nan)

            try:
                p_ove = run_ove_gp(Xtr, ytr, Xte, ls, C, STEPS, DRAWS)
                met["OVE"]["acc"].append(accuracy_fn(p_ove, yte))
                met["OVE"]["bri"].append(brier_fn(p_ove, yte, C))
                met["OVE"]["ece"].append(ece_fn(p_ove, yte))
            except np.linalg.LinAlgError:
                met["OVE"]["acc"].append(np.nan)
                met["OVE"]["bri"].append(np.nan)
                met["OVE"]["ece"].append(np.nan)

            if (rep + 1) % 5 == 0:
                print(f"    Rep {rep+1}/{N_REPS} done")

        # Compute means and stds
        r = {}
        for method in ["SM", "OVE"]:
            r[method] = {}
            for m in ["acc", "bri", "ece"]:
                vals = [v for v in met[method][m] if not np.isnan(v)]
                r[method][m] = (np.mean(vals), np.std(vals) / np.sqrt(len(vals)))

        all_results[sc["name"]] = r

        print(f"\n  {'Metric':<15s} {'Polson-GP':>20s} {'OVE-GP':>20s}")
        print(f"  {'─'*15} {'─'*20} {'─'*20}")
        for m, label in [("acc","Accuracy ↑"),("bri","Brier ↓"),("ece","ECE ↓")]:
            sm_m, sm_se = r["SM"][m]
            ove_m, ove_se = r["OVE"][m]
            print(f"  {label:<15s} {sm_m:>8.3f} ± {sm_se:.3f}    {ove_m:>8.3f} ± {ove_se:.3f}")

    # =========================================================================
    # SUMMARY PLOT
    # =========================================================================

    scenario_names = [s["name"].split("(")[0].strip() for s in scenarios]
    n_sc = len(scenarios)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (metric, title) in zip(axes, [("acc","Accuracy ↑"),("bri","Brier ↓"),("ece","ECE ↓")]):
        x = np.arange(n_sc)
        w = 0.35
        sm_vals = [all_results[s["name"]]["SM"][metric][0] for s in scenarios]
        ove_vals = [all_results[s["name"]]["OVE"][metric][0] for s in scenarios]
        sm_errs = [all_results[s["name"]]["SM"][metric][1] for s in scenarios]
        ove_errs = [all_results[s["name"]]["OVE"][metric][1] for s in scenarios]

        ax.bar(x - w/2, sm_vals, w, yerr=sm_errs, label="Softmax (Polson)",
               color="steelblue", alpha=0.8, capsize=4)
        ax.bar(x + w/2, ove_vals, w, yerr=ove_errs, label="OVE",
               color="darkorange", alpha=0.8, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_names, fontsize=9)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

    plt.suptitle(f"Softmax vs OVE on synthetic data (C={C}, {STEPS} Gibbs steps, {N_REPS} reps)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("figures/simulation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n{'='*80}")
    print("  SIMULATION COMPLETE")
    print(f"{'='*80}")
    print("\n  Saved: figures/simulation_comparison.png")
    print(f"\n  Key question: with {STEPS} Gibbs steps and data generated from")
    print("  the true softmax, do the two methods converge to the same results?")


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    experiment_3()
