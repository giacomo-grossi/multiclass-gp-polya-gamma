"""Experiment 2 (report, Section 9): exact softmax (Polson-GP) against One-vs-Each (OVE-GP) on
Palmer Penguins, two standardized bill measurements, three species.

    experiment_2   accuracy, Brier score and ECE for 3, 5, 10, 20 training points per class,
                   averaged over 5 random splits (Table 2)
    experiment_2b  mixing of the two samplers on one split: effective sample size and trace plot

    python exp2_penguins.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gpclass import (accuracy_fn, brier_fn, ece_fn, effective_sample_size,  # noqa: E402
                     run_ove_gp, run_polson_gp)


def experiment_2():
    print("\n" + "="*80)
    print("EXPERIMENT 2: SOFTMAX vs OVE on Palmer Penguins (2D)")
    print("="*80)

    import palmerpenguins
    from sklearn.preprocessing import StandardScaler

    df = palmerpenguins.load_penguins().dropna()
    sp = {"Adelie":0, "Gentoo":1, "Chinstrap":2}
    y_all = df["species"].map(sp).values
    X_all = StandardScaler().fit_transform(df[["bill_length_mm","bill_depth_mm"]].values)
    C = 3; ls = 1.0; n_splits = 5; STEPS = 20; DRAWS = 5
    train_sizes = [3, 5, 10, 20]
    results_sm = {"acc":[], "bri":[], "ece":[]}
    results_ove = {"acc":[], "bri":[], "ece":[]}

    print(f"\n  Penguins 2D (bill length, bill depth), C={C}")
    print(f"  RBF kernel ls={ls}, {STEPS} Gibbs steps, {DRAWS} chains, {n_splits} splits\n")

    print(f"  {'N/cl':>5s} | {'Acc(SM)':>7s} {'Acc(OVE)':>8s} | {'Bri(SM)':>7s} {'Bri(OVE)':>8s} | {'ECE(SM)':>7s} {'ECE(OVE)':>8s}")
    print(f"  {'─'*5} | {'─'*7} {'─'*8} | {'─'*7} {'─'*8} | {'─'*7} {'─'*8}")

    for npc in train_sizes:
        met = {m: {k: [] for k in ["SM","OVE"]} for m in ["acc","bri","ece"]}
        for s in range(n_splits):
            np.random.seed(s*100+npc)
            itr, ite = [], []
            for c in range(C):
                ci = np.where(y_all==c)[0]; p = np.random.permutation(ci)
                na = min(npc, len(ci)-5); itr.extend(p[:na]); ite.extend(p[na:])
            Xtr, ytr = X_all[itr], y_all[itr]
            Xte, yte = X_all[ite], y_all[ite]

            try:
                p_sm = run_polson_gp(Xtr, ytr, Xte, ls, C, STEPS, DRAWS)
                met["acc"]["SM"].append(accuracy_fn(p_sm, yte))
                met["bri"]["SM"].append(brier_fn(p_sm, yte, C))
                met["ece"]["SM"].append(ece_fn(p_sm, yte))
            except np.linalg.LinAlgError:
                for m in met: met[m]["SM"].append(np.nan)

            try:
                p_ove = run_ove_gp(Xtr, ytr, Xte, ls, C, STEPS, DRAWS)
                met["acc"]["OVE"].append(accuracy_fn(p_ove, yte))
                met["bri"]["OVE"].append(brier_fn(p_ove, yte, C))
                met["ece"]["OVE"].append(ece_fn(p_ove, yte))
            except np.linalg.LinAlgError:
                for m in met: met[m]["OVE"].append(np.nan)

        v = {mk: {k: np.nanmean(met[mk][k]) for k in ["SM","OVE"]} for mk in met}
        results_sm["acc"].append(v["acc"]["SM"]); results_sm["bri"].append(v["bri"]["SM"]); results_sm["ece"].append(v["ece"]["SM"])
        results_ove["acc"].append(v["acc"]["OVE"]); results_ove["bri"].append(v["bri"]["OVE"]); results_ove["ece"].append(v["ece"]["OVE"])
        print(f"  {npc:>5d} | {v['acc']['SM']:>7.3f} {v['acc']['OVE']:>8.3f} | "
              f"{v['bri']['SM']:>7.3f} {v['bri']['OVE']:>8.3f} | "
              f"{v['ece']['SM']:>7.3f} {v['ece']['OVE']:>8.3f}")

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Accuracy
    axes[0].plot(train_sizes, results_sm["acc"], 'o-', color="steelblue", lw=2, markersize=8, label="Softmax (Polson)")
    axes[0].plot(train_sizes, results_ove["acc"], 's-', color="darkorange", lw=2, markersize=8, label="OVE")
    axes[0].set_xlabel("Training points per class"); axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].set_ylim(0.75, 1.0)
    axes[0].grid(alpha=0.3)

    # Brier
    axes[1].plot(train_sizes, results_sm["bri"], 'o-', color="steelblue", lw=2, markersize=8, label="Softmax (Polson)")
    axes[1].plot(train_sizes, results_ove["bri"], 's-', color="darkorange", lw=2, markersize=8, label="OVE")
    axes[1].set_xlabel("Training points per class"); axes[1].set_ylabel("Brier Score")
    axes[1].set_title("Brier Score "); axes[1].legend()
    axes[1].grid(alpha=0.3)

    # ECE
    axes[2].plot(train_sizes, results_sm["ece"], 'o-', color="steelblue", lw=2, markersize=8, label="Softmax (Polson)")
    axes[2].plot(train_sizes, results_ove["ece"], 's-', color="darkorange", lw=2, markersize=8, label="OVE")
    axes[2].set_xlabel("Training points per class"); axes[2].set_ylabel("ECE")
    axes[2].set_title("Expected Calibration Error"); axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.suptitle("Softmax vs OVE on Palmer Penguins (2D)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("figures/exp2_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved: figures/exp2_comparison.png")


def experiment_2b():
    print("\n" + "="*80)
    print("EXPERIMENT 2b: MCMC Diagnostics on Palmer Penguins")
    print("="*80)

    import palmerpenguins
    from sklearn.preprocessing import StandardScaler

    df = palmerpenguins.load_penguins().dropna()
    sp = {"Adelie":0, "Gentoo":1, "Chinstrap":2}
    y_all = df["species"].map(sp).values
    X_all = StandardScaler().fit_transform(df[["bill_length_mm","bill_depth_mm"]].values)
    C = 3; ls = 1.0

    STEPS = 100; BURNIN = 30; DRAWS = 1
    npc = 20

    np.random.seed(42)
    itr, ite = [], []
    for c in range(C):
        ci = np.where(y_all==c)[0]; p = np.random.permutation(ci)
        na = min(npc, len(ci)-5); itr.extend(p[:na]); ite.extend(p[na:])
    Xtr, ytr = X_all[itr], y_all[itr]
    Xte, yte = X_all[ite], y_all[ite]
    N = len(ytr)

    print(f"\n  Penguins 2D, C={C}, N_train={N}, N_test={len(yte)}")
    print(f"  Single chain, {STEPS} steps, {BURNIN} burn-in\n")

    # --- Run both with trace ---
    p_sm, trace_sm = run_polson_gp(Xtr, ytr, Xte, ls, C, STEPS, DRAWS, return_trace=True)
    p_ove, trace_ove = run_ove_gp(Xtr, ytr, Xte, ls, C, STEPS, DRAWS, return_trace=True)

    # --- ESS (post burn-in) ---
    tr_sm = trace_sm[BURNIN:]
    tr_ove = trace_ove[BURNIN:]
    n_post = STEPS - BURNIN

    ess_sm = effective_sample_size(tr_sm)
    ess_ove = effective_sample_size(tr_ove)

    print(f"  {'Metric':<25s} {'Polson-GP':>12s} {'OVE-GP':>12s}")
    print(f"  {'─'*25} {'─'*12} {'─'*12}")
    print(f"  {'Mean ESS':<25s} {ess_sm.mean():>12.1f} {ess_ove.mean():>12.1f}")
    print(f"  {'ESS / sample':<25s} {ess_sm.mean()/n_post:>12.3f} {ess_ove.mean()/n_post:>12.3f}")

    # --- Traceplot ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(tr_sm[:, 0], color="steelblue", lw=0.8)
    axes[0].set_ylabel(r"$f^1_1$"); axes[0].set_title("Polson-GP (sequential)")
    axes[0].axhline(tr_sm[:,0].mean(), color="red", ls="--", lw=1)
    axes[1].plot(tr_ove[:, 0], color="darkorange", lw=0.8)
    axes[1].set_ylabel(r"$f^1_1$"); axes[1].set_title("OVE-GP (block)")
    axes[1].axhline(tr_ove[:,0].mean(), color="red", ls="--", lw=1)
    axes[1].set_xlabel("Iteration (post burn-in)")
    plt.tight_layout()
    plt.savefig("figures/exp2b_traceplot.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n  Saved: figures/exp2b_traceplot.png")


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    experiment_2()
    experiment_2b()
