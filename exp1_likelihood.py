"""Experiment 1 (report, Section 8): how close is the One-vs-Each likelihood to the softmax?

Likelihood and posterior contours for C = 3, y = 1, f_3 = 0; a priori confidence histogram;
correlation, mean absolute difference, KL divergence and argmax agreement over 50,000 random
logit vectors f_c ~ N(0, 1) (Table 1).

    python exp1_likelihood.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from gpclass import sigmoid  # noqa: E402


def experiment_1():
    print("\n" + "="*80)
    print("EXPERIMENT 1: SOFTMAX vs OVE — How tight is the approximation?")
    print("="*80)

    np.random.seed(42)

    # --- 1a: Likelihood and posterior contours ---
    gr = 3.0; n = 200
    f1 = np.linspace(-gr, gr, n); f2 = np.linspace(-gr, gr, n)
    F1, F2 = np.meshgrid(f1, f2)

    L_sm = np.exp(F1) / (np.exp(F1) + np.exp(F2) + 1.0)
    L_ove = sigmoid(F1 - F2) * sigmoid(F1)

    prior = np.exp(-0.5*(F1**2 + F2**2))
    P_sm = L_sm * prior; P_sm /= P_sm.sum()
    P_ove = L_ove * prior; P_ove /= P_ove.sum()

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    for ax, (L, t, cm) in zip(axes[0], [(L_sm, "Softmax", "Blues"),
                                          (L_ove, "OVE", "Oranges")]):
        cs = ax.contourf(F1, F2, L, levels=20, cmap=cm)
        ax.contour(F1, F2, L, levels=10, colors="k", linewidths=0.3, alpha=0.5)
        plt.colorbar(cs, ax=ax, shrink=0.8)
        ax.set_xlabel(r"$f_1$"); ax.set_ylabel(r"$f_2$")
        ax.set_title(f"Likelihood: {t}"); ax.set_aspect("equal")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.axvline(0, color="gray", lw=0.5, ls="--")

    for ax, (P, t, cm) in zip(axes[1], [(P_sm, "Softmax", "Blues"),
                                          (P_ove, "OVE", "Oranges")]):
        cs = ax.contourf(F1, F2, P, levels=20, cmap=cm)
        ax.contour(F1, F2, P, levels=10, colors="k", linewidths=0.3, alpha=0.5)
        plt.colorbar(cs, ax=ax, shrink=0.8)
        ax.set_xlabel(r"$f_1$"); ax.set_ylabel(r"$f_2$")
        ax.set_title(f"Posterior: {t}"); ax.set_aspect("equal")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.axvline(0, color="gray", lw=0.5, ls="--")

    fig.suptitle(r"$C=3$, $y=1$, $f_3=0$. Top: likelihood. Bottom: posterior with $\mathcal{N}(0,I)$ prior.", fontsize=13)
    plt.tight_layout()
    plt.savefig("figures/exp1_contours.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- 1b: Confidence histogram ---
    C = 3; ns = 50000
    fs = np.random.randn(ns, C)

    sm = np.exp(fs) / np.exp(fs).sum(1, keepdims=True)
    conf_sm = sm.max(1)

    lp = np.zeros_like(fs)
    for c in range(C):
        for cp in range(C):
            if cp != c: lp[:,c] += np.log(sigmoid(fs[:,c]-fs[:,cp])+1e-15)
    lp -= lp.max(1, keepdims=True)
    po = np.exp(lp); po /= po.sum(1, keepdims=True)
    conf_ove = po.max(1)

    fig2, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0.3, 1.0, 40)
    ax.hist(conf_sm, bins=bins, alpha=0.4, density=True, label="Softmax", color="steelblue")
    ax.hist(conf_ove, bins=bins, alpha=0.4, density=True, label="OVE", color="darkorange")
    for d, col in [(conf_sm,"steelblue"),(conf_ove,"darkorange")]:
        kde = gaussian_kde(d); x = np.linspace(0.3,1,200)
        ax.plot(x, kde(x), color=col, lw=2.5)
    ax.set_xlabel("Confidence (max class probability)")
    ax.set_ylabel("Density")
    ax.set_title(f"A priori confidence ($C={C}$, $f_c \\sim N(0,1)$)")
    ax.legend(); ax.set_xlim(0.3, 1.0)
    plt.tight_layout()
    plt.savefig("figures/exp1_confidence.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- 1c: Quantitative ---
    corr = np.corrcoef(sm.reshape(-1), po.reshape(-1))[0,1]
    mad = np.mean(np.abs(sm - po))
    kl = np.mean(np.sum(sm * np.log(sm/(po+1e-15)+1e-15), 1))
    agree = np.mean(sm.argmax(1) == po.argmax(1))

    print(f"\n  Quantitative (C={C}, {ns} samples):")
    print(f"    Correlation:          {corr:.4f}")
    print(f"    Mean abs. difference: {mad:.4f}")
    print(f"    KL(Softmax || OVE):   {kl:.4f}")
    print(f"    Argmax agreement:     {agree:.4f}")
    print("\n  Saved: figures/exp1_contours.png, figures/exp1_confidence.png")


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    experiment_1()
