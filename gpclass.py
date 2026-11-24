"""Multi-class Gaussian-process classification with Polya-Gamma augmentation.

Two Gibbs samplers for C classes with independent GP priors f^c ~ GP(0, k), c = 1..C:

    run_polson_gp  exact softmax likelihood; one class at a time, conditioning on the others
                   (Holmes and Held, 2006; Polson, Scott and Windle, 2013)
    run_ove_gp     One-vs-Each likelihood (Titsias, 2016) read as a pairwise composite
                   likelihood; all of f in one block (Snell and Zemel, 2021)

Both return the predictive class probabilities at the test inputs, averaged over `draws`
independent chains (the last state of each chain after `steps` Gibbs iterations), with the
latent functions at the test inputs set to the GP conditional mean given the training values.

The Polya-Gamma draws use the infinite-sum representation of PG(1, c) truncated at 20 terms.
"""

import numpy as np


def rbf_kernel(x1, x2=None, ls=1.0):
    if x2 is None: x2 = x1
    x1, x2 = np.atleast_2d(x1), np.atleast_2d(x2)
    sq = np.sum(x1**2,1)[:,None] + np.sum(x2**2,1)[None,:] - 2*x1@x2.T
    return np.exp(-0.5 * sq / ls**2)

def safe_chol(A, jit=1e-6):
    for i in range(6):
        try: return np.linalg.cholesky(A + jit*(10**i)*np.eye(A.shape[0]))
        except np.linalg.LinAlgError: continue
    raise np.linalg.LinAlgError

def solve_chol(L, b):
    return np.linalg.solve(L.T, np.linalg.solve(L, b))

def sample_mvn(mu, L):
    return mu + L @ np.random.randn(len(mu))

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sample_pg(c):
    c = np.atleast_1d(np.abs(np.asarray(c, dtype=np.float64)))
    r = np.zeros_like(c)
    for k in range(1, 21):
        g = np.random.gamma(1.0, 1.0, size=c.shape)
        r += g / ((k-0.5)**2 + c**2/(4*np.pi**2))
    return r / (2*np.pi**2)

def softmax_probs(f):
    f = f - f.max(axis=1, keepdims=True)
    e = np.exp(f); return e / e.sum(axis=1, keepdims=True)

def ove_probs(f):
    C = f.shape[1]; lp = np.zeros_like(f)
    for c in range(C):
        for cp in range(C):
            if cp != c: lp[:,c] += np.log(sigmoid(f[:,c]-f[:,cp])+1e-15)
    lp -= lp.max(1, keepdims=True)
    p = np.exp(lp); return p / (p.sum(1, keepdims=True)+1e-15)

def accuracy_fn(p, y): return np.mean(p.argmax(1)==y)

def brier_fn(p, y, C): return np.mean(np.sum((p-np.eye(C)[y])**2,1))

def ece_fn(p, y, nb=10):
    c, pr = p.max(1), p.argmax(1); co = (pr==y).astype(float)
    b = np.linspace(0,1,nb+1); e=0
    for i in range(nb):
        m = (c>b[i])&(c<=b[i+1])
        if m.sum()>0: e += m.sum()*np.abs(co[m].mean()-c[m].mean())
    return e/len(y)

def blkdiag(Kb, C):
    N = Kb.shape[0]; K = np.zeros((C*N, C*N))
    for c in range(C): K[c*N:(c+1)*N, c*N:(c+1)*N] = Kb
    return K

def build_A(y, C, N):
    M = N*(C-1); A = np.zeros((M, N*C)); row = 0
    for i in range(N):
        for cp in range(C):
            if cp != y[i]:
                A[row, y[i]*N+i] = 1.0; A[row, cp*N+i] = -1.0; row += 1
    return A

def autocorrelation(x, max_lag=30):
    x = x - x.mean(); n = len(x)
    var = np.sum(x**2) / n
    if var < 1e-15: return np.zeros(min(max_lag, n))
    return np.array([np.sum(x[:n-l]*x[l:])/n/var for l in range(min(max_lag, n))])

def effective_sample_size(traces):
    n, dim = traces.shape; ess = np.zeros(dim)
    for d in range(dim):
        acf = autocorrelation(traces[:,d], min(50, n//2))
        total = 0.0
        for k in range(1, len(acf)):
            if acf[k] < 0: break
            total += acf[k]
        ess[d] = n / (1.0 + 2.0*total)
    return ess


def run_ove_gp(Xtr, ytr, Xte, ls, C, steps=20, draws=5, return_trace=False):
    """GP + OVE likelihood, block Gibbs update."""
    N = Xtr.shape[0]
    Kb = rbf_kernel(Xtr, ls=ls) + 1e-6*np.eye(N)
    K = blkdiag(Kb, C); A = build_A(ytr, C, N)
    AK = A@K; AKA_T = AK@A.T
    mu = np.zeros(C*N); LK = safe_chol(K); Kim = solve_chol(LK, mu)
    Ksb = rbf_kernel(Xte, Xtr, ls=ls); LKb = safe_chol(Kb)
    all_p = []; trace = []
    for d in range(draws):
        f = sample_mvn(mu, LK)
        for _ in range(steps):
            psi = A@f; om = sample_pg(psi)
            kap = 0.5*np.ones_like(om); Oi = np.diag(1.0/om)
            Li = safe_chol(AKA_T + Oi); sol = solve_chol(Li, AK)
            Sig = K - AK.T@sol; mut = Sig@(Kim + A.T@kap)
            f = sample_mvn(mut, safe_chol(Sig))
            if return_trace: trace.append(f.copy())
        fp = np.zeros((Xte.shape[0], C))
        for c in range(C): fp[:,c] = Ksb @ solve_chol(LKb, f[c*N:(c+1)*N])
        all_p.append(ove_probs(fp))
    if return_trace:
        return np.mean(all_p, 0), np.array(trace)
    return np.mean(all_p, 0)

def run_polson_gp(Xtr, ytr, Xte, ls, C, steps=20, draws=5, return_trace=False):
    """GP + exact softmax, Polson-style sequential Gibbs."""
    N = Xtr.shape[0]
    Kb = rbf_kernel(Xtr, ls=ls) + 1e-6*np.eye(N)
    LK = safe_chol(Kb); Kinv = solve_chol(LK, np.eye(N))
    Ksb = rbf_kernel(Xte, Xtr, ls=ls)
    all_p = []; trace = []
    for d in range(draws):
        f = np.zeros((C, N))
        for c in range(C): f[c] = sample_mvn(np.zeros(N), LK)
        for _ in range(steps):
            for j in range(C):
                oth = np.delete(f, j, 0); mx = oth.max(0)
                Cij = mx + np.log(np.exp(oth - mx).sum(0) + 1e-15)
                eta = f[j] - Cij
                yj = (ytr==j).astype(float); kap = yj - 0.5
                om = sample_pg(eta); Om = np.diag(om)
                Lpp = safe_chol(Kinv + Om)
                St = solve_chol(Lpp, np.eye(N))
                mt = St @ (kap + Om@Cij)
                f[j] = sample_mvn(mt, safe_chol(St))
            if return_trace: trace.append(f.reshape(-1).copy())
        fp = np.zeros((Xte.shape[0], C))
        for c in range(C): fp[:,c] = Ksb @ solve_chol(LK, f[c])
        all_p.append(softmax_probs(fp))
    if return_trace:
        return np.mean(all_p, 0), np.array(trace)
    return np.mean(all_p, 0)
