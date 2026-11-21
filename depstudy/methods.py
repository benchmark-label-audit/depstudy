"""Truth-inference and label-error scoring methods.

Input for every method: an answer matrix X of shape (n_items, n_models) with integer codes, -1 = missing,
and a gold vector g of shape (n_items,) with integer codes. Codes index a per-item value set; for
multiple-choice items the codes are 0..K-1, for free-form answers they are per-item value ids (the gold
value is always encoded, possibly unobserved among the answers).

Every method returns a score per item: higher = more likely that the shipped gold label is wrong.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-9


def _candidates(X: np.ndarray, g: np.ndarray, K: int | None = None) -> np.ndarray:
    """Boolean (n_items, V) mask of candidate values, V = max code + 1: the K options for multiple-choice items,
    otherwise the values this panel gave plus the key (free-form items). Free-form codes are numbered over the whole
    population, so a panel's codes are sparse; values no panel source gave are not candidates."""
    n_i = X.shape[0]
    V = int(max(X.max(), g.max())) + 1
    if K is not None:
        return np.broadcast_to(np.arange(V)[None, :] < int(K), (n_i, V)).copy()
    P = np.zeros((n_i, V), bool)
    r, c = np.nonzero(X >= 0)
    P[r, X[r, c]] = True
    P[np.arange(n_i), g] = True
    return P


def _n_values(X: np.ndarray, g: np.ndarray, K: int | None = None) -> np.ndarray:
    """Number of candidate values per item (see _candidates)."""
    if K is not None:
        return np.full(X.shape[0], int(K))
    return _candidates(X, g, None).sum(1)


def logodds_against_key(lp, g, valid=None):
    """log sum_{v != g} exp(lp[v]) - lp[g]: the log posterior odds that the true value differs from the key,
    computed in log space. Ranks items exactly as 1 - posterior(g) does in exact arithmetic, without the
    saturation of double-precision posteriors when thousands of sources multiply their likelihoods."""
    lp = np.asarray(lp, np.float64)
    r = np.arange(len(g))
    key = lp[r, g].copy()
    other = lp.copy()
    other[r, g] = -np.inf
    if valid is not None:
        other[~valid] = -np.inf
    m = other.max(1)
    fin = np.isfinite(m)
    lse = np.full(len(g), -np.inf)
    lse[fin] = m[fin] + np.log(np.exp(other[fin] - m[fin, None]).sum(1))
    return lse - key


def onehot_counts(X, g, w=None):
    """C[i, v] = sum of weights of models answering value v on item i (dense up to max code)."""
    n_i, n_m = X.shape
    V = int(max(X.max(), g.max())) + 1
    C = np.zeros((n_i, V))
    w = np.ones(n_m) if w is None else w
    for j in range(n_m):
        ok = X[:, j] >= 0
        np.add.at(C, (np.nonzero(ok)[0], X[ok, j]), w[j])
    return C


# --------------------------------------------------------------------------------------------- MV
def majority_vote(X, g):
    C = onehot_counts(X, g)
    tot = C.sum(1) + EPS
    return 1.0 - C[np.arange(len(g)), g] / tot


# ------------------------------------------------------------------------------- Dawid-Skene
def dawid_skene_onecoin(X, g, iters=50, prior_acc=0.7, K=None, logodds=False, n_false=None):
    """One-coin DS: model j answers the truth with prob p_j, otherwise uniformly among the other values.

    Candidate truths are the K options (multiple choice) or the observed values plus the key (free form). The
    number of false values in the likelihood is K - 1, or |V_i| - 1 per item, or a fixed n_false when given.
    The E-step is vectorised: log P(v) = sum_j log((1 - p_j) / n_f) + sum_{j: X_ij = v} [log p_j - log((1 - p_j) / n_f)];
    dawid_skene_onecoin_loop is the per-model reference implementation."""
    n_i, n_m = X.shape
    nv = _n_values(X, g, K).astype(float)
    nf = np.maximum(nv - 1, 1) if n_false is None else np.full(n_i, float(n_false))
    V = int(max(X.max(), g.max())) + 1
    p = np.full(n_m, prior_acc)
    obs = X >= 0
    Xc = np.where(obs, X, 0)
    rows = np.arange(n_i)[:, None]
    flat = (rows * V + Xc)[obs]
    n_obs = obs.sum(0)
    valid = _candidates(X, g, K)
    lnf = np.log(nf)[:, None]
    post = None
    for _ in range(iters):
        lw = np.log((1 - p)[None, :] / np.exp(lnf) + EPS)              # (n_i, n_m)
        lr = np.log(p + EPS)[None, :]
        base = np.where(obs, lw, 0.0).sum(1)
        delta = np.broadcast_to(lr, lw.shape)[obs] - lw[obs]
        lp = np.bincount(flat, weights=delta, minlength=n_i * V).reshape(n_i, V) + base[:, None]
        lp[~valid] = -np.inf
        lp -= lp.max(1, keepdims=True)
        post = np.exp(lp); post /= post.sum(1, keepdims=True)
        q = np.where(obs, post[rows, Xc], 0.0)
        p = np.clip((q.sum(0) + 1) / (n_obs + 2), 0.01, 0.99)
    if logodds:
        return logodds_against_key(lp, g, valid), post, p
    return 1.0 - post[np.arange(n_i), g], post, p


def dawid_skene_onecoin_loop(X, g, iters=50, prior_acc=0.7, K=None, logodds=False, n_false=None):
    """One-coin DS: model j answers the truth with prob p_j, otherwise uniformly among the other values.

    Candidate truths are the K options (multiple choice) or the observed values plus the key (free form). The
    number of false values in the likelihood is K - 1, or |V_i| - 1 per item, or a fixed n_false when given."""
    n_i, n_m = X.shape
    nv = _n_values(X, g, K).astype(float)
    nf = np.maximum(nv - 1, 1) if n_false is None else np.full(n_i, float(n_false))
    V = int(max(X.max(), g.max())) + 1
    p = np.full(n_m, prior_acc)
    obs = X >= 0
    post = None
    for _ in range(iters):
        # E-step: log posterior over values v for each item
        lp = np.zeros((n_i, V))
        valid = _candidates(X, g, K)
        for j in range(n_m):
            o = obs[:, j]
            lw = np.log((1 - p[j]) / nf[o] + EPS)
            lr = np.log(p[j] + EPS)
            lp[o] += lw[:, None]
            lp[np.nonzero(o)[0], X[o, j]] += lr - lw
        lp[~valid] = -np.inf
        lp -= lp.max(1, keepdims=True)
        post = np.exp(lp); post /= post.sum(1, keepdims=True)
        # M-step
        for j in range(n_m):
            o = obs[:, j]
            p[j] = np.clip((post[np.nonzero(o)[0], X[o, j]].sum() + 1) / (o.sum() + 2), 0.01, 0.99)
    if logodds:
        return logodds_against_key(lp, g, valid), post, p
    return 1.0 - post[np.arange(n_i), g], post, p


def dawid_skene_full(X, g, K, iters=50, smooth=1.0, logodds=False):
    """Full-confusion DS for K-way multiple choice (codes 0..K-1)."""
    n_i, n_m = X.shape
    obs = X >= 0
    # init posterior with majority vote
    C = np.zeros((n_i, K))
    for j in range(n_m):
        o = obs[:, j]; np.add.at(C, (np.nonzero(o)[0], X[o, j]), 1)
    post = (C + 0.01) / (C + 0.01).sum(1, keepdims=True)
    for _ in range(iters):
        pi_class = post.mean(0) + EPS
        conf = np.zeros((n_m, K, K))
        for j in range(n_m):
            o = obs[:, j]
            Y = np.zeros((o.sum(), K)); Y[np.arange(o.sum()), X[o, j]] = 1
            conf[j] = post[o].T @ Y + smooth * (np.eye(K) * 2 + 1) / K
            conf[j] /= conf[j].sum(1, keepdims=True)
        lp = np.tile(np.log(pi_class), (n_i, 1))
        lc = np.log(conf + EPS)
        for j in range(n_m):
            o = obs[:, j]
            lp[o] += lc[j][:, X[o, j]].T
        lp -= lp.max(1, keepdims=True)
        post = np.exp(lp); post /= post.sum(1, keepdims=True)
    if logodds:
        return logodds_against_key(lp, g), post
    return 1.0 - post[np.arange(n_i), g], post


# ------------------------------------------------------------------------------------- EBCC
def _logsumexp(a, axis):
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m, axis) + np.log(np.exp(a - m).sum(axis))


def ebcc(X, g, K, M=10, a_pi=0.1, alpha=1.0, a_v=4.0, b_v=1.0, iters=500, tol=1e-3, seed=1234, logodds=True):
    """EBCC (Li, Rubinstein, Cohn, ICML 2019): each true class k is a mixture of M subtypes, and every worker has one
    confusion matrix per (class, subtype), so workers that err alike on a subtype of items are modelled as
    correlated. Mean-field variational Bayes with the authors' defaults (M = 10, a_pi = 0.1, alpha = 1, a_v = 4,
    b_v = 1, at most 500 iterations, stop when the class posteriors move by less than 1e-3), following their
    reference implementation (github.com/yuan-li/truth-inference-at-scale, code/ebcc.py). Two changes are numerical
    only: the E-step is normalised in log space, and items are ranked by log posterior odds against the key
    (Eq. 1), since with thousands of workers the exponentiated E-step of the reference code under- or overflows."""
    import scipy.sparse as ssp
    from scipy.special import digamma
    n_i, n_m = X.shape
    obs = X >= 0
    r, c = np.nonzero(obs)
    v = X[r, c]
    Y = [ssp.csr_matrix((np.ones(int((v == l).sum())), (r[v == l], c[v == l])), shape=(n_i, n_m)) for l in range(K)]
    Yt = [y.T.tocsr() for y in Y]
    beta = np.eye(K) * (a_v - b_v) + b_v                               # (k, l) prior of each confusion row
    z = np.stack([np.asarray(y.sum(1)).ravel() for y in Y], 1) + EPS   # majority-vote initialisation
    z /= z.sum(1, keepdims=True)
    rng = np.random.default_rng(seed)
    zg = rng.dirichlet(np.ones(M), size=(n_i, K)) * z[:, :, None]      # (i, k, m)
    L = None
    for _ in range(iters):
        eta = a_pi / M + zg.sum(0)                                      # (k, m)
        nu = alpha + z.sum(0)                                           # (k,)
        flat = zg.reshape(n_i, K * M)
        mu = np.empty((n_m, K, M, K))
        for l in range(K):
            mu[:, :, :, l] = (Yt[l] @ flat).reshape(n_m, K, M) + beta[None, :, None, l]
        Elog_pi = digamma(eta) - digamma(eta.sum(1, keepdims=True))
        Elog_tau = digamma(nu) - digamma(nu.sum())
        Elog_v = digamma(mu) - digamma(mu.sum(-1, keepdims=True))
        L = np.broadcast_to(Elog_pi[None] + Elog_tau[None, :, None], (n_i, K, M)).copy()
        for l in range(K):
            L += (Y[l] @ Elog_v[:, :, :, l].reshape(n_m, K * M)).reshape(n_i, K, M)
        mx = L.max((1, 2), keepdims=True)
        zg = np.exp(L - mx); zg /= zg.sum((1, 2), keepdims=True)
        z_old, z = z, zg.sum(2)
        if np.allclose(z_old, z, atol=tol):
            break
    lz = _logsumexp(L, 2)                                               # unnormalised log q(z_i = k)
    if logodds:
        return logodds_against_key(lz, g), z
    return 1.0 - z[np.arange(n_i), g], z


def ebcc_onecoin(X, g, K=None, n_false=None, M=10, a_pi=0.1, a_v=4.0, b_v=3.0, iters=500, tol=1e-3, seed=1234, logodds=True):
    """EBCC's subtype model with one-coin confusion, for free-form answers that have no global class set: each item
    belongs to one of M latent subtypes, and worker j answers the truth with probability p_jm on items of subtype m,
    otherwise one of n_f false values uniformly. Shared subtype accuracies are what correlate the workers, as in EBCC.
    Priors follow EBCC's defaults: a_pi = 0.1 over subtypes and Beta(a_v, b_v) = Beta(4, 3) per accuracy, the
    correct-versus-wrong mass of EBCC's K = 4 confusion prior. Candidate truths and n_f are those of one-coin DS.
    Mean-field VB; ranking by log posterior odds against the key."""
    import scipy.sparse as ssp
    from scipy.special import digamma
    n_i, n_m = X.shape
    obs = X >= 0
    valid = _candidates(X, g, K)
    V = valid.shape[1]
    nv = valid.sum(1).astype(float)
    nf = np.maximum(nv - 1, 1) if n_false is None else np.full(n_i, float(n_false))
    lnf = np.log(nf)
    r, c = np.nonzero(obs)
    S = ssp.csr_matrix((np.ones(len(r)), (r * V + X[r, c], c)), shape=(n_i * V, n_m))   # (item, value) x worker
    St = S.T.tocsr()
    O = ssp.csr_matrix(obs.astype(np.float64))
    Ot = O.T.tocsr()
    n_obs_i = obs.sum(1).astype(float)
    C = np.asarray(S.sum(1)).reshape(n_i, V)
    z = np.where(valid, C + EPS, 0.0); z /= z.sum(1, keepdims=True)
    rng = np.random.default_rng(seed)
    R = rng.dirichlet(np.ones(M), size=(n_i, V)) * z[:, :, None]                       # q(z_i = v, s_i = m)
    L = None
    for _ in range(iters):
        qs = R.sum(1)                                                                   # (i, m)
        a = a_v + St @ R.reshape(n_i * V, M)                                            # (j, m)
        tot = Ot @ qs
        b = b_v + np.maximum(tot - (a - a_v), 0.0)
        eta = a_pi / M + qs.sum(0)
        Elog_tau = digamma(eta) - digamma(eta.sum())
        Elp = digamma(a) - digamma(a + b)
        El1 = digamma(b) - digamma(a + b)
        base = (O @ El1) - n_obs_i[:, None] * lnf[:, None]                              # every claim taken as false
        D = Elp - El1                                                                   # (j, m) without the n_f term
        L = (S @ D).reshape(n_i, V, M) + (C * lnf[:, None])[:, :, None] + base[:, None, :] + Elog_tau[None, None, :]
        L[~valid] = -np.inf
        mx = L.max((1, 2), keepdims=True)
        R = np.exp(L - mx); R /= R.sum((1, 2), keepdims=True)
        z_old, z = z, R.sum(2)
        if np.allclose(z_old, z, atol=tol):
            break
    lz = _logsumexp(L, 2)
    if logodds:
        return logodds_against_key(lz, g, valid), z
    return 1.0 - z[np.arange(n_i), g], z


# ----------------------------------------------------------------------------- PrecRecCorr
def _source_clusters(X, truth, max_size):
    """Average-linkage clusters of sources by the correlation of their correctness vectors, cut at the smallest number
    of clusters whose largest cluster has at most max_size sources (Pochampally et al. cluster sources by pairwise
    correlation and treat sources in different clusters as independent)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    n_m = X.shape[1]
    if n_m <= max_size:
        return [np.arange(n_m)]
    Y = (X == truth[:, None]).astype(np.float64)
    sd = Y.std(0)
    Y = (Y - Y.mean(0)) / np.where(sd > 0, sd, 1.0)
    R = np.clip((Y.T @ Y) / len(Y), -1, 1)
    np.fill_diagonal(R, 1.0)
    Z = linkage(squareform(1.0 - R, checks=False), "average")
    lo, hi = 1, n_m
    while lo < hi:                                     # smallest cluster count with every cluster <= max_size
        mid = (lo + hi) // 2
        lab = fcluster(Z, mid, "maxclust")
        if np.bincount(lab).max() <= max_size:
            hi = mid
        else:
            lo = mid + 1
    lab = fcluster(Z, lo, "maxclust")
    return [np.flatnonzero(lab == c) for c in np.unique(lab)]


def precreccorr(X, g, K=None, max_cluster=8, kappa=10.0, iters=5, logodds=True):
    """PrecRecCorr (Pochampally, Das Sarma, Dong, Meliou, Srivastava, SIGMOD 2014): the probability of a triple
    (item, value) from the joint recall and joint false positive rate of the sources that do and do not provide it
    (their Eqs. 10-12). Within a cluster of correlated sources the exact inclusion-exclusion sum equals the frequency of
    the cluster's provider pattern among true and among false triples, which is computed directly; sources in
    different clusters are treated as independent, as Pochampally et al. do for their large BOOK dataset. Training
    triples are the (item, value) pairs that at least one source provides; with no gold standard, the truth is
    estimated iteratively from majority voting (truth step: the value with the largest likelihood ratio). Unseen
    patterns are smoothed toward the independent model of their Theorem 3.1 with weight kappa. Scores are log
    posterior odds against the key over the candidate values."""
    n_i, n_m = X.shape
    obs = X >= 0
    valid = _candidates(X, g, K)
    V = valid.shape[1]
    C = onehot_counts(X, g)
    prov = (C > 0) & valid
    truth = np.where(valid, C, -1.0).argmax(1)
    clusters = _source_clusters(X, truth, max_cluster)
    L = None
    for _ in range(iters):
        T = np.zeros((n_i, V), bool); T[np.arange(n_i), truth] = True
        T &= prov
        F = prov & ~T
        nT, nF = T.sum(), F.sum()
        L = np.zeros((n_i, V))
        for members in clusters:
            k = len(members)
            P = np.zeros((n_i, V), np.int32)
            for b, j in enumerate(members):
                o = obs[:, j]
                P[np.flatnonzero(o), X[o, j]] |= (1 << b)
            npat = 1 << k
            Ht = np.bincount(P[T], minlength=npat).astype(float)
            Hf = np.bincount(P[F], minlength=npat).astype(float)
            bits = ((np.arange(npat)[:, None] >> np.arange(k)[None, :]) & 1).astype(bool)   # (pattern, member)
            r = np.array([((P[T] >> b) & 1).mean() for b in range(k)]).clip(1e-3, 1 - 1e-3)
            q = np.array([((P[F] >> b) & 1).mean() for b in range(k)]).clip(1e-3, 1 - 1e-3)
            pi_t = np.exp(np.where(bits, np.log(r), np.log1p(-r)).sum(1))
            pi_f = np.exp(np.where(bits, np.log(q), np.log1p(-q)).sum(1))
            lt = np.log((Ht + kappa * pi_t) / (nT + kappa))
            lf = np.log((Hf + kappa * pi_f) / (nF + kappa))
            L += (lt - lf)[P]
        L[~valid] = -np.inf
        truth = L.argmax(1)
    if logodds:
        return logodds_against_key(L, g, valid), L, clusters
    m = L.max(1, keepdims=True)
    post = np.exp(L - m); post /= post.sum(1, keepdims=True)
    return 1.0 - post[np.arange(n_i), g], L, clusters


def dong_order(G, D, acc):
    """Greedy order of Dong et al. (PVLDB 2009, Sec. 3.3) for the sources G that provide one value: first the more
    accurate source of the most dependent pair, then each time the source with the largest dependence on a source
    already selected."""
    sub = D[np.ix_(G, G)].copy()
    np.fill_diagonal(sub, -1.0)
    a, b = np.unravel_index(np.argmax(sub), sub.shape)
    first = a if acc[G[a]] >= acc[G[b]] else b
    sel = [first]
    left = np.ones(len(G), bool); left[first] = False
    best = sub[first].copy()
    for _ in range(len(G) - 1):
        cand = np.where(left, best, -np.inf)
        nxt = int(np.argmax(cand))
        sel.append(nxt); left[nxt] = False
        best = np.maximum(best, sub[nxt])
    return G[np.array(sel)]


# --------------------------------------------------------------------------------------- CRH
def crh(X, g, iters=20):
    """CRH for categorical data (Li et al., SIGMOD 2014): weighted vote with w_j = -log(loss_j / sum loss)."""
    n_i, n_m = X.shape
    obs = X >= 0
    w = np.ones(n_m)
    for _ in range(iters):
        C = onehot_counts(X, g, w)
        truth = C.argmax(1)
        loss = np.array([(X[obs[:, j], j] != truth[obs[:, j]]).sum() + 1.0 for j in range(n_m)])
        w = -np.log(loss / loss.sum())
    C = onehot_counts(X, g, w)
    return 1.0 - C[np.arange(n_i), g] / (C.sum(1) + EPS), w


# --------------------------------------------------------------------------------------- CATD
def catd(X, g, iters=20, alpha=0.05):
    """CATD for categorical data (Li et al., PVLDB 2014): weighted vote with the confidence-aware weight
    w_j = chi2_{alpha/2}(n_j) / loss_j, where n_j is the number of answers of model j and loss_j its 0/1 loss
    against the current truths (the lower confidence bound of the source's precision; small sources are shrunk)."""
    from scipy.stats import chi2
    n_i, n_m = X.shape
    obs = X >= 0
    n_obs = obs.sum(0)
    q = chi2.ppf(alpha / 2, np.maximum(n_obs, 1))
    w = np.ones(n_m)
    for _ in range(iters):
        C = onehot_counts(X, g, w)
        truth = C.argmax(1)
        loss = np.array([(X[obs[:, j], j] != truth[obs[:, j]]).sum() + 1.0 for j in range(n_m)])
        w = q / loss
        w = w / w.max()
    C = onehot_counts(X, g, w)
    return 1.0 - C[np.arange(n_i), g] / (C.sum(1) + EPS), w


# --------------------------------------------------------------------------------------- GLAD
def _logsig(x):
    return -np.logaddexp(0.0, -x)


def glad(X, g, K=None, n_false=None, iters=40, newton=3, prior=1.0, logodds=True, tol=1e-5):
    """Multi-class GLAD (Whitehill et al., NeurIPS 2009): model j answers the truth of item i with probability
    sigmoid(alpha_j * beta_i), beta_i = exp(b_i) > 0 the item's easiness, and otherwise uniformly among the false
    values (K - 1, |V_i| - 1, or a fixed n_false). EM with damped diagonal Newton steps in the M-step and
    Gaussian priors alpha_j ~ N(1, 1), b_i ~ N(0, 1). Returns (score, post, alpha, beta); the score is the log
    posterior odds against the key when logodds=True."""
    n_i, n_m = X.shape
    obs = X >= 0
    Xc = np.where(obs, X, 0)
    nv = _n_values(X, g, K)
    V = int(max(X.max(), g.max())) + 1
    valid = _candidates(X, g, K)
    lnf = np.log(np.maximum(nv - 1, 1) if n_false is None else np.full(n_i, float(n_false)))
    rows = np.arange(n_i)[:, None]
    flat = (rows * V + Xc)[obs]
    # initialise the posterior with smoothed vote shares
    C = np.bincount(flat, minlength=n_i * V).reshape(n_i, V).astype(np.float64)
    post = np.where(valid, C + 0.5, 0.0)
    post /= post.sum(1, keepdims=True)
    alpha = np.ones(n_m)
    b = np.zeros(n_i)
    lp = None
    prev = None
    for _ in range(iters):
        q = np.where(obs, post[rows, Xc], 0.0)          # expected correctness of each answer
        for _ in range(newton):
            beta = np.exp(b)
            a = alpha[None, :] * beta[:, None]
            s = 1.0 / (1.0 + np.exp(-a))
            r = np.where(obs, q - s, 0.0)
            h = np.where(obs, s * (1 - s), 0.0)
            ga = (r * beta[:, None]).sum(0) - prior * (alpha - 1.0)
            ha = (h * beta[:, None] ** 2).sum(0) + prior
            alpha = alpha + np.clip(ga / ha, -1.0, 1.0)
            a = alpha[None, :] * beta[:, None]
            s = 1.0 / (1.0 + np.exp(-a))
            r = np.where(obs, q - s, 0.0)
            h = np.where(obs, s * (1 - s), 0.0)
            gb = (r * a).sum(1) - prior * b
            hb = (h * a ** 2).sum(1) + prior
            b = b + np.clip(gb / hb, -1.0, 1.0)
        beta = np.exp(b)
        a = alpha[None, :] * beta[:, None]
        lw = _logsig(-a) - lnf[:, None]                     # log P(a specific false value)
        base = np.where(obs, lw, 0.0).sum(1)
        delta = np.where(obs, _logsig(a) - lw, 0.0)[obs]
        lp = np.bincount(flat, weights=delta, minlength=n_i * V).reshape(n_i, V) + base[:, None]
        lp = np.where(valid, lp, -np.inf)
        z = lp - lp.max(1, keepdims=True)
        post = np.exp(z)
        post /= post.sum(1, keepdims=True)
        if prev is not None and np.abs(post - prev).max() < tol:
            break
        prev = post
    if logodds:
        return logodds_against_key(lp, g, valid), post, alpha, beta
    return 1.0 - post[np.arange(n_i), g], post, alpha, beta


# ------------------------------------------------------------------------------ key-blind baseline
def answer_entropy(X, g):
    """Shannon entropy of the answer distribution of each item. It ignores the key: it ranks items by how much
    the population disagrees, the signal an annotator would use without any truth-inference model."""
    C = onehot_counts(X, g)
    P = C / (C.sum(1, keepdims=True) + EPS)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.where(P > 0, P * np.log(P), 0.0).sum(1)
    return H


# ------------------------------------------------------------------- Dong et al. ACCU / DEPEN
def same_false_rate(X, truth_val):
    """Empirical probability that two erring answers on the same item coincide:
    sum_i sum_v C(f_iv, 2) / sum_i C(F_i, 2), with f_iv the number of sources giving false value v on item i and
    F_i the number of false answers on item i. Under Dong et al.'s uniform false-value assumption this is 1/n."""
    num = den = 0.0
    for i in range(X.shape[0]):
        a = X[i][(X[i] >= 0) & (X[i] != truth_val[i])]
        if len(a) < 2:
            continue
        _, f = np.unique(a, return_counts=True)
        num += float((f * (f - 1)).sum()) / 2
        den += len(a) * (len(a) - 1) / 2
    return num / den if den > 0 else 0.0


def _agreement_counts(X, truth_val, chunk=4096):
    """same[a, b]: items on which sources a and b give the same value; same_true: the same restricted to the value
    truth_val. Computed over the (item, value) cells provided by at least two sources, in column chunks, so the cost
    does not grow with the number of distinct free-form values."""
    n_i, n_m = X.shape
    ii, mm = np.nonzero(X >= 0)
    vv = X[ii, mm].astype(np.int64)
    key = ii.astype(np.int64) * (int(X.max()) + 1) + vv
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    sel = cnt[inv] >= 2
    ii, mm, col = ii[sel], mm[sel], inv[sel]
    ucol, col = np.unique(col, return_inverse=True)
    cell_item = (uk[ucol] // (int(X.max()) + 1)).astype(np.int64)
    cell_val = (uk[ucol] % (int(X.max()) + 1)).astype(np.int64)
    is_true = cell_val == np.asarray(truth_val)[cell_item]
    same = np.zeros((n_m, n_m), np.float32)
    same_true = np.zeros((n_m, n_m), np.float32)
    n_c = len(ucol)
    order = np.argsort(col, kind="stable")
    col_s, mm_s = col[order], mm[order]
    bounds = np.searchsorted(col_s, np.arange(0, n_c + chunk, chunk))
    for b in range(len(bounds) - 1):
        lo, hi = bounds[b], bounds[b + 1]
        if lo == hi:
            continue
        c0 = b * chunk
        S = np.zeros((n_m, min(chunk, n_c - c0)), np.float32)
        S[mm_s[lo:hi], col_s[lo:hi] - c0] = 1.0
        same += S @ S.T
        St = S[:, is_true[c0:c0 + S.shape[1]]]
        if St.shape[1]:
            same_true += St @ St.T
    return same, same_true  # the diagonal undercounts singleton cells; dependence_posterior zeroes it


def dependence_posterior(X, truth_val, acc, n_false, alpha=0.2, c=0.8, pf_same=None, soft_true=None):
    """Pairwise copying posterior (Dong, Berti-Equille, Srivastava, PVLDB 2009) from agreement counts.

    Counts are over items both models answered: kt = same & true, kf = same & false, kd = different.
    pf_same is the probability that two independent sources that both err choose the same false value: 1/n_false
    under the uniform false-value assumption (default), or an empirical rate (the NONUNI relaxation of Dong et al.).
    soft_true, when given, is the probability that any provided value is true; shared values then count as
    soft_true shared true values and (1 - soft_true) shared false values. Dong et al. start their iteration this
    way, with every value true with probability 1 - error rate, before any truth is selected.
    Returns a symmetric (n_m, n_m) matrix of P(dependent).
    """
    n_i, n_m = X.shape
    obs = (X >= 0).astype(np.float32)
    same, same_true = _agreement_counts(X, truth_val)
    common = obs.T @ obs
    if soft_true is not None:
        same_true = soft_true * same
    kt, kf = same_true, same - same_true
    kd = common - same
    A1, A2 = acc[:, None], acc[None, :]
    Pt = A1 * A2
    Pf = (1 - A1) * (1 - A2) * ((1.0 / n_false) if pf_same is None else pf_same)
    Pd = np.clip(1 - Pt - Pf, EPS, 1)
    li = kt * np.log(Pt + EPS) + kf * np.log(Pf + EPS) + kd * np.log(Pd)
    def ldep(Acop):
        return (kt * np.log(Acop * c + Pt * (1 - c) + EPS) + kf * np.log((1 - Acop) * c + Pf * (1 - c) + EPS)
                + kd * np.log(Pd * (1 - c)))
    l12, l21 = ldep(A1), ldep(A2)
    m = np.maximum(np.maximum(li, l12), l21)
    num = alpha / 2 * (np.exp(l12 - m) + np.exp(l21 - m))
    den = num + (1 - alpha) * np.exp(li - m)
    D = num / den
    np.fill_diagonal(D, 0.0)
    return D


def accu_depen(X, g, n_false, iters=5, alpha=0.2, c=0.8, use_dependence=True, K=None, logodds=False, nonuni=False,
               truth_override=None, eps=0.2, init=None, vote_order="accuracy"):
    """ACCU with dependence (Dong et al. 2009): accuracy-weighted votes, each vote discounted by the
    probability that the value was copied from a source ordered earlier (sources ordered by accuracy).
    nonuni=True replaces the uniform false-value assumption in the copying test by the empirical same-false-value
    rate (Dong et al.'s NONUNI relaxation). truth_override fixes the truth used by the copying test (oracle runs).
    As in Dong et al., the first round treats every provided value as true with probability 1 - eps and every source
    as having accuracy 1 - eps; later rounds use the selected true values.
    init = (truth_val, acc) warm-starts the iteration from given true values and accuracies (e.g. the converged state of
    ACCU without dependence) and skips the soft first round."""
    n_i, n_m = X.shape
    obs = X >= 0
    acc = np.full(n_m, 1.0 - eps)
    V = int(max(X.max(), g.max())) + 1
    truth = majority_vote(X, g)  # placeholder, replaced below
    C = onehot_counts(X, g); truth_val = C.argmax(1)
    if init is not None:
        truth_val, acc = np.asarray(init[0]).copy(), np.asarray(init[1], float).copy()
    D = np.zeros((n_m, n_m))
    post = None
    for it in range(iters):
        if use_dependence:
            tv = truth_val if truth_override is None else truth_override
            ps = same_false_rate(X, tv) if nonuni else None
            first = it == 0 and truth_override is None and init is None
            D = dependence_posterior(X, tv, acc, n_false, alpha, c, pf_same=ps, soft_true=(1.0 - eps) if first else None)
        a = np.log(n_false * np.clip(acc, 0.01, 0.99) / (1 - np.clip(acc, 0.01, 0.99)))
        order = np.argsort(-acc)
        rank = np.empty(n_m, int); rank[order] = np.arange(n_m)
        L = np.log(np.clip(1 - c * D, EPS, 1))
        conf = np.zeros((n_i, V))
        for i in range(n_i):
            o = np.nonzero(obs[i])[0]
            vals = X[i, o]
            for v in np.unique(vals):
                G = o[vals == v]
                if use_dependence and len(G) > 1:
                    G = dong_order(G, D, acc) if vote_order == "dependence" else G[np.argsort(rank[G])]
                    sub = L[np.ix_(G, G)]
                    disc = np.exp(np.tril(sub, -1).sum(1))  # product over earlier sources in G
                else:
                    disc = np.ones(len(G))
                conf[i, v] = (a[G] * disc).sum()
        # truth probabilities: softmax over candidate values (unobserved values get confidence 0)
        valid = _candidates(X, g, K)
        z = np.where(valid, conf, -np.inf)
        z -= z.max(1, keepdims=True)
        post = np.exp(z); post /= post.sum(1, keepdims=True)
        truth_val = post.argmax(1)
        acc = np.array([post[np.nonzero(obs[:, j])[0], X[obs[:, j], j]].mean() if obs[:, j].any() else 0.5
                        for j in range(n_m)])
    if logodds:
        return logodds_against_key(z, g, valid), post, acc, D
    return 1.0 - post[np.arange(n_i), g], post, acc, D


# ------------------------------------------------------------------------------------ IRT 4PL
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def irt_abilities(Y, M, iters=400, lr=0.05, min_r=0.2):
    """2PL joint fit on 'aligned' items (item-rest correlation > min_r) to fix the ability scale.
    Y: (n_i, n_m) 0/1 correctness, M: mask of observed entries."""
    n_i, n_m = Y.shape
    Yf = np.where(M, Y, 0.0)
    acc_m = (Yf.sum(0) / np.maximum(M.sum(0), 1))
    rest = (Yf * 1.0)
    r = np.array([np.corrcoef(Yf[i, M[i]], acc_m[M[i]])[0, 1] if M[i].sum() > 2 and Yf[i, M[i]].std() > 0 else 0
                  for i in range(n_i)])
    al = np.nan_to_num(r) > min_r
    Ya, Ma = Yf[al], M[al]
    th = (acc_m - acc_m.mean()) / (acc_m.std() + EPS)
    a = np.ones(Ya.shape[0]); b = np.zeros(Ya.shape[0])
    for _ in range(iters):
        z = a[:, None] * (th[None, :] - b[:, None])
        p = _sig(z)
        e = (Ya - p) * Ma
        th += lr * (e * a[:, None]).sum(0) / np.maximum(Ma.sum(0), 1)
        th = (th - th.mean()) / (th.std() + EPS)
        a += lr * (e * (th[None, :] - b[:, None])).sum(1) / np.maximum(Ma.sum(1), 1)
        b += lr * (-(e * a[:, None])).sum(1) / np.maximum(Ma.sum(1), 1)
        a = np.clip(a, 1 / 15, 15); b = np.clip(b, -2, 2)
    return th, al


def _fit_3p(Y, M, th, ceiling, iters=600, lr=0.05):
    """Vectorised per-item fit of P = c + (ceiling - c) * sigmoid(a (theta - b)), theta fixed.
    Returns per-item log-likelihood. Bounds a in [1/15, 15], b in [-2, 2], c in [0, 1]."""
    n_i = Y.shape[0]
    a = np.ones(n_i); b = np.zeros(n_i); cl = np.full(n_i, -1.5)  # c = sigmoid(cl)
    m_a = v_a = m_b = v_b = m_c = v_c = 0.0
    b1, b2 = 0.9, 0.999
    for t in range(1, iters + 1):
        cc = _sig(cl)
        s = _sig(a[:, None] * (th[None, :] - b[:, None]))
        P = np.clip(cc[:, None] + (ceiling - cc[:, None]) * s, 1e-6, 1 - 1e-6)
        dLdP = (Y / P - (1 - Y) / (1 - P)) * M
        ds = s * (1 - s)
        ga = (dLdP * (ceiling - cc[:, None]) * ds * (th[None, :] - b[:, None])).sum(1)
        gb = (dLdP * (ceiling - cc[:, None]) * ds * (-a[:, None])).sum(1)
        gc = (dLdP * (1 - s)).sum(1) * cc * (1 - cc)
        n = np.maximum(M.sum(1), 1)
        ga, gb, gc = ga / n, gb / n, gc / n
        m_a = b1 * m_a + (1 - b1) * ga; v_a = b2 * v_a + (1 - b2) * ga ** 2
        m_b = b1 * m_b + (1 - b1) * gb; v_b = b2 * v_b + (1 - b2) * gb ** 2
        m_c = b1 * m_c + (1 - b1) * gc; v_c = b2 * v_c + (1 - b2) * gc ** 2
        corr = np.sqrt(1 - b2 ** t) / (1 - b1 ** t)
        a = np.clip(a + lr * corr * m_a / (np.sqrt(v_a) + 1e-8), 1 / 15, 15)
        b = np.clip(b + lr * corr * m_b / (np.sqrt(v_b) + 1e-8), -2, 2)
        cl = np.clip(cl + lr * corr * m_c / (np.sqrt(v_c) + 1e-8), -12, 12)
    cc = _sig(cl)
    P = np.clip(cc[:, None] + (ceiling - cc[:, None]) * _sig(a[:, None] * (th[None, :] - b[:, None])), 1e-6, 1 - 1e-6)
    return (np.where(M, Y * np.log(P) + (1 - Y) * np.log(1 - P), 0.0)).sum(1)


def irt_ceiling_lr_ref(X, g, Y=None, M=None):
    """Reference (float64, fixed iterations) implementation, kept for validation of the fast version."""
    if M is None:
        M = X >= 0
    if Y is None:
        Y = (X == g[:, None]).astype(float)
    th, _ = irt_abilities(Y, M)
    l1 = _fit_3p(Y, M, th, ceiling=1.0)
    l0 = _fit_3p(Y, M, th, ceiling=0.0)
    return l0 - l1


def _abilities_fast(Y, M, iters=400, lr=0.05, min_r=0.2, tol=1e-5):
    """Same 2PL ability fit as irt_abilities, in float32 with a vectorised item-rest correlation and early
    stopping once the abilities stop moving."""
    f = np.float32
    Mf = M.astype(f); Yf = (Y * M).astype(f)
    nm = np.maximum(Mf.sum(0), 1)
    acc = Yf.sum(0) / nm
    ni = np.maximum(Mf.sum(1), 1)
    my = Yf.sum(1) / ni
    ma = (Mf * acc[None, :]).sum(1) / ni
    dy = (Yf - my[:, None]) * Mf
    da = (acc[None, :] - ma[:, None]) * Mf
    cov = (dy * da).sum(1); vy = (dy * dy).sum(1); va = (da * da).sum(1)
    r = np.where((vy > 0) & (va > 0) & (Mf.sum(1) > 2), cov / np.sqrt(np.maximum(vy * va, 1e-30)), 0.0)
    al = r > min_r
    Ya, Ma = Yf[al], Mf[al]
    th = ((acc - acc.mean()) / (acc.std() + EPS)).astype(f)
    a = np.ones(Ya.shape[0], f); b = np.zeros(Ya.shape[0], f)
    nma = np.maximum(Ma.sum(0), 1); nia = np.maximum(Ma.sum(1), 1)
    for k in range(iters):
        d = th[None, :] - b[:, None]
        e = (Ya - _sig(a[:, None] * d)) * Ma
        th_new = th + lr * (e * a[:, None]).sum(0) / nma
        th_new = (th_new - th_new.mean()) / (th_new.std() + EPS)
        a = np.clip(a + lr * (e * (th_new[None, :] - b[:, None])).sum(1) / nia, 1 / 15, 15).astype(f)
        b = np.clip(b - lr * (e * a[:, None]).sum(1) / nia, -2, 2).astype(f)
        moved = float(np.abs(th_new - th).max())
        th = th_new.astype(f)
        if moved < tol:
            break
    return th, al


def _fit_3p_fast(Y, M, th, ceiling, iters=600, lr=0.05, tol=1e-4):
    """Same per-item 3-parameter fit as _fit_3p, in float32, with early stopping when no item's parameters
    move by more than tol over 20 iterations."""
    f = np.float32
    Y = Y.astype(f); Mf = M.astype(f); th = th.astype(f)
    n_i = Y.shape[0]
    a = np.ones(n_i, f); b = np.zeros(n_i, f); cl = np.full(n_i, -1.5, f)
    m_a = np.zeros(n_i, f); v_a = np.zeros(n_i, f); m_b = np.zeros(n_i, f); v_b = np.zeros(n_i, f)
    m_c = np.zeros(n_i, f); v_c = np.zeros(n_i, f)
    b1, b2 = 0.9, 0.999
    n = np.maximum(Mf.sum(1), 1)
    snap = (a.copy(), b.copy(), cl.copy())
    for t in range(1, iters + 1):
        cc = _sig(cl)
        d = th[None, :] - b[:, None]
        s = _sig(a[:, None] * d)
        P = np.clip(cc[:, None] + (ceiling - cc[:, None]) * s, 1e-6, 1 - 1e-6)
        dLdP = (Y / P - (1 - Y) / (1 - P)) * Mf
        w = dLdP * (ceiling - cc[:, None]) * (s * (1 - s))
        ga = (w * d).sum(1) / n
        gb = -(w.sum(1) * a) / n
        gc = (dLdP * (1 - s)).sum(1) * cc * (1 - cc) / n
        m_a = b1 * m_a + (1 - b1) * ga; v_a = b2 * v_a + (1 - b2) * ga * ga
        m_b = b1 * m_b + (1 - b1) * gb; v_b = b2 * v_b + (1 - b2) * gb * gb
        m_c = b1 * m_c + (1 - b1) * gc; v_c = b2 * v_c + (1 - b2) * gc * gc
        corr = np.sqrt(1 - b2 ** t) / (1 - b1 ** t)
        a = np.clip(a + lr * corr * m_a / (np.sqrt(v_a) + 1e-8), 1 / 15, 15).astype(f)
        b = np.clip(b + lr * corr * m_b / (np.sqrt(v_b) + 1e-8), -2, 2).astype(f)
        cl = np.clip(cl + lr * corr * m_c / (np.sqrt(v_c) + 1e-8), -12, 12).astype(f)
        if t % 20 == 0:
            moved = max(float(np.abs(a - snap[0]).max()), float(np.abs(b - snap[1]).max()), float(np.abs(cl - snap[2]).max()))
            if moved < tol:
                break
            snap = (a.copy(), b.copy(), cl.copy())
    cc = _sig(cl)
    P = np.clip(cc[:, None] + (ceiling - cc[:, None]) * _sig(a[:, None] * (th[None, :] - b[:, None])), 1e-6, 1 - 1e-6)
    return (Mf * (Y * np.log(P) + (1 - Y) * np.log(1 - P))).sum(1).astype(np.float64)


def irt_ceiling_lr(X, g, Y=None, M=None, min_r=0.2):
    """Land & Bikel-style indicator: Delta l_i = l_i(d=0) - l_i(d=1) with abilities fixed from a 2PL fit.
    Y/M (correctness and observation mask) may be passed explicitly, e.g. to count unparseable free-form
    answers as observed and incorrect while truth-inference methods treat them as missing.
    Fast float32 implementation with early stopping; validated against irt_ceiling_lr_ref."""
    if M is None:
        M = X >= 0
    if Y is None:
        Y = (X == g[:, None]).astype(np.float32)
    th, _ = _abilities_fast(Y, M, min_r=min_r)
    l1 = _fit_3p_fast(Y, M, th, ceiling=1.0)
    l0 = _fit_3p_fast(Y, M, th, ceiling=0.0)
    return l0 - l1


# ------------------------------------------------------------------------------------ metrics
def precision_at_k(score, y, k):
    k = min(k, len(score))
    idx = np.argsort(-score, kind="stable")[:k]
    return float(y[idx].mean())


def average_precision(score, y):
    idx = np.argsort(-score, kind="stable")
    yy = y[idx].astype(float)
    if yy.sum() == 0:
        return float("nan")
    cum = np.cumsum(yy)
    return float((cum / np.arange(1, len(yy) + 1))[yy == 1].mean())


def _tie_groups(score):
    """Sorted order (descending) and, per sorted position, the tie group's start offset and size."""
    s = np.asarray(score, np.float64)
    idx = np.argsort(-s, kind="stable")
    ss = s[idx]
    new = np.empty(len(ss), bool)
    new[0] = True
    new[1:] = ss[1:] != ss[:-1]
    gid = np.cumsum(new) - 1
    start = np.flatnonzero(new)
    size = np.diff(np.append(start, len(ss)))
    return idx, gid, start, size


def average_precision_expected(score, y):
    """Expected AP when tied scores are ordered uniformly at random (McSherry & Najork, ECIR 2008).
    Position i (1-based) of a tie group with n items and k positives, preceded by o items of which p are positive,
    is positive with probability k/n and then has 1 + (i-1)(k-1)/(n-1) expected positives up to it in the group."""
    y = np.asarray(y).astype(np.float64)
    R = y.sum()
    if R == 0:
        return float("nan")
    idx, gid, start, size = _tie_groups(score)
    yy = y[idx]
    kpos = np.bincount(gid, weights=yy)                     # positives per group
    p_before = np.concatenate([[0.0], np.cumsum(kpos)[:-1]])
    n = size[gid].astype(np.float64)
    k = kpos[gid]
    o = start[gid].astype(np.float64)
    i = np.arange(len(yy)) - start[gid] + 1.0
    frac = np.where(n > 1, (i - 1) * (k - 1) / np.maximum(n - 1, 1), 0.0)
    contrib = (k / n) * (p_before[gid] + 1 + frac) / (o + i)
    return float(contrib.sum() / R)


def precision_at_k_expected(score, y, k):
    """Expected precision at k when the tie group straddling the cutoff is ordered at random."""
    y = np.asarray(y).astype(np.float64)
    idx, gid, start, size = _tie_groups(score)
    yy = y[idx]
    kpos = np.bincount(gid, weights=yy)
    g_end = start + size
    full = g_end <= k
    hits = kpos[full].sum()
    cut = np.flatnonzero((start < k) & (g_end > k))
    if len(cut):
        c = cut[0]
        hits += kpos[c] * (k - start[c]) / size[c]
    return float(hits / k)


def average_precision_expected_batch(S, Y):
    """Row-wise expected AP under random ordering of ties (same formula as average_precision_expected) for score and
    label matrices of shape (B, n), e.g. bootstrap resamples; duplicated items in a resample are tied with each other."""
    S = np.asarray(S, np.float64); Y = np.asarray(Y, np.float64)
    B, n = S.shape
    o = np.argsort(-S, axis=1, kind="stable")
    Ss = np.take_along_axis(S, o, 1); Ys = np.take_along_axis(Y, o, 1)
    new = np.ones((B, n), bool); new[:, 1:] = Ss[:, 1:] != Ss[:, :-1]
    gid = np.cumsum(new, 1) - 1
    G = gid + (np.arange(B) * n)[:, None]                      # globally unique group ids
    flatG = G.ravel()
    size = np.bincount(flatG, minlength=B * n).astype(np.float64)
    kpos = np.bincount(flatG, weights=Ys.ravel(), minlength=B * n)
    pos_idx = np.broadcast_to(np.arange(n), (B, n))
    start = np.full(B * n, n, np.int64)
    np.minimum.at(start, flatG, pos_idx.ravel())
    # positives before each group: cumulative positives at the group's start position
    cum = np.cumsum(Ys, 1) - Ys
    p_before = np.zeros(B * n)
    first = new.ravel()
    p_before[flatG[first]] = cum.ravel()[first]
    n_g = size[G]; k_g = kpos[G]; o_g = start[G].astype(np.float64); pb = p_before[G]
    i = pos_idx - start[G] + 1.0
    frac = np.where(n_g > 1, (i - 1) * (k_g - 1) / np.maximum(n_g - 1, 1), 0.0)
    contrib = (k_g / n_g) * (pb + 1 + frac) / (o_g + i)
    R = Ys.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(R > 0, contrib.sum(1) / R, np.nan)
