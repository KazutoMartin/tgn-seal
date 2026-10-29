#!/usr/bin/env python
"""
hawkes_analysis.py
==================
Tests whether node-level interaction streams in a CTDG dataset are
self-exciting (Hawkes), controlling for daily/weekly seasonality, and derives
a data-driven decay rate for the Spatiotemporal Decaying Heuristic

    W_decay = lambda_decay ** (dt / decay_scale)  ==  exp(-beta * dt),
    beta = -ln(lambda_decay) / decay_scale

Place this file in the TGN repo root (next to train_self_supervised.py) and run:

    python hawkes_analysis.py -d <dataset_name> [<dataset_name> ...]

Fitting uses the TRAIN split only (same split as train_self_supervised.py), so
the derived hyperparameter does not leak validation/test information.

What it computes, per dataset
-----------------------------
1. Burstiness B = (sigma - m)/(sigma + m) and memory coefficient of node
   inter-event times (Goh & Barabasi 2008). Poisson => B = 0.
2. Per-node exponential-kernel Hawkes processes
       lambda_i(t) = mu_i * s(t) + alpha_i * sum_{t_j < t} exp(-beta (t - t_j))
   with node-specific mu_i, alpha_i (branching n_i = alpha_i / beta) and a
   SHARED beta, chosen by profile likelihood. s(t) is either 1 (homogeneous
   baseline) or an hour-of-week profile estimated from the data (seasonal
   baseline). The seasonal version is the conservative one: daily/weekly
   rhythms alone can masquerade as self-excitation.
3. Likelihood-ratio / AIC comparison against (seasonal) Poisson.
4. Goodness of fit via the time-rescaling theorem (compensator increments
   should be i.i.d. Exp(1)): KS distance + QQ plot.
5. Optional two-time-scale kernel  a1 e^{-b1 dt} + a2 e^{-b2 dt}.
6. Where the current hand-tuned lambda_decay / decay_scale sits on the profile.
7. Recommended --lambda_decay / --decay_scale values.

Outputs: printed report, <outdir>/<dataset>/summary.json and PNG plots.
Dependencies: numpy, scipy, (matplotlib for plots), (numba for speed; optional).
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
from scipy import stats
from scipy.optimize import minimize, minimize_scalar

try:
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pure-python fallback
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f


LN2 = math.log(2.0)
WEEK = 7 * 86400.0


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_events(name, split):
    """Returns (sources, destinations, timestamps) for the requested split."""
    try:
        sys.path.insert(0, os.getcwd())
        from utils.data_processing import get_data

        out = get_data(
            name,
            different_new_nodes_between_val_and_test=False,
            randomize_features=False,
        )
        full_data, train_data = out[2], out[3]
        d = train_data if split == "train" else full_data
        src, dst, ts = d.sources, d.destinations, d.timestamps
        print(f"[data] loaded '{name}' via utils.data_processing.get_data ({split} split)")
    except Exception as e:  # fallback: raw TGN csv, 70% chronological train split
        import pandas as pd

        path = f"./data/ml_{name}.csv"
        print(f"[data] get_data failed ({type(e).__name__}: {e}); reading {path}")
        df = pd.read_csv(path)
        src, dst, ts = df["u"].values, df["i"].values, df["ts"].values.astype(np.float64)
        if split == "train":
            val_time = np.quantile(ts, 0.70)
            m = ts <= val_time
            src, dst, ts = src[m], dst[m], ts[m]
    return (
        np.asarray(src, dtype=np.int64),
        np.asarray(dst, dtype=np.int64),
        np.asarray(ts, dtype=np.float64),
    )


def node_streams(src, dst, ts, min_events, merge_ties):
    """Per-node sorted event times (an event counts for both endpoints)."""
    nodes = np.concatenate([src, dst])
    times = np.concatenate([ts, ts])
    order = np.lexsort((times, nodes))
    nodes, times = nodes[order], times[order]
    cuts = np.flatnonzero(np.diff(nodes)) + 1
    starts = np.r_[0, cuts]
    streams = {}
    n_ties_removed = 0
    for nid, tt in zip(nodes[starts], np.split(times, cuts)):
        if merge_ties:
            u = np.unique(tt)
            n_ties_removed += len(tt) - len(u)
            tt = u
        if len(tt) >= min_events:
            streams[int(nid)] = np.ascontiguousarray(tt, dtype=np.float64)
    return streams, n_ties_removed


# ----------------------------------------------------------------------------
# Seasonal (hour-of-week) baseline profile
# ----------------------------------------------------------------------------
class WeeklyProfile:
    """Piecewise-constant periodic rate multiplier s(t) with mean 1."""

    def __init__(self, ts, enabled=True, period=WEEK, bin_s=3600.0):
        nb = int(round(period / bin_s))
        if enabled:
            idx = ((ts % period) // bin_s).astype(int)
            counts = np.bincount(idx, minlength=nb).astype(float) + 1.0  # Laplace smoothing
            self.s = counts / counts.mean()
        else:
            self.s = np.ones(nb)
        self.P, self.bin, self.nb = period, bin_s, nb
        self.cum = np.concatenate([[0.0], np.cumsum(self.s * bin_s)])

    def rate(self, t):
        b = np.minimum(((np.asarray(t) % self.P) // self.bin).astype(int), self.nb - 1)
        return self.s[b]

    def integral(self, t):
        """\\int_0^t s(u) du"""
        t = np.asarray(t, dtype=np.float64)
        w = np.floor(t / self.P)
        r = t - w * self.P
        b = np.minimum((r // self.bin).astype(int), self.nb - 1)
        return w * self.cum[-1] + self.cum[b] + self.s[b] * (r - b * self.bin)


def build_node_data(streams, T_end, prof):
    nd = {}
    for nid, t in streams.items():
        sk = prof.rate(t[1:])
        cumS = prof.integral(t)
        S = float(prof.integral(np.array([T_end]))[0] - cumS[0])
        if S <= 0:
            continue
        nd[nid] = dict(t=t, sk=sk, S=S, cumS=cumS, sum_log_sk=float(np.log(sk).sum()))
    return nd


# ----------------------------------------------------------------------------
# Hawkes likelihood (exponential kernels)
# ----------------------------------------------------------------------------
@njit(cache=True)
def excitation_sums(t, beta):
    """A[k] = sum_{j<k} exp(-beta (t_k - t_j)), computed recursively in O(N)."""
    n = t.shape[0]
    A = np.zeros(n)
    for k in range(1, n):
        A[k] = math.exp(-beta * (t[k] - t[k - 1])) * (1.0 + A[k - 1])
    return A


def poisson_ll(d):
    N1 = len(d["sk"])
    mu = N1 / d["S"]
    return N1 * math.log(mu) + d["sum_log_sk"] - N1


def fit_hawkes(d, A_full_list, betas, T_end, x0=None):
    """
    Maximises the (concave) log-likelihood over mu and alpha_c for fixed betas.
    The first event of a node opens its observation window and is conditioned
    on, so the likelihood covers events 1..N-1 on (t_0, T_end].
    Parametrisation: mu = x0 * N/S, alpha_c = n_c * beta_c  (n_c = branching).
    Returns (loglik, x = [mu_tilde, n_1, ..., n_m]).
    """
    sk, S, t = d["sk"], d["S"], d["t"]
    N1 = len(sk)
    rate0 = N1 / S
    bet = np.asarray(betas, dtype=np.float64)
    A = np.stack([a[1:] for a in A_full_list])  # (m, N-1)
    G = np.array([(1.0 - np.exp(-b * (T_end - t))).sum() / b for b in bet])

    def f(x):
        mu = x[0] * rate0
        alpha = x[1:] * bet
        lam = np.maximum(mu * sk + alpha @ A, 1e-300)
        inv = 1.0 / lam
        ll = np.log(lam).sum() - mu * S - alpha @ G
        gmu = (sk * inv).sum() - S
        ga = A @ inv - G
        return -ll, -np.concatenate([[gmu * rate0], ga * bet])

    m = len(bet)
    if x0 is None or len(x0) != m + 1:
        x0 = np.r_[0.5, np.full(m, 0.3 / m)]
    bounds = [(1e-6, None)] + [(0.0, 20.0)] * m
    r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bounds)
    return -float(r.fun), r.x


def eval_beta(nd, beta, T_end, warm=None, Acache=None, key=None):
    """Profile log-likelihood (sum over nodes) at a shared beta."""
    tot, per_ll, per_x = 0.0, {}, {}
    for nid, d in nd.items():
        if Acache is not None and (nid, key) in Acache:
            A = Acache[(nid, key)]
        else:
            A = excitation_sums(d["t"], beta)
            if Acache is not None and key is not None:
                Acache[(nid, key)] = A
        ll, x = fit_hawkes(d, [A], [beta], T_end, None if warm is None else warm.get(nid))
        tot += ll
        per_ll[nid], per_x[nid] = ll, x
    return tot, per_ll, per_x


def profile_ci(D, beta_star, ll_star, betas, prof, T_end, warm, drop=1.92):
    """95% profile-likelihood CI on the half-life, by root finding in log(beta)."""
    from scipy.optimize import brentq

    target = ll_star - drop
    g = lambda lb: eval_beta(D, math.exp(lb), T_end, warm)[0] - target
    lbs = math.log(beta_star)
    out = []
    for side in (+1, -1):  # +1: larger beta (shorter half-life)
        cand = [math.log(b) for b, p in zip(betas, prof)
                if p < target and (math.log(b) - lbs) * side > 0]
        if not cand:
            out.append(float("nan"))  # profile never drops below threshold on this side
            continue
        lb_far = min(cand, key=lambda v: abs(v - lbs))
        try:
            out.append(hl(math.exp(brentq(g, *sorted((lbs, lb_far)), xtol=1e-4))))
        except ValueError:
            out.append(float("nan"))
    return (out[0], out[1])  # (short half-life end, long half-life end)


# ----------------------------------------------------------------------------
# Drifting baseline: per-node rate re-estimated in fixed windows (e.g. 30 d)
# ----------------------------------------------------------------------------
def add_drift_windows(nd, T_end, prof, t_origin, window_s):
    """Adds per-node window index of each event and the baseline integral per window."""
    n_w = int(math.ceil((T_end - t_origin) / window_s))
    edges = t_origin + window_s * np.arange(n_w + 1)
    edges[-1] = max(edges[-1], T_end)
    for d in nd.values():
        t = d["t"]
        a = np.clip(edges[:-1], t[0], T_end)
        b = np.clip(edges[1:], t[0], T_end)
        Sw = prof.integral(b) - prof.integral(a)
        widx = np.clip(np.searchsorted(edges, t[1:], side="right") - 1, 0, n_w - 1)
        keep = Sw > 0
        remap = -np.ones(n_w, dtype=np.int64)
        remap[keep] = np.arange(keep.sum())
        d["w_idx"] = remap[widx]
        d["S_w"] = Sw[keep]
        if (d["w_idx"] < 0).any():  # should not happen (s(t) > 0 everywhere)
            d["w_idx"] = np.maximum(d["w_idx"], 0)


def poisson_ll_drift(d):
    Nw = np.bincount(d["w_idx"], minlength=len(d["S_w"])).astype(float)
    m = Nw > 0
    return float((Nw[m] * np.log(Nw[m] / d["S_w"][m])).sum() + d["sum_log_sk"] - Nw.sum())


def fit_hawkes_drift(d, A_full_list, betas, T_end, x0=None):
    """Same as fit_hawkes, but mu is piecewise constant over the drift windows."""
    sk, t, widx, Sw = d["sk"], d["t"], d["w_idx"], d["S_w"]
    W = len(Sw)
    rate0 = len(sk) / d["S"]
    bet = np.asarray(betas, dtype=np.float64)
    A = np.stack([a[1:] for a in A_full_list])
    G = np.array([(1.0 - np.exp(-b * (T_end - t))).sum() / b for b in bet])

    def f(x):
        mu = x[:W] * rate0
        alpha = x[W:] * bet
        lam = np.maximum(mu[widx] * sk + alpha @ A, 1e-300)
        inv = 1.0 / lam
        ll = np.log(lam).sum() - mu @ Sw - alpha @ G
        gmu = np.bincount(widx, weights=sk * inv, minlength=W) - Sw
        ga = A @ inv - G
        return -ll, -np.concatenate([gmu * rate0, ga * bet])

    m = len(bet)
    if x0 is None or len(x0) != W + m:
        Nw = np.bincount(widx, minlength=W).astype(float)
        x0 = np.r_[0.5 * Nw / (Sw * rate0) + 1e-3, np.full(m, 0.3 / m)]
    bounds = [(1e-8, None)] * W + [(0.0, 20.0)] * m
    r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bounds)
    return -float(r.fun), r.x


def rescaled_times(d, A_full_list, betas, x, T_end):
    """Time-rescaling theorem: compensator increments between events."""
    t, cumS = d["t"], d["cumS"]
    N1 = len(t) - 1
    mu = x[0] * N1 / d["S"]
    dt = np.diff(t)
    tau = mu * np.diff(cumS)
    for A, b, n in zip(A_full_list, betas, x[1:]):
        alpha = n * b
        tau = tau + (alpha / b) * (1.0 - np.exp(-b * dt)) * (1.0 + A[:-1])
    return tau


def ks_exp1(tau):
    tau = tau[np.isfinite(tau) & (tau >= 0)]
    return float(stats.kstest(tau, "expon").statistic), tau


# ----------------------------------------------------------------------------
# Descriptive statistics
# ----------------------------------------------------------------------------
def burstiness_stats(streams):
    Bs, Ms, w, iets = [], [], [], []
    for t in streams.values():
        iet = np.diff(t)
        iet = iet[iet > 0]
        if len(iet) < 10:
            continue
        m, s = iet.mean(), iet.std()
        Bs.append((s - m) / (s + m))
        if len(iet) > 3 and iet[:-1].std() > 0 and iet[1:].std() > 0:
            Ms.append(np.corrcoef(iet[:-1], iet[1:])[0, 1])
        w.append(len(iet))
        iets.append(iet)
    Bs, w = np.array(Bs), np.array(w, dtype=float)
    return dict(
        n_nodes=int(len(Bs)),
        B_median=float(np.median(Bs)),
        B_weighted_mean=float(np.average(Bs, weights=w)),
        B_frac_positive=float((Bs > 0).mean()),
        memory_coef_median=float(np.median(Ms)) if Ms else float("nan"),
    ), (np.concatenate(iets) if iets else np.array([]))


def B_of(x):
    m, s = x.mean(), x.std()
    return (s - m) / (s + m)


def hl(beta):
    return LN2 / beta


def fmt_dur(sec):
    if not np.isfinite(sec):
        return "n/a"
    if sec < 3600:
        return f"{sec / 60:.1f} min"
    if sec < 86400 * 2:
        return f"{sec / 3600:.2f} h"
    return f"{sec / 86400:.2f} d"


# ----------------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------------
def analyse(name, args):
    t0 = time.time()
    out_dir = os.path.join(args.outdir, name)
    os.makedirs(out_dir, exist_ok=True)
    R = {"dataset": name, "split": args.split}

    src, dst, ts = load_events(name, args.split)
    T_end = float(ts.max())
    span = T_end - float(ts.min())
    print(f"\n{'=' * 78}\nDATASET: {name}   events={len(ts)}   span={fmt_dur(span)} (assuming seconds)")
    if span < 30 * 86400:
        print("[warn] span < 30 days: check that timestamps are in seconds; "
              "seasonal profile and half-lives assume seconds.")
    R["n_events"], R["span_seconds"] = int(len(ts)), span

    streams, n_ties = node_streams(src, dst, ts, args.min_events, not args.keep_ties)
    n_ev = sum(len(t) for t in streams.values())
    print(f"[nodes] {len(streams)} nodes with >= {args.min_events} events "
          f"({n_ev} node-events; {n_ties} same-timestamp duplicates merged)")
    R["n_nodes_fitted"], R["node_events"], R["ties_merged"] = len(streams), int(n_ev), int(n_ties)
    if len(streams) == 0:
        print("No nodes to fit; lower --min_events.")
        return R

    # ---- 1. Burstiness ---------------------------------------------------
    bst, all_iets = burstiness_stats(streams)
    R["burstiness"] = bst
    print(f"\n[1] Burstiness (Poisson => B=0): median B={bst['B_median']:.3f}, "
          f"event-weighted B={bst['B_weighted_mean']:.3f}, "
          f"{100 * bst['B_frac_positive']:.1f}% nodes B>0, "
          f"memory coef median={bst['memory_coef_median']:.3f}")

    # ---- 2. Profile likelihood over shared beta --------------------------
    prof_seas = WeeklyProfile(ts, enabled=True)
    prof_flat = WeeklyProfile(ts, enabled=False)
    nd = {"flat": build_node_data(streams, T_end, prof_flat),
          "seasonal": build_node_data(streams, T_end, prof_seas)}
    R["weekly_profile_max_over_min"] = float(prof_seas.s.max() / prof_seas.s.min())

    half_lives = np.geomspace(args.hl_min, args.hl_max, args.n_grid)
    betas = LN2 / half_lives
    beta_hand = -math.log(args.lambda_hand) / args.scale_hand
    Acache = {}

    print(f"\n[2] Profile likelihood over shared beta ({args.n_grid} half-lives "
          f"{fmt_dur(args.hl_min)} .. {fmt_dur(args.hl_max)})"
          f"{'' if HAVE_NUMBA else '  [numba not found: slower]'}")
    res = {}
    for base in ("flat", "seasonal"):
        D = nd[base]
        ll_pois = sum(poisson_ll(d) for d in D.values())
        prof = np.zeros(len(betas))
        node_ids = list(D.keys())
        per_node = np.zeros((len(node_ids), len(betas)))
        warm = None
        for bi, b in enumerate(betas):
            tot, pll, px = eval_beta(D, b, T_end, warm, Acache, key=bi)
            prof[bi] = tot
            per_node[:, bi] = [pll[n] for n in node_ids]
            warm = px
        ib = int(np.argmax(prof))
        # refine beta* continuously between grid neighbours
        lo, hi = math.log(betas[min(ib + 1, len(betas) - 1)]), math.log(betas[max(ib - 1, 0)])
        if hi > lo:
            r = minimize_scalar(lambda lb: -eval_beta(D, math.exp(lb), T_end, warm)[0],
                                bounds=(lo, hi), method="bounded", options={"xatol": 1e-3})
            beta_star = math.exp(r.x)
        else:
            beta_star = betas[ib]
        ll_star, pll_star, px_star = eval_beta(D, beta_star, T_end, warm)
        ll_hand = eval_beta(D, beta_hand, T_end, warm)[0]

        # 95% profile-likelihood CI: solve ll(beta) = ll* - 1.92 on each side of beta*
        edge = ib in (0, len(betas) - 1)
        ci = profile_ci(D, beta_star, ll_star, betas, prof, T_end, warm)

        n_br = np.array([px_star[n][1] for n in node_ids])
        w = np.array([len(D[n]["sk"]) for n in node_ids], dtype=float)
        k_extra = len(node_ids) + 1  # one alpha per node + shared beta
        lr = 2 * (ll_star - ll_pois)
        dAIC = 2 * k_extra - lr  # negative => Hawkes preferred

        # per-node unshared optimum (heterogeneity)
        node_best_hl = half_lives[np.argmax(per_node, axis=1)]
        q = np.quantile(node_best_hl, [0.25, 0.5, 0.75])

        res[base] = dict(
            beta_star=beta_star, half_life_star_s=hl(beta_star),
            half_life_ci95_s=ci, at_grid_edge=bool(edge),
            ll_poisson=ll_pois, ll_hawkes=ll_star, LR_stat=lr, delta_AIC=dAIC,
            ll_hand=ll_hand, delta_ll_hand_vs_star=ll_hand - ll_star,
            branching_median=float(np.median(n_br)),
            branching_weighted_mean=float(np.average(n_br, weights=w)),
            branching_frac_gt1=float((n_br > 1).mean()),
            per_node_half_life_q25_q50_q75_s=[float(v) for v in q],
            profile_half_lives_s=half_lives.tolist(), profile_ll=prof.tolist(),
            _px=px_star, _nodes=node_ids,
        )
        print(f"  [{base:8s}] beta*={beta_star:.3e}/s  half-life*={fmt_dur(hl(beta_star))} "
              f"(95% CI {fmt_dur(ci[0])} .. {fmt_dur(ci[1])}){'  [AT GRID EDGE]' if edge else ''}")
        print(f"             branching n: median={np.median(n_br):.3f}, "
              f"weighted mean={np.average(n_br, weights=w):.3f}, frac n>1={100 * (n_br > 1).mean():.1f}%")
        print(f"             LR vs {base} Poisson = {lr:.1f}, dAIC = {dAIC:.1f} "
              f"({'Hawkes preferred' if dAIC < 0 else 'Poisson preferred'})")
        print(f"             hand-tuned (lambda={args.lambda_hand}, scale={args.scale_hand:.0f}s, "
              f"half-life {fmt_dur(hl(beta_hand))}): dll vs optimum = {ll_hand - ll_star:.1f}")
        print(f"             per-node optimal half-life quartiles: "
              f"{fmt_dur(q[0])} / {fmt_dur(q[1])} / {fmt_dur(q[2])}")

    # ---- 3. Two-time-scale kernel (seasonal baseline) ---------------------
    two = None
    if not args.no_two_scale:
        D = nd["seasonal"]
        idx = list(range(0, len(betas), 2))
        pairs = [(i, j) for a, i in enumerate(idx) for j in idx[a + 2:]]  # >= ~4x apart
        print(f"\n[3] Two-time-scale kernel: {len(pairs)} (beta_fast, beta_slow) pairs on seasonal baseline")
        best = (-np.inf, None, None)
        for (i, j) in pairs:
            tot, px = 0.0, {}
            for nid, d in D.items():
                ll, x = fit_hawkes(d, [Acache[(nid, i)], Acache[(nid, j)]], [betas[i], betas[j]], T_end)
                tot += ll
                px[nid] = x
            if tot > best[0]:
                best = (tot, (i, j), px)
        _, (i, j), px_grid = best

        # continuous refinement of (beta_fast, beta_slow) from the best grid pair
        def two_ll(lb, warm2):
            b1, b2 = math.exp(lb[0]), math.exp(lb[1])
            tot, px = 0.0, {}
            for nid, d in D.items():
                A1 = excitation_sums(d["t"], b1)
                A2 = excitation_sums(d["t"], b2)
                ll, x = fit_hawkes(d, [A1, A2], [b1, b2], T_end, warm2.get(nid))
                tot += ll
                px[nid] = x
            return tot, px

        state = {"px": px_grid}

        def obj(lb):
            tot, px = two_ll(lb, state["px"])
            state["px"] = px
            return -tot

        lb0 = np.array([math.log(betas[i]), math.log(betas[j])])
        r = minimize(obj, lb0, method="Nelder-Mead",
                     options={"maxiter": args.two_scale_iters, "xatol": 1e-2, "fatol": 0.5})
        ll2, px2 = two_ll(r.x, state["px"])
        b_a, b_b = math.exp(r.x[0]), math.exp(r.x[1])
        # order components: fast = larger beta (shorter half-life)
        swap = b_a < b_b
        bf, bs = (b_b, b_a) if swap else (b_a, b_b)
        nodes = list(D.keys())
        n_fast = np.array([px2[n][2 if swap else 1] for n in nodes])
        n_slow = np.array([px2[n][1 if swap else 2] for n in nodes])
        a_fast, a_slow = np.median(n_fast) * bf, np.median(n_slow) * bs
        frac_fast = a_fast / (a_fast + a_slow) if (a_fast + a_slow) > 0 else float("nan")
        ll1 = res["seasonal"]["ll_hawkes"]
        lr2 = 2 * (ll2 - ll1)
        dAIC2 = 2 * (len(nodes) + 1) - lr2
        two = dict(beta_fast=bf, beta_slow=bs, half_life_fast_s=hl(bf), half_life_slow_s=hl(bs),
                   branching_fast_median=float(np.median(n_fast)),
                   branching_slow_median=float(np.median(n_slow)),
                   kernel_weight_fast_at_dt0=float(frac_fast),
                   ll_two_scale=ll2, LR_vs_single=lr2, delta_AIC_vs_single=dAIC2,
                   _px=px2, _betas=(b_a, b_b))
        print(f"  fast half-life={fmt_dur(hl(bf))} (median n={np.median(n_fast):.3f}), "
              f"slow half-life={fmt_dur(hl(bs))} (median n={np.median(n_slow):.3f})")
        print(f"  kernel mass at dt=0: fast {100 * frac_fast:.1f}% / slow {100 * (1 - frac_fast):.1f}%")
        print(f"  LR vs single-exp = {lr2:.1f}, dAIC = {dAIC2:.1f} "
              f"({'two scales preferred' if dAIC2 < 0 else 'single scale sufficient'})")

    # ---- 3b. Three-time-scale kernel: is there a slow (days) component? ---
    three, slow_scan = None, None
    if not args.no_three_scale and two is not None:
        three, slow_scan = three_scale_analysis(nd["seasonal"], two, T_end, prof_seas,
                                                float(ts.min()), beta_hand, args)

    # ---- 4. Goodness of fit (time-rescaling) ------------------------------
    print("\n[4] Time-rescaling goodness of fit (KS distance to Exp(1); smaller is better)")
    gof, taus = {}, {}
    Dflat, Dseas = nd["flat"], nd["seasonal"]
    for label, D, model in (("Poisson (flat)", Dflat, None),
                            ("Poisson (seasonal)", Dseas, None),
                            ("Hawkes (seasonal, 1-exp)", Dseas, "one"),
                            ("Hawkes (seasonal, 2-exp)", Dseas, "two"),
                            ("Hawkes (seasonal, 3-exp)", Dseas, "three")):
        if (model == "two" and two is None) or (model == "three" and three is None):
            continue
        if model == "three":
            bb = three["_betas"]
            parts = [rescaled_times(d, [excitation_sums(d["t"], b) for b in bb], bb,
                                    three["_px"][nid], T_end) for nid, d in D.items()]
            D_ks, tau = ks_exp1(np.concatenate(parts))
            gof[label] = dict(KS_D=D_ks, B_of_rescaled=float(B_of(tau)))
            taus[label] = tau
            print(f"  {label:28s} KS D={D_ks:.4f}   burstiness of rescaled times={B_of(tau):+.3f}")
            continue
        parts = []
        for nid, d in D.items():
            if model is None:
                parts.append(rescaled_times(d, [], [], np.array([1.0]), T_end))  # mu = MLE N/S
            elif model == "one":
                b = res["seasonal"]["beta_star"]
                parts.append(rescaled_times(d, [excitation_sums(d["t"], b)], [b], res["seasonal"]["_px"][nid], T_end))
            else:
                b1, b2 = two["_betas"]
                parts.append(rescaled_times(d, [excitation_sums(d["t"], b1), excitation_sums(d["t"], b2)],
                                            [b1, b2], two["_px"][nid], T_end))
        D_ks, tau = ks_exp1(np.concatenate(parts))
        gof[label] = dict(KS_D=D_ks, B_of_rescaled=float(B_of(tau)))
        taus[label] = tau
        print(f"  {label:28s} KS D={D_ks:.4f}   burstiness of rescaled times={B_of(tau):+.3f}")
    print("  (KS p-values are ~0 for any model at this sample size; compare D across models.)")

    # ---- 5. Recommendation ------------------------------------------------
    b_star = res["seasonal"]["beta_star"]
    h_star = hl(b_star)
    rec = dict(
        primary=dict(lambda_decay=0.5, decay_scale=h_star,
                     note="lambda=0.5 with decay_scale = fitted half-life (identical kernel, readable units)"),
        equivalent_daily_scale=dict(lambda_decay=math.exp(-b_star * 86400.0), decay_scale=86400.0),
        ci95_half_life_s=res["seasonal"]["half_life_ci95_s"],
    )
    if two is not None:
        rec["two_scale_kernel"] = dict(
            formula="w(dt) = f*exp(-b_fast*dt) + (1-f)*exp(-b_slow*dt)",
            f=two["kernel_weight_fast_at_dt0"], b_fast=two["beta_fast"], b_slow=two["beta_slow"],
            note="needs a small code change in the decay heuristic")
    print("\n[5] RECOMMENDED HYPERPARAMETERS (seasonal-baseline MLE, train split)")
    rec["self_excitation_supported"] = bool(res["seasonal"]["delta_AIC"] < 0)
    if res["seasonal"]["delta_AIC"] >= 0:
        print("  !! WARNING: seasonal Poisson is preferred (dAIC >= 0). There is no evidence of\n"
              "  !! self-excitation beyond daily/weekly rhythm, so beta is NOT identified and the\n"
              "  !! values below should not be presented as a Hawkes-derived hyperparameter.")
    elif res["seasonal"]["at_grid_edge"]:
        print("  !! WARNING: optimum at the edge of the half-life grid; widen --hl_min/--hl_max.")
    print(f"  --lambda_decay 0.5 --decay_scale {h_star:.1f}        # half-life {fmt_dur(h_star)}")
    print(f"  equivalently: --lambda_decay {math.exp(-b_star * 86400):.6g} --decay_scale 86400")
    ci = res["seasonal"]["half_life_ci95_s"]
    sens = [v for v in (ci[0], h_star, ci[1]) if np.isfinite(v)]
    print(f"  sensitivity runs (lambda_decay 0.5): decay_scale in {{{', '.join(f'{v:.0f}' for v in sens)}}}"
          f"   # 95% CI ends + optimum")
    if three is not None:
        hs = three["half_life_slow_s"]
        cs = three["half_life_slow_ci95_s"]
        rec["slow_component"] = dict(
            lambda_decay=0.5, decay_scale=hs,
            equivalent_daily_lambda=0.5 ** (86400.0 / hs),
            ci95_half_life_s=cs,
            supported_seasonal=three["slow_supported"],
            survives_drift={w: d["slow_supported"] for w, d in three.get("drift", {}).items()},
            verdict=three.get("verdict"))
        print(f"\n  SLOW (relationship-level) COMPONENT of the 3-scale kernel:")
        if not three["slow_supported"]:
            print("  !! slow component NOT supported (dAIC >= 0) -> do not use it as a hyperparameter")
        elif three["slow_at_grid_edge"]:
            print("  !! slow optimum at the grid edge -> widen --slow_hl_min / --hl_max")
        print(f"  --lambda_decay 0.5 --decay_scale {hs:.0f}        # half-life {fmt_dur(hs)}")
        print(f"  equivalently: --lambda_decay {0.5 ** (86400.0 / hs):.4f} --decay_scale 86400")
        sens = [v for v in (cs[0], hs, cs[1]) if np.isfinite(v)]
        print(f"  sensitivity runs (lambda_decay 0.5): decay_scale in {{{', '.join(f'{v:.0f}' for v in sens)}}}")
    R["recommendation"] = rec

    # ---- 6. Plots ------------------------------------------------------------
    if not args.no_plots:
        try:
            make_plots(out_dir, name, res, half_lives, beta_hand, prof_seas, all_iets, taus)
            print(f"\n[plots] saved to {out_dir}/")
        except Exception as e:
            print(f"[plots] skipped ({e})")
        if slow_scan is not None:
            try:
                plot_slow_scan(out_dir, name, slow_scan, hl(beta_hand))
            except Exception as e:
                print(f"[plots] slow-scan plot skipped ({e})")

    # ---- save ------------------------------------------------------------
    for base in res:
        res[base].pop("_px"), res[base].pop("_nodes")
    if two is not None:
        two.pop("_px"), two.pop("_betas")
    if three is not None:
        three.pop("_px"), three.pop("_betas")
    R["profile"], R["two_scale"], R["goodness_of_fit"] = res, two, gof
    R["three_scale"], R["slow_scan"] = three, slow_scan
    R["hand_tuned"] = dict(lambda_decay=args.lambda_hand, decay_scale=args.scale_hand,
                           half_life_s=hl(beta_hand))
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(R, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"[done] {name} in {time.time() - t0:.1f}s  ->  {out_dir}/summary.json")
    return R


def three_scale_analysis(D, two, T_end, prof, t_origin, beta_hand, args):
    """
    Adds a third, slow exponential component to the two-scale kernel.
      [3b] seasonal baseline: scan the slow half-life, refine all three decay rates
           jointly, profile CI on the slow half-life, compare with the hand-tuned value.
      [3c] drift control: each node's baseline is re-estimated every
           --drift_window_days; the two- and three-scale kernels are re-fitted from
           scratch under that baseline and compared. A slow 'excitation' that is really
           a node becoming more / less active over months is absorbed by the drift.
    """
    from scipy.optimize import brentq

    nodes = list(D.keys())
    w_ev = np.array([len(D[n]["sk"]) for n in nodes], dtype=float)
    slow_hls = np.geomspace(args.slow_hl_min, args.hl_max, args.n_slow_grid)
    k_extra = len(nodes) + 1  # one branching per node + one shared decay

    def total(fitter, betas, warm=None, A_fixed=None):
        """Sum of per-node maximised log-likelihoods for a fixed set of decay rates.
        A_fixed: optional {node: [A for betas[0..k-1]]} to skip recomputing those sums."""
        tot, px = 0.0, {}
        for n in nodes:
            d = D[n]
            if A_fixed is not None:
                A = A_fixed[n] + [excitation_sums(d["t"], b) for b in betas[len(A_fixed[n]):]]
            else:
                A = [excitation_sums(d["t"], b) for b in betas]
            ll, x = fitter(d, A, list(betas), T_end, None if warm is None else warm.get(n))
            tot += ll
            px[n] = x
        return tot, px

    def refine(fitter, betas0, warm):
        state = {"px": warm}

        def obj(lb):
            tot, px = total(fitter, list(np.exp(lb)), state["px"])
            state["px"] = px
            return -tot

        r = minimize(obj, np.log(betas0), method="Nelder-Mead",
                     options={"maxiter": args.two_scale_iters, "xatol": 1e-2, "fatol": 0.5})
        b = np.sort(np.exp(r.x))[::-1]  # fast (largest beta) first
        tot, px = total(fitter, list(b), None)
        return tot, b, px

    def scan(fitter, b_fm):
        A_fm = {n: [excitation_sums(D[n]["t"], b) for b in b_fm] for n in nodes}
        lls, warm, best = [], None, (-np.inf, None, None)
        for h in slow_hls:
            tot, px = total(fitter, list(b_fm) + [LN2 / h], warm, A_fm)
            lls.append(tot)
            warm = px
            if tot > best[0]:
                best = (tot, LN2 / h, px)
        return np.array(lls), best

    def slow_ci(fitter, b_fm, b_slow, ll_ref):
        """Profile CI for the slow half-life with fast/medium fixed: bracket by stepping."""
        A_fm = {n: [excitation_sums(D[n]["t"], b) for b in b_fm] for n in nodes}
        target = ll_ref - 1.92
        g = lambda lb: total(fitter, list(b_fm) + [math.exp(lb)], None, A_fm)[0] - target
        lb0, out = math.log(b_slow), []
        lo_lim, hi_lim = math.log(LN2 / (slow_hls[-1] * 4)), math.log(LN2 / (slow_hls[0] / 4))
        for side in (+1, -1):
            step, lb, found = 0.15, lb0, None
            while lo_lim < lb < hi_lim:
                lb = lb + side * step
                if g(lb) < 0:
                    found = lb
                    break
                step *= 1.6
            if found is None:
                out.append(float("nan"))
                continue
            try:
                out.append(hl(math.exp(brentq(g, *sorted((lb0, found)), xtol=1e-3))))
            except ValueError:
                out.append(float("nan"))
        return (out[0], out[1])  # (short end, long end)

    def analyse_baseline(fitter, b2_init, ll_two_ref, label):
        # two-scale reference under this baseline (re-fitted unless given)
        if ll_two_ref is None:
            ll2, b2, _ = refine(fitter, b2_init, None)
        else:
            ll2, b2 = ll_two_ref, np.asarray(b2_init)
        lls, (ll_s, bs, px_s) = scan(fitter, b2)
        ib = int(np.argmax(lls))
        ll3, b3, px3 = refine(fitter, list(b2) + [bs], px_s)
        if ll3 < ll_s:  # keep the better of scan / refinement
            ll3, b3, px3 = ll_s, np.array(list(b2) + [bs]), px_s
        ll_slow_ref = total(fitter, list(b3), None)[0]
        ci = slow_ci(fitter, b3[:2], b3[2], ll_slow_ref)
        ll_hand = total(fitter, list(b3[:2]) + [beta_hand], None)[0]
        n3 = np.array([px3[n][-3:] for n in nodes])  # branching (fast, medium, slow)
        n_w = np.average(n3, axis=0, weights=w_ev)
        share = n_w / n_w.sum() if n_w.sum() > 0 else np.full(3, np.nan)
        lr = 2 * (ll3 - ll2)
        dAIC = 2 * k_extra - lr
        hin = bool(np.isfinite(ci[0]) and np.isfinite(ci[1]) and ci[0] <= hl(beta_hand) <= ci[1])
        out = dict(
            half_life_fast_s=hl(b3[0]), half_life_medium_s=hl(b3[1]), half_life_slow_s=hl(b3[2]),
            beta_fast=float(b3[0]), beta_medium=float(b3[1]), beta_slow=float(b3[2]),
            half_life_slow_ci95_s=ci, slow_at_grid_edge=bool(ib in (0, len(slow_hls) - 1)),
            branching_median=[float(v) for v in np.median(n3, axis=0)],
            branching_weighted_mean=[float(v) for v in n_w],
            share_of_triggered_events=[float(v) for v in share],
            frac_nodes_slow_branching_gt_0p01=float((n3[:, 2] > 0.01).mean()),
            two_scale_half_lives_s=[hl(b) for b in b2],
            ll_two_scale=float(ll2), ll_three_scale=float(ll3),
            LR_vs_two_scale=float(lr), delta_AIC_vs_two_scale=float(dAIC),
            slow_supported=bool(dAIC < 0),
            delta_ll_hand_as_slow=float(ll_hand - ll_slow_ref), hand_inside_slow_ci95=hin,
        )
        print(f"  [{label}] half-lives: fast {fmt_dur(hl(b3[0]))}, medium {fmt_dur(hl(b3[1]))}, "
              f"slow {fmt_dur(hl(b3[2]))} (slow 95% CI {fmt_dur(ci[0])} .. {fmt_dur(ci[1])})"
              f"{'  [AT GRID EDGE]' if out['slow_at_grid_edge'] else ''}")
        print(f"      branching (event-weighted): fast {n_w[0]:.3f}, medium {n_w[1]:.3f}, slow {n_w[2]:.3f}"
              f"  -> share of triggered events {100 * share[0]:.0f}% / {100 * share[1]:.0f}% / {100 * share[2]:.0f}%")
        print(f"      nodes with slow branching > 0.01: {100 * out['frac_nodes_slow_branching_gt_0p01']:.0f}%")
        print(f"      LR vs two-scale = {lr:.1f}, dAIC = {dAIC:.1f} "
              f"({'slow component supported' if dAIC < 0 else 'no slow component'})")
        print(f"      hand-tuned half-life {fmt_dur(hl(beta_hand))} as the slow component: "
              f"dll vs optimum = {ll_hand - ll_slow_ref:.1f} ({'inside' if hin else 'outside'} the 95% CI)")
        return out, lls, b3, px3

    print(f"\n[3b] Three-time-scale kernel: slow component scanned "
          f"{fmt_dur(slow_hls[0])} .. {fmt_dur(slow_hls[-1])} ({len(slow_hls)} values)")
    three, lls, b3, px3 = analyse_baseline(
        fit_hawkes, [two["beta_fast"], two["beta_slow"]], two["ll_two_scale"], "seasonal baseline")
    three["_px"], three["_betas"] = px3, [float(b) for b in b3]
    scan_out = dict(slow_half_lives_s=slow_hls.tolist(), ll_seasonal=lls.tolist())

    if not args.no_drift_control:
        ll_p_seas = sum(poisson_ll(D[n]) for n in nodes)
        three["drift"] = {}
        for wd in args.drift_window_days:
            add_drift_windows(D, T_end, prof, t_origin, wd * 86400.0)
            print(f"\n[3c] Drift control: each node's baseline re-estimated every {wd:g} days; "
                  f"kernels re-fitted from scratch")
            ll_p_drift = sum(poisson_ll_drift(D[n]) for n in nodes)
            print(f"      activity drift itself: LR drifting vs fixed-baseline Poisson = "
                  f"{2 * (ll_p_drift - ll_p_seas):.1f}")
            drift, lls_d, _, _ = analyse_baseline(fit_hawkes_drift, list(b3[:2]), None,
                                                  f"drift {wd:g} d")
            drift["window_days"] = wd
            drift["LR_drift_poisson_vs_seasonal_poisson"] = float(2 * (ll_p_drift - ll_p_seas))
            three["drift"][f"{wd:g}"] = drift
            scan_out[f"ll_drift_{wd:g}d"] = lls_d.tolist()
        sup = {w: d["slow_supported"] for w, d in three["drift"].items()}
        short_w = min(three["drift"], key=float)
        if not three["slow_supported"]:
            v = "no slow component"
        elif all(sup.values()):
            v = ("slow component SURVIVES baseline drift at every window -> genuine slow "
                 "(relationship-level) self-excitation")
        elif not any(sup.values()):
            v = ("slow component ABSORBED by baseline drift at every window -> the slow 'memory' is "
                 "changing activity levels (non-stationarity), not self-excitation")
        else:
            ok = sorted((float(w) for w, x in sup.items() if x))
            v = (f"AMBIGUOUS: slow component survives drift windows of {', '.join(f'{w:g}' for w in ok)} d "
                 f"but not {short_w} d -> slow self-excitation and month-scale activity drift cannot be "
                 f"separated; describe it as slow recency / activity memory")
        print(f"\n      VERDICT: {v}")
        three["verdict"] = v
    return three, scan_out


def plot_slow_scan(out_dir, name, scan, hand_hl):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = np.array(scan["slow_half_lives_s"]) / 86400
    fig, ax = plt.subplots(figsize=(6, 4))
    keys = [("ll_seasonal", "-", "seasonal baseline")] + [
        (k, "--", f"drifting baseline ({k[len('ll_drift_'):]})") for k in scan if k.startswith("ll_drift_")]
    for key, st, lab in keys:
        if key in scan:
            p = np.array(scan[key])
            ax.plot(h, p - p.max(), st, marker="o", ms=3, label=lab)
    ax.axvline(hand_hl / 86400, c="red", lw=1, label="hand-tuned")
    ax.axhline(-1.92, c="gray", lw=0.8, ls=":", label="95% CI threshold")
    ax.set_xscale("log")
    ax.set_xlabel("slow-component half-life (days)")
    ax.set_ylabel(r"$\ell-\ell_{\max}$")
    ax.set_ylim(bottom=max(ax.get_ylim()[0], -200))
    ax.set_title(f"{name}: slow component of 3-scale kernel")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "slow_component_scan.png"), dpi=150)
    plt.close(fig)


def make_plots(out_dir, name, res, half_lives, beta_hand, prof, iets, taus):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # profile likelihood
    fig, ax = plt.subplots(figsize=(6, 4))
    for base, st in (("flat", "--"), ("seasonal", "-")):
        p = np.array(res[base]["profile_ll"])
        ax.plot(half_lives / 3600, p - p.max(), st, label=f"{base} baseline")
        ax.axvline(res[base]["half_life_star_s"] / 3600, ls=st, lw=0.8, c="gray")
    ax.axvline(hl(beta_hand) / 3600, c="red", lw=1, label="hand-tuned")
    ax.set_xscale("log"); ax.set_ylim(bottom=max(ax.get_ylim()[0], -500))
    ax.set_xlabel("kernel half-life (hours)"); ax.set_ylabel(r"$\ell(\beta)-\ell_{\max}$")
    ax.set_title(f"{name}: profile log-likelihood"); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "profile_likelihood.png"), dpi=150); plt.close(fig)

    # QQ plots
    fig, ax = plt.subplots(figsize=(5, 5))
    p = np.linspace(0.001, 0.999, 999)
    th = stats.expon.ppf(p)
    for lab, tau in taus.items():
        ax.plot(th, np.quantile(tau, p), label=lab, lw=1.2)
    ax.plot(th, th, "k:", lw=1)
    ax.set_xlabel("Exp(1) quantiles"); ax.set_ylabel("rescaled inter-event quantiles")
    ax.set_title(f"{name}: time-rescaling QQ"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "qq_time_rescaling.png"), dpi=150); plt.close(fig)

    # inter-event time distribution
    if len(iets):
        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.geomspace(max(iets.min(), 1.0), iets.max(), 60)
        h, e = np.histogram(iets, bins=bins, density=True)
        c = np.sqrt(e[1:] * e[:-1])
        ax.loglog(c[h > 0], h[h > 0], "o", ms=3, label="empirical")
        lam = 1 / iets.mean()
        ax.loglog(c, lam * np.exp(-lam * c), "--", label="Poisson (same mean)")
        ax.set_xlabel("node inter-event time (s)"); ax.set_ylabel("density")
        ax.set_title(f"{name}: inter-event times"); ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "inter_event_times.png"), dpi=150); plt.close(fig)

    # weekly profile
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.step(np.arange(len(prof.s)), prof.s, where="post")
    ax.set_xlabel("hour of week (relative to dataset t=0)"); ax.set_ylabel("s(t)")
    ax.set_title(f"{name}: estimated seasonal baseline"); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "weekly_profile.png"), dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser("Hawkes self-excitation analysis for CTDG datasets")
    ap.add_argument("-d", "--data", nargs="+", required=True, help="dataset name(s) as used by train_self_supervised.py")
    ap.add_argument("--split", choices=["train", "full"], default="train",
                    help="fit on the train split (default, no leakage) or the full stream")
    ap.add_argument("--min_events", type=int, default=30, help="min events per node to fit")
    ap.add_argument("--keep_ties", action="store_true",
                    help="keep same-timestamp events per node (default merges them: multi-recipient "
                         "emails otherwise look like infinite instantaneous excitation)")
    ap.add_argument("--n_grid", type=int, default=40, help="number of half-lives in the beta grid")
    ap.add_argument("--hl_min", type=float, default=60.0, help="smallest half-life (s)")
    ap.add_argument("--hl_max", type=float, default=60 * 86400.0, help="largest half-life (s)")
    ap.add_argument("--lambda_hand", type=float, default=0.9, help="current hand-tuned lambda_decay")
    ap.add_argument("--scale_hand", type=float, default=86400.0, help="current hand-tuned decay_scale")
    ap.add_argument("--no_two_scale", action="store_true", help="skip the two-time-scale kernel fit")
    ap.add_argument("--two_scale_iters", type=int, default=40,
                    help="Nelder-Mead iterations when refining the two-scale kernel")
    ap.add_argument("--no_three_scale", action="store_true", help="skip the three-time-scale kernel fit")
    ap.add_argument("--slow_hl_min", type=float, default=6 * 3600.0,
                    help="smallest half-life (s) scanned for the slow component")
    ap.add_argument("--n_slow_grid", type=int, default=16, help="number of slow half-lives scanned")
    ap.add_argument("--no_drift_control", action="store_true", help="skip the drifting-baseline control")
    ap.add_argument("--drift_window_days", type=float, nargs="+", default=[30.0, 90.0],
                    help="window length(s) in days for the per-node drifting baseline; short windows are "
                         "conservative (can absorb real slow excitation), long ones permissive")
    ap.add_argument("--no_plots", action="store_true")
    ap.add_argument("--outdir", default="hawkes_results")
    args = ap.parse_args()

    all_R = [analyse(name, args) for name in args.data]

    print(f"\n{'=' * 78}\nSUMMARY (seasonal baseline)")
    print(f"{'dataset':16s} {'B':>6s} {'half-life*':>11s} {'95% CI':>22s} {'n_med':>6s} "
          f"{'dAIC/node':>9s} {'KS P->H':>14s} {'lambda@1d':>10s}")
    for R in all_R:
        if "profile" not in R:
            continue
        s = R["profile"]["seasonal"]
        g = R["goodness_of_fit"]
        ci = s["half_life_ci95_s"]
        print(f"{R['dataset']:16s} {R['burstiness']['B_weighted_mean']:6.3f} "
              f"{fmt_dur(s['half_life_star_s']):>11s} {fmt_dur(ci[0]) + '..' + fmt_dur(ci[1]):>22s} "
              f"{s['branching_median']:6.3f} {s['delta_AIC'] / R['n_nodes_fitted']:9.1f} "
              f"{g['Poisson (seasonal)']['KS_D']:.3f}->{g['Hawkes (seasonal, 1-exp)']['KS_D']:.3f} "
              f"{R['recommendation']['equivalent_daily_scale']['lambda_decay']:10.4g}")

    if any(R.get("three_scale") for R in all_R):
        print(f"\nSLOW COMPONENT (3-scale kernel)")
        print(f"{'dataset':16s} {'fast':>9s} {'medium':>9s} {'slow':>9s} {'slow 95% CI':>20s} "
              f"{'share slow':>10s} {'dAIC':>9s} {'lambda@1d':>9s}  drift windows: slow half-life (dAIC)")
        for R in all_R:
            t3 = R.get("three_scale")
            if not t3:
                continue
            ci = t3["half_life_slow_ci95_s"]
            dr = "  ".join(f"{w}d: {fmt_dur(d['half_life_slow_s'])} ({d['delta_AIC_vs_two_scale']:.0f})"
                           for w, d in t3.get("drift", {}).items())
            print(f"{R['dataset']:16s} {fmt_dur(t3['half_life_fast_s']):>9s} {fmt_dur(t3['half_life_medium_s']):>9s} "
                  f"{fmt_dur(t3['half_life_slow_s']):>9s} {fmt_dur(ci[0]) + '..' + fmt_dur(ci[1]):>20s} "
                  f"{100 * t3['share_of_triggered_events'][2]:9.0f}% {t3['delta_AIC_vs_two_scale']:9.1f} "
                  f"{0.5 ** (86400.0 / t3['half_life_slow_s']):9.4f}  {dr}")
        for R in all_R:
            if R.get("three_scale"):
                print(f"  {R['dataset']}: {R['three_scale'].get('verdict', '')}")


if __name__ == "__main__":
    main()