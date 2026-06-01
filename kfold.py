from collections import defaultdict
from itertools import combinations
import numpy as np
from scipy.stats import kendalltau
from utils import OfflineBTRanker

def _theta_vector(bt_model, old_models):
    return np.array([bt_model.theta[bt_model.idx[m]] for m in old_models], dtype=float)

def _split_queries_kfold(qids, n_folds, rng):
    qids = list(qids)
    rng.shuffle(qids)
    folds = [set() for _ in range(n_folds)]
    for t, q in enumerate(qids):
        folds[t % n_folds].add(q)

    folds = [f for f in folds if len(f) > 0]
    return folds

def kfold_consistency_single_cluster(votes_cluster, old_models, n_folds=5, 
                                     random_state=0, min_votes_per_fold=5):
 
    rng = np.random.default_rng(random_state)

    votes_by_q = defaultdict(list)
    for q, i, j, y in votes_cluster:
        votes_by_q[q].append((i, j, y))

    qids = list(votes_by_q.keys())
    if len(qids) == 0:
        return {'avg_tau': np.nan, 'tau_matrix': np.array([[]]),
                'fold_models': [], 'n_queries': 0, 'n_votes': 0, 'n_folds_used': 0}
    Kf = min(n_folds, len(qids))
    folds = _split_queries_kfold(qids, Kf, rng)

    fold_models, fold_sizes = [], []
    for fset in folds:
        fold_votes, weights = [], []
        for q in fset:
            Nq = max(1, len(votes_by_q[q]))
            for (i, j, y) in votes_by_q[q]:
                fold_votes.append((q, [[i, j]], [y]))
                weights.append(1.0 / Nq)
        if len(fold_votes) < min_votes_per_fold:
            fold_models.append(None)
            fold_sizes.append(0)
            continue
        ranker = OfflineBTRanker(models=old_models)
        ranker.fit(votes=fold_votes, existing_bt=None, weights=weights)

        fold_models.append(ranker)
        fold_sizes.append(len(fold_votes))

    valid = [(m, sz) for m, sz in zip(fold_models, fold_sizes) if m is not None]
    
    if len(valid) < 2:
        return {'avg_tau': np.nan, 'tau_matrix': np.array([[]]),
                'fold_models': [m for m,_ in valid],
                'n_queries': len(qids), 'n_votes': sum(len(v) for v in votes_by_q.values()),
                'n_folds_used': len(valid)}
    
    fold_models = [m for m,_ in valid]
    Kf_used = len(fold_models)

    thetas = np.vstack([_theta_vector(m, old_models) for m in fold_models])
    tau_mat = np.full((Kf_used, Kf_used), np.nan, dtype=float)
    for i, j in combinations(range(Kf_used), 2):
        tau = kendalltau(thetas[i], thetas[j], variant='b').correlation
        tau_mat[i, j] = tau
        tau_mat[j, i] = tau

    avg_tau = float(np.nanmean(tau_mat))
    return {
        'avg_tau': avg_tau,
        'tau_matrix': tau_mat,
        'fold_models': fold_models,
        'n_queries': len(qids),
        'n_votes': sum(len(v) for v in votes_by_q.values()),
        'n_folds_used': Kf_used
    }

def kfold_consistency_all_clusters(train_votes, z, old_models, n_folds=5, random_state=0,
                                   weight='queries'):
    cluster_votes = defaultdict(list)
    for q, models, ys in train_votes:
        for (i, j), y in zip(models, ys):
            k = z.get(q, None)
            if k is None: 
                continue
            cluster_votes[k].append((q, i, j, y))

    per_cluster = {}
    weights = []
    taus = []
    for k, vlist in cluster_votes.items():
        res = kfold_consistency_single_cluster(
            vlist, old_models, n_folds=n_folds, random_state=random_state + k
        )
        per_cluster[k] = res
        taus.append(res['avg_tau'])
        if weight == 'queries':
            weights.append(res['n_queries'])
        elif weight == 'votes':
            weights.append(res['n_votes'])
        else:
            weights.append(1.0)

    taus = np.array(taus, dtype=float)
    weights = np.array(weights, dtype=float)
    
    mask = np.isfinite(taus) & (weights > 0)
    if np.any(mask):
        overall = float(np.sum(taus[mask] * weights[mask]) / (np.sum(weights[mask]) + 1e-12))
    else:
        overall = np.nan

    return {'per_cluster': per_cluster, 'overall_avg_tau': overall}



def kfold_consistency_all_clusters_repeated(
    train_votes, z, old_models, n_folds=5, repeats=10, base_seed=0, weight='queries'
):
    per_cluster_hist = {}  
    overall_hist = []      

    for r in range(repeats):
        res = kfold_consistency_all_clusters(
            train_votes, z, old_models,
            n_folds=n_folds, random_state=base_seed + r, weight=weight
        )
        overall_hist.append(res['overall_avg_tau'])
        for k, kre in res['per_cluster'].items():
            per_cluster_hist.setdefault(k, []).append(kre['avg_tau'])

    def _mean_ci(xs):
        xs = np.array([x for x in xs if np.isfinite(x)], dtype=float)
        if len(xs) == 0:
            return np.nan, np.nan, (np.nan, np.nan), 0
        mu = float(np.mean(xs))
        sd = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
        se = sd / max(1, np.sqrt(len(xs)))
        ci = (mu - 1.96*se, mu + 1.96*se)
        return mu, sd, ci, len(xs)

    per_cluster_summary = {}
    for k, xs in per_cluster_hist.items():
        mu, sd, ci, n = _mean_ci(xs)
        per_cluster_summary[k] = {
            'avg_tau_mean': mu,
            'avg_tau_std': sd,
            'avg_tau_ci95': ci,
            'n_repeats_used': n
        }

    overall_mu, overall_sd, overall_ci, n_used = _mean_ci(overall_hist)

    return {
        'per_cluster': per_cluster_summary,
        'overall': {
            'avg_tau_mean': overall_mu,
            'avg_tau_std': overall_sd,
            'avg_tau_ci95': overall_ci,
            'n_repeats_used': n_used
        }
    }