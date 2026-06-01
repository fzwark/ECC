import numpy as np
import math
from collections import defaultdict

EPS = 1e-12

def _nll_for_votes(pred_fn, votes, average="vote"):
    if not votes:
        return np.nan
    if average == "vote":
        losses = []
        for _, i, j, y in votes:
            p = np.clip(pred_fn(i, j), EPS, 1 - EPS)
            losses.append(-(y * math.log(p) + (1 - y) * math.log1p(-p)))
        
        return float(np.mean(losses))
    elif average == "query":
        by_q = defaultdict(list)
        for q, i, j, y in votes:
            by_q[q].append((i, j, y))
        q_losses = []
        for q, pairs in by_q.items():
            l = 0.0
            for i, j, y in pairs:
                p = np.clip(pred_fn(i, j), EPS, 1 - EPS)
                l += -(y * math.log(p) + (1 - y) * math.log1p(-p))

            q_losses.append(l / len(pairs))
        return float(np.mean(q_losses))
    else:
        raise ValueError("undefined")



def diag_gap_per_cluster(loss, counts, how="mean"):
    K = loss.shape[0]
    gaps, ws = [], []
    for b in range(K):
        col = loss[:, b]
        diag = col[b]
        others = np.delete(col, b)
        others = others[np.isfinite(others)]
        if not np.isfinite(diag) or others.size == 0:
            continue
        opp = others.mean() if how == "mean" else others.min()
        gaps.append(opp - diag)                    
        ws.append(counts[b].get("n_votes", 0) or 0) 
    gaps = np.asarray(gaps, float)
    ws   = np.asarray(ws, float)
    return gaps, ws

def cross_cluster_prediction_matrix(
    bt_models,              
    z,                      
    test_votes,             
    K,                     
    global_bt=None,        
    average="vote"         
):
    votes_by_cluster = defaultdict(list)
    queries_by_cluster = defaultdict(set)
    for q, models, ys in test_votes:
        for (i, j), y in zip(models, ys):
            b = z.get(q, None)
            if b is None:
                continue
            votes_by_cluster[b].append((q, i, j, y))
            queries_by_cluster[b].add(q)

    pred_fns = [m.predict_pairwise for m in bt_models]
    pred_g = global_bt.predict_pairwise if global_bt is not None else None

    loss = np.full((K, K), np.nan, dtype=float)
    global_loss = np.full(K, np.nan, dtype=float) if global_bt is not None else None
    counts = {}

    for b in range(K):
        vlist = votes_by_cluster.get(b, [])
        counts[b] = {
            'n_votes': len(vlist),
            'n_queries': len(queries_by_cluster.get(b, set()))
        }
        if global_bt is not None:
            global_loss[b] = _nll_for_votes(pred_g, vlist, average=average)
        for a in range(K):
            tmp = _nll_for_votes(pred_fns[a], vlist, average=average)
            
            loss[a, b] = tmp

    gaps, ws = diag_gap_per_cluster(loss, counts)

    return {
        "gaps": gaps,   
    }