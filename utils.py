
from collections import defaultdict
import json, math
import numpy as np
import pandas as pd
import argparse
import yaml, os
import math, collections
from typing import Dict, List, Tuple, Iterable, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.stats import kendalltau, spearmanr
from collections import defaultdict, Counter, deque
from scipy.stats import kendalltau
from itertools import combinations
import random
  

L2_REG   = 1e-2              
EPS = 1e-12

class OfflineBTRanker:
    def __init__(self, models: Iterable[str]):

        self.models: List[str] = models
        self.M: int = len(self.models)
        self.idx: Dict[str, int] = {m: i for i, m in enumerate(self.models)}
        self.theta: np.ndarray = np.zeros(self.M)    
        self.T: int = 0

    def fit(self, votes, existing_bt = None, l2_reg: float = L2_REG, weights = None):
        X_list, y_list, weights_list = [], [], []

        for idx, (_, models, ys) in enumerate(votes):
            for (model_a, model_b), y in zip(models, ys):
        
                if model_a not in self.idx or model_b not in self.idx:
                    continue

                w = weights[idx] if weights is not None else 1.0
                
                idx_a, idx_b = self.idx[model_a], self.idx[model_b]
                x = np.zeros(self.M)
                x[idx_a], x[idx_b] = -1.0, 1.0 
                
                if y == 1.0:
                    X_list.append(x); y_list.append(1); weights_list.append(w)
                elif y == 0.0: 
                    X_list.append(x); y_list.append(0); weights_list.append(w)
                elif y == 0.5: 
                    X_list.append(x); y_list.append(1); weights_list.append(0.5 * w)
                    X_list.append(x); y_list.append(0); weights_list.append(0.5 * w)


        X = np.array(X_list)
        y = np.array(y_list)
        sample_weights = np.array(weights_list)
        self.T = X.shape[0]

        if self.T == 0:
            return
        
        unique_y_classes = np.unique(y)
        if len(unique_y_classes) < 2:
            return

        C = 1.0 / l2_reg if l2_reg > 0 else 1e9
        if existing_bt is not None:
            pass
        lr = LogisticRegression(fit_intercept=False, solver='lbfgs')
        lr.fit(X, y, sample_weight=sample_weights)
        
        theta_e_scale = lr.coef_[0]
        self.theta = theta_e_scale


    def predict_pairwise(self, i: str, j: str) -> float:
        """Probability that model j is preferred over model i"""
        return 1 / (1 + math.exp(-(self.theta[self.idx[j]] - self.theta[self.idx[i]])))

    def score_rank(self):
        ranks: Dict[str, int] = {}
        for m, idx in self.idx.items():
            ranks[m] = int(np.sum(self.theta > self.theta[idx])) + 1
        
        return ranks

    def base_e_scores(self) -> Dict[str, float]:
        return {m: self.theta[self.idx[m]] for m in self.models}
    


def bt_pairwise_stats(bt_models, old_models, show_matrix=True):
    K = len(bt_models)
    M = len(old_models)
    thetas = np.array([
        [bt.theta[bt.idx[m]] for m in old_models]
        for bt in bt_models
    ], dtype=float)

    stats = {}
    pairs_tau = []

    for i, j in combinations(range(K), 2):
        tau = kendalltau(thetas[i], thetas[j]).correlation
        stats[(i, j)] = {'tau': float(tau)}
        pairs_tau.append(np.nan if tau is None else float(tau))

    pairs_tau = np.array(pairs_tau, dtype=float)
    mean_tau = float(np.nanmean(pairs_tau)) if pairs_tau.size else np.nan
    return mean_tau, pairs_tau

def norm_vec(x): 
    n = np.linalg.norm(x)
    return x / max(n, 1e-12)