from data import Data
from utils import *
from sklearn.model_selection import train_test_split
from kfold import kfold_consistency_all_clusters_repeated
from cross import cross_cluster_prediction_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--pairs', type=int, default=7)
parser.add_argument('--k', type=int, default=30)
parser.add_argument('--nnn', type=int, default=1)
parser.add_argument('--lam', type=float, default=3)
parser.add_argument('--data', type=str, default="sprout")
parser.add_argument('--cc', type=int, default=1)
parser.add_argument('--ec', type=int, default=1)
parser.add_argument('--tp', type=float, default=0.2)
parser.add_argument('--T', type=float, default=0.1)

args = parser.parse_args()
K = args.k 
pairs = args.pairs
nnn = args.nnn
lambda_val = args.lam
cc = args.cc
ec = args.ec
tp = args.tp 

T = args.T


# load the data
router_bench_models= [
    'WizardLM/WizardLM-13B-V1.2', 'claude-instant-v1', 'claude-v1', 'claude-v2', 'gpt-3.5-turbo-1106', 'gpt-4-1106-preview', 'meta/code-llama-instruct-34b-chat', 'meta/llama-2-70b-chat', 'mistralai/mistral-7b-chat', 'mistralai/mixtral-8x7b-chat', 'zero-one-ai/Yi-34B-Chat'
]
sprout_models = [
    "aws-claude-3-5-sonnet-v1", "aws-titan-text-premier-v1", "openai-gpt-4o", "openai-gpt-4o-mini", "wxai-granite-3-2b-instruct-8k-max-tokens",
    "wxai-granite-3-8b-instruct-8k-max-tokens", "wxai-llama-3-1-70b-instruct", "wxai-llama-3-1-8b-instruct", "wxai-llama-3-2-1b-instruct", 
    "wxai-llama-3-2-3b-instruct", "wxai-llama-3-3-70b-instruct", "wxai-llama-3-405b-instruct", "wxai-mixtral-8x7b-instruct-v01"
]

leader_models = ['01-ai/Yi-34B-Chat', 'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO', 'Qwen/QwQ-32B-Preview', 'Qwen/Qwen2-72B-Instruct', 'Qwen/Qwen2.5-72B-Instruct', 
      'Qwen/Qwen2.5-7B-Instruct', 'alpindale/WizardLM-2-8x22B', 'deepseek-ai/deepseek-llm-67b-chat', 'google/gemma-2-27b-it', 'google/gemma-2-9b-it', 
      'google/gemma-2b-it', 'meta-llama/Llama-2-13b-chat-hf', 'meta-llama/Meta-Llama-3.1-70B-Instruct', 'mistralai/Mistral-7B-Instruct-v0.1', 
      'mistralai/Mistral-7B-Instruct-v0.2', 'mistralai/Mistral-7B-Instruct-v0.3', 'mistralai/Mixtral-8x7B-Instruct-v0.1', 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF']


if args.data == "sprout":
    input_models = sprout_models
elif args.data == "leaderboard":
    input_models = leader_models
else:
    input_models = router_bench_models

data = Data(models=input_models)
data.load_raw_data(data=args.data)
old_data, all_models = data.load_data(pairs=pairs,data=args.data)

cols = ["index", "pair_models", "pair_winner", "prompt"]  
all_votes = list(zip(*(old_data[c] for c in cols)))

qids = sorted({q for q, _, _, _ in all_votes})
qidtoprompt = {q : prompt for q, _, _, prompt in all_votes}
all_votes = [item[:-1] for item in all_votes]

qids_tr, qids_te = train_test_split(qids, test_size=tp, shuffle=True, random_state=42)

train_votes = [v for v in all_votes if v[0] in qids_tr]
test_raw = [v for v in all_votes if v[0] in qids_te]
query_ids = {q for q, _, _ in train_votes}

test_votes  = [(qid, models[:nnn], ys[:nnn]) for (qid, models, ys) in test_raw]
test_votes_real  = [(qid, models[nnn:], ys[nnn:]) for (qid, models, ys) in test_raw]

if args.data == "sprout":
    embeddings = np.load("bge_embeddings0.npy")
elif args.data == "leaderboard":
    embeddings = np.load("bge_embeddings2.npy") 
else:
    embeddings = np.load("bge_embeddings1.npy")



EPS = 1e-12
def _vote_loss(model, i, j, y):
    p = model.predict_pairwise(i, j)
    p = min(max(p, EPS), 1 - EPS)
    if y == 1.0:
        return -math.log(p)
    elif y == 0.0:
        return  -math.log1p(-p)
    
def logloss_single(test_votes):
    total, n = 0.0, 0
    for _, models, ys in test_votes:
        for (i, j), y in zip(models, ys):
            total += _vote_loss(offline_gt, i, j, y)
            n += 1
    return total / max(1, n)


# global
offline_gt = OfflineBTRanker(models=all_models)
offline_gt.fit(votes=train_votes, existing_bt=None)


seed = 0
max_iter = int(1e5)

rng = np.random.default_rng(seed=seed)
query_ids = {q for q, _, _ in train_votes}
resp = {q: rng.dirichlet(np.ones(K)) for q in query_ids} 
pi = np.ones(K) / K

embedding_dim = embeddings.shape[1]
qs = [q for q in qids_tr]
E = np.stack([embeddings[q] for q in qs], axis=0)   # (Q', D)
norms = np.linalg.norm(E, axis=1, keepdims=True)
E = E / np.maximum(norms, 1e-12)

def l2norm(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


R = np.stack([resp[q] for q in qs], axis=0) # (Q', K)
weights = R.sum(axis=0)                          
denom = np.clip(weights[:, None], 1e-12, None)    
centroids = (R.T @ E) / denom
centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)

beta = lambda_val / T

CONV_TOL_C = 1e-4   
PRINT_EVERY = 5
PATIENCE = 5    

no_move_rounds = 0

for it in range(max_iter):
    S = E @ centroids.T                        
    A = beta * S
    A -= A.max(axis=1, keepdims=True)          
    P = np.exp(A)
    P /= P.sum(axis=1, keepdims=True)         
    prev_centroids = centroids.copy()
    R = P
    weights = R.sum(axis=0)                                   
    centroids = (R.T @ E) / np.clip(weights[:, None], 1e-12, None)
    centroids = l2norm(centroids)

    c_shift = float(np.linalg.norm(centroids - prev_centroids))

    if it % PRINT_EVERY == 0:
        entropy = (-R * np.log(np.clip(R, 1e-12, None))).sum(axis=1).mean()
        print(f"[iter {it}] beta={beta:.2f} | c_shift={c_shift:.4e} | avg_entropy(R)={entropy:.4f}")

    if c_shift < CONV_TOL_C:
        no_move_rounds += 1
    else:
        no_move_rounds = 0

    if no_move_rounds >= PATIENCE:
        print(f"[EM] converged at iter {it} | c_shift={c_shift:.4e} | beta={beta:.2f}")
        break


resp = {q: R[i] for i, q in enumerate(qs)}

bt_models = []
for k in range(K):
    weighted_votes = []
    weights = []
    for q, models, ys in train_votes:
        Nq = max(1, len(models))
        for (i, j), y in zip(models, ys):
            w = resp[q][k] / Nq  
            if w < 1e-12:         
                continue
            weighted_votes.append((q, [[i, j]], [y]))
            weights.append(w)
    ranker = OfflineBTRanker(models=all_models)
    ranker.fit(votes=weighted_votes,
                existing_bt=None,
                weights=weights)      
    bt_models.append(ranker)

query_ids = {q for q, _, _ in train_votes}
labels = {q: int(np.argmax(resp[q])) for q in query_ids}

with open(f"clusters/{args.data}_emb_{K}_{pairs}_{lambda_val}_{cc}_{ec}_{tp}_cmp.json", "w+") as f:
    json.dump(labels, f)
    

clustertoprompts = defaultdict(list)
for q, l in labels.items():
    if q in qidtoprompt:        
        clustertoprompts[l].append(qidtoprompt[q])

# in-cluster consistency
mean_tau, pairs_tau = bt_pairwise_stats(bt_models, list(all_models))
res_all = kfold_consistency_all_clusters_repeated(train_votes, labels, all_models, n_folds=5, weight='queries')


prompts_by_cluster = defaultdict(list)
for query_id, cluster_id in labels.items():
    embedding = embeddings[query_id]
    prompts_by_cluster[cluster_id].append(embedding.tolist())

def logloss_softmix(bt_models, centroids, embeddings, test_votes, eps=1e-12):
    preds = [m.predict_pairwise for m in bt_models]
    tot, n = 0.0, 0
    for q, models, ys in test_votes:
        for (i, j), y in zip(models, ys):
            p = 0.0
            e = norm_vec(embeddings[q]) 
            sims = e @ centroids.T
            logits = sims * beta
            logits = logits 
            logits -= logits.max()
            r = np.exp(logits); 
            r /= r.sum() + eps
    
            for k in range(len(bt_models)):
                pk = preds[k](i, j)
                pk = min(max(pk, eps), 1 - eps)
                p += r[k] * pk
            p = min(max(p, eps), 1 - eps)
            delta = -(y * math.log(p) + (1 - y) * math.log1p(-p))
            tot += delta
            n += 1
    return tot / n


def nll_soft_with_resp(bt_models, resp, train_votes, eps=1e-12):
    preds = [m.predict_pairwise for m in bt_models]
    tot, n = 0.0, 0
    for q, models, ys in train_votes:
        r = np.clip(resp[q], eps, 1.0); r /= r.sum()
        for (i, j), y in zip(models, ys):
            p = 0.0
            for k in range(len(bt_models)):
                pk = np.clip(preds[k](i, j), eps, 1-eps)
                p += r[k] * pk
            p = np.clip(p, eps, 1-eps)
            tot += -(y*math.log(p) + (1-y)*math.log1p(-p))
            n += 1
    return tot / max(1, n)

def compute_resp_for_test(bt_models, pi, test_query_ids):
    K = len(bt_models)
    preds = [m.predict_pairwise for m in bt_models]

    votes_by_q = defaultdict(list)
    for q, models, ys in test_votes:
        for (i, j), y in zip(models, ys):
            votes_by_q[q].append((i, j, y))

    resp_test = {}
    for q in test_query_ids:
        ll = np.zeros(K)
        e = norm_vec(embeddings[q])
        sims = e @ centroids.T
        ll_embed = lambda_val * sims
        
        for k in range(K):
            s = 0.0
            for i, j, y in votes_by_q[q]:
                p = preds[k](i, j)
                p = min(max(p, EPS), 1-EPS)
                if y == 1.0:
                    s += math.log(p)
                elif y == 0.0:
                    s += math.log1p(-p)

            ll[k] = s/len(votes_by_q[q])
        if ec == 0:
            ll_total = (ll)/T
        elif ec == 1:
            ll_total = (ll + ll_embed)/T
        log_post = ll_total 
        log_post -= log_post.max()          
        r = np.exp(log_post)
        r /= r.sum()
        resp_test[q] = r

    return resp_test


loss_global = logloss_single(test_votes_real)
resp_test = compute_resp_for_test(bt_models, pi, {q for q, *_ in test_votes})
query_ids = {q for q, _, _ in test_votes}
labels = {q: int(np.argmax(resp_test[q])) for q in query_ids}

loss_res = nll_soft_with_resp(bt_models, resp_test, test_votes_real)
loss_emb = logloss_softmix(bt_models, centroids, embeddings, test_votes_real, eps=1e-12)

res = cross_cluster_prediction_matrix(bt_models, labels, test_votes_real, K, global_bt=offline_gt, average="vote")


ans = {
    "global" : loss_global,
    "clustering_with_probe" : loss_res,
    "clustering_emb_only" : loss_emb,
    "drop" : (loss_global - loss_res)/loss_global,
    'mean_tau' : mean_tau,
    "pairs_tau" : pairs_tau,
    "within-cluster avg tau" : res_all,
    "cross-cluster avg loss" : res
}


with open(f"clusters/{args.data}_emb_{K}_{pairs}_{lambda_val}_{cc}_{ec}_{tp}_prompts.json", "w+") as f:
    json.dump(clustertoprompts, f)


with open(f"clusters/{args.data}_emb_{K}_{pairs}_{lambda_val}_{cc}_{ec}_{tp}_embeddings.json", "w+") as f:
    json.dump(prompts_by_cluster, f)


def default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.int32, np.int64)):
        return int(o)
    return str(o)

with open(f"results/{args.data}_emb_{K}_{pairs}_{lambda_val}_{cc}_{ec}_{tp}.json", "w+") as f:
    json.dump(ans, f, default=default)