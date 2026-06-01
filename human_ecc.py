from data_human import Data
from utils import *
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--pairs', type=int, default=7)
parser.add_argument('--nnn', type=int, default=1)
parser.add_argument('--lam', type=float, default=3)
parser.add_argument('--data', type=str, default="mmlu")
parser.add_argument('--cc', type=int, default=1)
parser.add_argument('--ec', type=int, default=1)
parser.add_argument('--tp', type=float, default=0.2)


args = parser.parse_args()
pairs = args.pairs
nnn = args.nnn
cc = args.cc
ec = args.ec
tp = args.tp 

if args.data == "mmlu":
    K = 57
elif args.data == "mmlupro":
    K = 14
elif args.data == "math":
    K = 7

T = 0.1
lambda_val = args.lam

if cc == 0 :
    nm = "comp"
else:
    nm = "ecc"

CONV_TOL_KL   = 1e-3     
CONV_TOL_C    = 1e-4     
CONV_TOL_REL  = 1e-4    
CONV_PATIENCE = 5 


print(f"clusters: {K}")

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


if args.data == "mmlupro" or args.data == "math":
    input_models = sprout_models
elif args.data == "mmlu":
    input_models = router_bench_models


data = Data(models=input_models)
data.load_raw_data(data=args.data)
old_data, all_models, _ = data.load_data(pairs=pairs,data=args.data)
cols = ["index", "pair_models", "pair_winner", "prompt"]  
all_votes = list(zip(*(old_data[c] for c in cols)))

qids = sorted({q for q, _, _, _ in all_votes})
qidtoprompt = {q : prompt for q, _, _, prompt in all_votes}
# delete prompt column
all_votes = [item[:-1] for item in all_votes]

qids_tr, qids_te = train_test_split(qids, test_size=tp, shuffle=True, random_state=42)

train_votes = [v for v in all_votes if v[0] in qids_tr]
test_raw = [v for v in all_votes if v[0] in qids_te]
query_ids = {q for q, _, _ in train_votes}


test_votes  = [(qid, models[:nnn], ys[:nnn]) for (qid, models, ys) in test_raw]
test_votes_real  = [(qid, models[nnn:], ys[nnn:]) for (qid, models, ys) in test_raw]


print(len(train_votes), len(test_votes))

if args.data == "mmlupro" or args.data == "math":
    embeddings = np.load("bge_embeddings0.npy")
elif args.data == "mmlu":
    embeddings = np.load("bge_embeddings1.npy")


def nll_soft_with_resp(bt_models, resp, votes, eps=1e-12):
    preds = [m.predict_pairwise for m in bt_models]
    tot, n = 0.0, 0
    for q, models, ys in votes:
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
resp = {q: rng.dirichlet(np.ones(K)) for q in query_ids} 
pi = np.ones(K) / K

def accumulate_loglik(model):
    acc = defaultdict(float)
    pred = model.predict_pairwise   
    log = math.log
    for q, models, ys in train_votes:
        Nq = max(1, len(models))
        for (mi, mj), y in zip(models, ys):
            p = pred(mi, mj)
            if y == 1.0:
                acc[q] += log(p)
            elif y == 0.0:
                acc[q] += log(1 - p)
            
        acc[q] /= Nq
    return acc


def _em_lower_bound(loglik, embeddings, centroids, pi, T, lam, qids):
    K = len(loglik)
    total = 0.0
    for q in qids:
        e = norm_vec(embeddings[q])
        sims = e @ centroids.T                    
        v = np.array([loglik[k][q] for k in range(K)], dtype=float)
        v = (v + lam * sims) / max(T, EPS)
        m = v.max()
        total += m + np.log(np.exp(v - m).sum())  # logsumexp
    return float(total)

def _mean_kl(old, new):
    s = 0.0
    for q in old.keys():
        p = np.clip(old[q], EPS, 1.0); p /= p.sum()
        qv= np.clip(new[q], EPS, 1.0); qv/= qv.sum()
        s += np.sum(p * (np.log(p) - np.log(qv)))
    return s / max(1,len(old))


def compute_resp_for_test(bt_models, test_query_ids, centroids):
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
            try:
                ll[k] = s/len(votes_by_q[q])
            except:
                print(len(votes_by_q[q]))
                print((votes_by_q[q]))
        
        if ec == 1:
            ll_total = (ll + ll_embed)/T
        elif ec == 0:
            ll_total = ll/T
        log_post = ll_total 
        log_post -= log_post.max()          
        r = np.exp(log_post)
        r /= r.sum()
        resp_test[q] = r

    return resp_test


embedding_dim = embeddings.shape[1]
centroids = np.zeros((K, embedding_dim))
qs = [q for q in qids_tr]
E = np.stack([embeddings[q] for q in qs], axis=0)  
norms = np.linalg.norm(E, axis=1, keepdims=True)
E = E / np.maximum(norms, 1e-12)
R = np.stack([resp[q] for q in qs], axis=0) 
weights = R.sum(axis=0)                          
denom = np.clip(weights[:, None], 1e-12, None)    
centroids = (R.T @ E) / denom
centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)


def norm_vec(x): 
    n = np.linalg.norm(x)
    return x / max(n, 1e-12)

for it in range(max_iter):
    prev_centroids = centroids.copy()
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

    new_centroids = np.zeros_like(centroids)
    total_weights = np.zeros(K)
    for q in query_ids:
        e_q = norm_vec(embeddings[q])
        r_q = resp[q]
        new_centroids += r_q[:, None] * e_q[None, :]
        total_weights += r_q
    for k in range(K):
        if total_weights[k] > 1e-9:
            centroids[k] = new_centroids[k] / total_weights[k]

    centroids = np.vstack([norm_vec(c) for c in centroids])

    loglik = [accumulate_loglik(m) for m in bt_models]

    prev_resp = {q: resp[q].copy() for q in query_ids}
    for q in query_ids:
        ll_vote = np.array([loglik[k][q] for k in range(K)], dtype=float)
        e = norm_vec(embeddings[q])
        sims = e @ centroids.T
        ll_embed = lambda_val * sims        
        if cc == 1:     
            ll_total = ll_vote + ll_embed
        elif cc == 0:
            ll_total = ll_vote
        final_ll = ll_total / T
        final_ll -= final_ll.max()                  
        p = np.exp(final_ll)
        p /= p.sum()                  
        resp[q] = p
    
    kl = _mean_kl(prev_resp, resp)                       
    c_shift = np.linalg.norm(centroids - prev_centroids)
    L_cur = _em_lower_bound(loglik, embeddings, centroids, pi, T, lambda_val, query_ids)
    if it == 0:
        L_prev = L_cur
        no_improve = 0
    drel = (L_cur - L_prev) / (abs(L_prev) + 1e-9)
    no_improve = (no_improve + 1) if (drel < CONV_TOL_REL) else 0
    L_prev = L_cur

    if it % 3 == 0:
        nll_train_resp = nll_soft_with_resp(bt_models, resp, train_votes)
        print(f"... | train_nll(resp-soft)={nll_train_resp:.6f} | ELBO={L_cur:.6f}")

    if (kl < CONV_TOL_KL and c_shift < CONV_TOL_C) or (no_improve > CONV_PATIENCE):
        print(f"[EM] converged at iter {it} | KL={kl:.2e}, Cshift={c_shift:.2e}, patience={no_improve}")
        break



ll_single = logloss_single(test_votes_real)
resp_test = compute_resp_for_test(bt_models, {q for q, *_ in test_votes}, centroids)
query_ids = {q for q, _, _ in test_votes}
labels = {q: int(np.argmax(resp_test[q])) for q in query_ids}

loss = nll_soft_with_resp(bt_models, resp_test, test_votes_real)



ans = {
    "global" : ll_single,
    "clustering" : loss,
    "drop" : (ll_single - loss)/ll_single,
}



def default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.int32, np.int64)):
        return int(o)
    return str(o)

with open(f"results_human/{args.data}_{nm}_{K}_{pairs}_{lambda_val}_{cc}_{ec}_{tp}.json", "w+") as f:
    json.dump(ans, f, default=default)