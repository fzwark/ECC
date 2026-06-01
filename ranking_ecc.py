from data_ranking import Data
from utils import *
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--pairs', type=int, default=3)
parser.add_argument('--k', type=int, default=30)
parser.add_argument('--nnn', type=int, default=1)
parser.add_argument('--comp_num', type=int, default=100)
parser.add_argument('--data', type=str, default="leaderboard")
parser.add_argument('--lam', type=float, default=3)
parser.add_argument('--cc', type=int, default=1)
parser.add_argument('--rd', type=int, default=4)

args = parser.parse_args()
K = args.k 
pairs = args.pairs
nnn = args.nnn
comp_num = args.comp_num
cc = args.cc
rd = args.rd

if cc == 0 :
    nm = "comp"
else:
    nm = "ecc"

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


# load the data
data = Data(models=input_models)
data.load_raw_data(data=args.data)
old_data, all_models = data.load_data(pairs=pairs,data=args.data)
cols = ["index", "pair_models", "pair_winner"]  
all_votes = list(zip(*(old_data[c] for c in cols)))
qids = sorted({q for q, _, _ in all_votes})


new_model = input_models[rd]

# split out old votes and new votes
old_models = all_models.copy()
old_models.remove(new_model)
old_set = set(old_models)
old_votes = []
new_votes = []
sampled_all_votes = []
rng = random.Random(42)
for qid, pair_models, pair_winner in all_votes:
    old_pm = []
    old_win = []
    new_pm = []
    new_win = []

    for (m_a, m_b), w in zip(pair_models, pair_winner):
        if (m_a in old_set) and (m_b in old_set):
            old_pm.append([m_a, m_b])
            old_win.append(w)
        else:
            assert m_a == new_model or m_b == new_model
            new_pm.append([m_a, m_b])
            new_win.append(w)

    if old_pm:
        idxs = rng.sample(range(len(old_pm)), min(pairs, len(old_pm)))
        sampled_old_pm  = [old_pm[i]  for i in idxs]
        sampled_old_win = [old_win[i] for i in idxs]
        old_votes.append((qid, sampled_old_pm, sampled_old_win))
    if new_pm:
        new_votes.append((qid, new_pm, new_win))

    sampled_all_votes.append((qid, sampled_old_pm + new_pm, sampled_old_win + new_win))
assert len(all_votes) == len(sampled_all_votes)

qids_tr, qids_te = train_test_split(qids, test_size=0.2, shuffle=True, random_state=42)


train_votes = [v for v in sampled_all_votes if v[0] in qids_tr]
train_votes_old = [v for v in old_votes if v[0] in qids_tr]
train_votes_new = [v for v in new_votes if v[0] in qids_tr]
test_votes = [v for v in sampled_all_votes if v[0] in qids_te]
test_votes_old = [v for v in old_votes if v[0] in qids_te]
test_votes_new = [v for v in new_votes if v[0] in qids_te]
final_train = train_votes_old + test_votes_old
query_ids = {q for q, _, _ in final_train}

if args.data == "sprout":
    embeddings = np.load("bge_embeddings0.npy")
elif args.data == "leaderboard":
    embeddings = np.load("bge_embeddings2.npy") 
else:
    embeddings = np.load("bge_embeddings1.npy")

def logloss_softmix(bt_models, centroids, embeddings, test_votes, eps=1e-12):
    preds = [m.predict_pairwise for m in bt_models]
    tot, n = 0.0, 0
    for q, models, ys in test_votes:
        for (i, j), y in zip(models, ys):
            p = 0.0
            e = norm_vec(embeddings[q]) 
            sims = e @ centroids.T
            logits = lambda_val * sims 
            logits = logits / T
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
    
def logloss_single(test_votes, bt):
    total, n = 0.0, 0
    for _, models, ys in test_votes:
        for (i, j), y in zip(models, ys):
            total += _vote_loss(bt, i, j, y)
            n += 1
    return total / max(1, n)


# old rankings
old_bt = OfflineBTRanker(models=old_models)
old_bt.fit(votes=final_train, existing_bt=None)
old_ranking = old_bt.score_rank()
train_old_dict = {}
for id, model_pairs, model_wins in final_train:
    train_old_dict[id] = (model_pairs, model_wins)

# baseline
votes_sample_baseline = []
sorted_items = sorted(old_ranking.items(), key=lambda x: x[1])
best_model, best_rank = sorted_items[0]        
mid_model, mid_rank = sorted_items[len(sorted_items) // 2]
worst_model, worst_rank = sorted_items[-1]

cts = 0
ct = 0
train_votes_new_baseline = train_votes_new.copy()
rng = random.Random(42)
rng.shuffle(train_votes_new_baseline)
for id, model_pairs, model_wins in train_votes_new_baseline:
    pm = []
    win = []
    for (m_a, m_b), w in zip(model_pairs, model_wins):
        if m_a == mid_model or m_b == mid_model:
            pm.append([m_a, m_b])
            win.append(w)

    if len(win) == 0:
        continue
    votes_sample_baseline.append((id, pm, win))
    cts += 1
    for item in win:
        if item == 1:
            ct += 1
    if cts == comp_num:
        break


qid_to_allvotes = {
    qid: (old_pairs.copy(), old_wins.copy())
    for qid, (old_pairs, old_wins) in train_old_dict.items()
}

for qid, new_pairs, new_wins in votes_sample_baseline:
    old_pairs, old_wins = qid_to_allvotes.get(qid, ([], []))
    qid_to_allvotes[qid] = (old_pairs + new_pairs, old_wins + new_wins)

votes_baseline = [(qid, pairs, wins) for qid, (pairs, wins) in qid_to_allvotes.items()]
baseline_bt = OfflineBTRanker(models=all_models)
baseline_bt.fit(votes=votes_baseline, existing_bt=None)


seed = 42
max_iter = int(1e5)
rng = np.random.default_rng(seed=seed)
resp = {q: rng.dirichlet(np.ones(K)) for q in query_ids} 
pi = np.ones(K) / K

def accumulate_loglik(model):
    acc = defaultdict(float)
    pred = model.predict_pairwise   
    log = math.log
    for q, models, ys in final_train:
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
        total += m + np.log(np.exp(v - m).sum())  
    return float(total)

def _mean_kl(old, new):
    s = 0.0
    for q in old.keys():
        p = np.clip(old[q], EPS, 1.0); p /= p.sum()
        qv= np.clip(new[q], EPS, 1.0); qv/= qv.sum()
        s += np.sum(p * (np.log(p) - np.log(qv)))
    return s / max(1,len(old))

T = 0.1
lambda_val = args.lam

CONV_TOL_KL   = 1e-3     
CONV_TOL_C    = 1e-4     
CONV_TOL_REL  = 1e-4    
CONV_PATIENCE = 4 


embedding_dim = embeddings.shape[1]
centroids = np.zeros((K, embedding_dim))
qs = list(query_ids)
E = np.stack([embeddings[q] for q in qs], axis=0)   # (Q', D)
norms = np.linalg.norm(E, axis=1, keepdims=True)
E = E / np.maximum(norms, 1e-12)
R = np.stack([resp[q] for q in qs], axis=0) # (Q', K)
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
        for q, models, ys in final_train:
            Nq = max(1, len(models))
            for (i, j), y in zip(models, ys):
                w = resp[q][k] / Nq  
                if w < 1e-12:         
                    continue
                weighted_votes.append((q, [[i, j]], [y]))
                weights.append(w)
        ranker = OfflineBTRanker(models=old_models)
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
        nll_train_resp = nll_soft_with_resp(bt_models, resp, final_train)
        print(f"... | train_nll(resp-soft)={nll_train_resp:.6f} | ELBO={L_cur:.6f}")

    if (kl < CONV_TOL_KL and c_shift < CONV_TOL_C) or (no_improve > CONV_PATIENCE):
        print(f"[EM] converged at iter {it} | KL={kl:.2e}, Cshift={c_shift:.2e}, patience={no_improve}")
        break


def compute_resp_for_test(bt_models, probe_vote, test_query_ids):
    K = len(bt_models)
    preds = [m.predict_pairwise for m in bt_models]

    votes_by_q = defaultdict(list)
    for q, models, ys in probe_vote:
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
        ll_total = (ll + ll_embed)/T
        log_post = ll_total 
        log_post -= log_post.max()          
        r = np.exp(log_post)
        r /= r.sum()
        resp_test[q] = r
    return resp_test



labels = {q: int(np.argmax(resp[q])) for q in query_ids}  

new_qids_pool = []
qid_to_pairswins_new = {}  
for qid, model_pairs, model_wins in train_votes_new:
    if qid in labels:  
        new_qids_pool.append(qid)
        qid_to_pairswins_new[qid] = (model_pairs, model_wins)

rng = random.Random(42)
rng.shuffle(new_qids_pool)

votes_sample_ours = []
ct = 0
for qid in new_qids_pool:
    k = labels[qid]
    pick_model = bt_models[k]
    pick_model_ranking = pick_model.score_rank()
    sorted_items = sorted(pick_model_ranking.items(), key=lambda x: x[1])
    mid_model   = sorted_items[len(sorted_items) // 2][0]
    keep_set = {mid_model}
    model_pairs, model_wins = qid_to_pairswins_new[qid]
    pm = []
    win = []
    for (m_a, m_b), w in zip(model_pairs, model_wins):
        if (m_a in keep_set) or (m_b in keep_set):
            pm.append([m_a, m_b])
            win.append(w)

    if len(pm) == 0:
        continue
    votes_sample_ours.append((qid, pm, win))
    ct += 1
    if ct == comp_num:
        break


votes_ours = []
qid_to_allvotes = {}
for qid, model_pairs, model_wins in votes_sample_ours:
    old_pairs, old_wins = train_old_dict.get(qid, ([], []))
    all_pairs = model_pairs + old_pairs
    all_wins  = model_wins  + old_wins
    qid_to_allvotes[qid] = (all_pairs, all_wins)

all_qids_for_bt = set(train_old_dict.keys()) | {qid for qid, _, _ in votes_sample_ours}

bt_models_cluster = []  
for k in range(K):
    weighted_votes = []
    weights = []

    for qid in all_qids_for_bt:
        all_pairs, all_wins = qid_to_allvotes.get(qid, train_old_dict.get(qid, ([], [])))
        if qid not in resp:
            continue
        r_qk = resp[qid][k]         
        Nq = max(1, len(all_pairs))

        w_pair = r_qk / Nq
        if w_pair < 1e-12:
            continue
        for (i, j), y in zip(all_pairs, all_wins):
            weighted_votes.append((qid, [[i, j]], [y]))
            weights.append(w_pair)

    bt_k = OfflineBTRanker(models=all_models)
    bt_k.fit(votes=weighted_votes, existing_bt=None, weights=weights)
    bt_models_cluster.append(bt_k)

test_qids_new = {q for q, _, _ in test_votes_new}
eval_qids = test_qids_new & set(resp.keys())   
labels = {q: int(np.argmax(resp[q])) for q in eval_qids}

test_votes_new_eval = [
    (q, models, ys) for (q, models, ys) in test_votes_new
    if q in eval_qids
]

ll_single = logloss_single(test_votes_new_eval, baseline_bt)
loss = nll_soft_with_resp(bt_models_cluster, resp, test_votes_new_eval)

ans = {
    "model" : new_model,
    "global" : ll_single,
    "clustering" : loss,
    "drop" : (ll_single - loss)/ll_single
}


with open(f"efficiency/{args.data}_{nm}_{K}_{comp_num}_{cc}_{rd}.json", "w+") as f:
    json.dump(ans, f)