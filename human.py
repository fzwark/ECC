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
old_data, old_models, human_clustered = data.load_data(pairs=pairs,data=args.data)
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


print(len(train_votes), len(test_votes))

if args.data == "mmlupro" or args.data == "math":
    embeddings = np.load("bge_embeddings0.npy")
elif args.data == "mmlu":
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
offline_gt = OfflineBTRanker(models=old_models)
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



q2cluster = {}
for c, qs in human_clustered.items():
    for q in qs:
        if q in q2cluster and q2cluster[q] != c:
            raise ValueError(f"Query {q} appears in multiple clusters: {q2cluster[q]} and {c}")
        q2cluster[q] = c


cluster_votes = defaultdict(list)
cluster_weights = defaultdict(list)

for q, models, ys in train_votes:
    c = q2cluster.get(q, None)
    if c is None:
        raise KeyError(f"q={q} not in human_clustered")

    Nq = max(1, len(models))
    for (i, j), y in zip(models, ys):
        w = 1.0 / Nq    
        if w < 1e-12:
            continue
        cluster_votes[c].append((q, [[i, j]], [y]))
        cluster_weights[c].append(w)

bt_models = {}
for c in human_clustered.keys():  
    ranker = OfflineBTRanker(models=old_models)
    ranker.fit(
        votes=cluster_votes[c],
        existing_bt=None,
        weights=cluster_weights[c],
    )
    bt_models[c] = ranker


def _loss_from_prob(p, y):
    if y == 1.0:
        return -math.log(p)
    elif y == 0.0:
        return -math.log1p(-p)        

def logloss_oracle(bt_models, labels, test_votes):
    total_loss, n = 0.0, 0
    for q, models, ys in test_votes:
        for (i, j), y in zip(models, ys):
            r_q = labels[q]                   
            p_mix = bt_models[r_q].predict_pairwise(i, j)
            p_mix = min(max(p_mix, EPS), 1-EPS)
            total_loss += _loss_from_prob(p_mix, y)
            n += 1
    return total_loss / n


ll_single = logloss_single(test_votes_real)
query_ids = {q for q, _, _ in test_votes}
labels = {q: q2cluster[q] for q in query_ids}
ll_oracle = logloss_oracle(bt_models, labels, test_votes_real)

ans = {
    "global" : ll_single,
    "clustering" : ll_oracle,
    "drop" : (ll_single - ll_oracle)/ll_single,
}

def default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.int32, np.int64)):
        return int(o)
    return str(o)

with open(f"results_human/{args.data}_human_{K}_{pairs}_{lambda_val}_{cc}_{ec}_{tp}.json", "w+") as f:
    json.dump(ans, f, default=default)