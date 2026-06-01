from data import Data
from utils import *
from sklearn.model_selection import train_test_split
from model import get_p2l_model, get_tokenizer
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--pairs', type=int, default=7)
parser.add_argument('--nnn', type=int, default=1)
parser.add_argument('--data', type=str, default="sprout")
parser.add_argument('--tp', type=float, default=0.2)

args = parser.parse_args()
pairs = args.pairs
tp = args.tp 
nnn = args.nnn

def norm_vec(x): 
    n = np.linalg.norm(x)
    return x / max(n, 1e-12)

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

# load model
model_lists = input_models
ckpt_dir = f"{args.data}_model"
chat_template = "{%- if messages[0]['role'] == 'system' %}\n    {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n\n"
model_type="qwen2"
loss_type="bt"      
head_type="bt"
device="cuda"
model2id = {m: i for i, m in enumerate(model_lists)}

# 2) tokenizer 
tokenizer = get_tokenizer(
    tokenizer_name=ckpt_dir,   
    chat_template=chat_template,
)

# 3) build model class & load weights
model_cls = get_p2l_model(
    model_type=model_type,
    loss_type=loss_type,
    head_type=head_type,
    init_type="reset_params",  # inference 不重要
)

tmodel = model_cls.from_pretrained(
    ckpt_dir,
    CLS_id=tokenizer.cls_token_id,
    num_models=len(model_lists),
).to(device)

tmodel.eval()

def logloss_p2l(test_votes, eps=1e-12):
    tot, n = 0.0, 0
    for q, models, ys in test_votes:
        prompt = qidtoprompt[q]
        chat_inputs = []
        chat_inputs.append([{"role": "user", "content": prompt}])
        formatted = tokenizer.apply_chat_template(
            chat_inputs,
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        formatted = [s.replace(tokenizer.cls_token, "<cls>") for s in formatted]
        formatted = [s + tokenizer.cls_token for s in formatted]

        enc = tokenizer(
            formatted,
            padding=True,
            truncation=True,
            max_length=16384,
            add_special_tokens=False,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = tmodel(**enc) 
            betas = out.coefs[0].detach().float().cpu()
            
        for (i, j), y in zip(models, ys):
            y = float(y) 

            bi = float(betas[model2id[i]])
            bj = float(betas[model2id[j]])
            logit = bi - bj
            p = 1.0 / (1.0 + math.exp(-logit))
            p = min(max(p, eps), 1.0 - eps)

            # p2l models the probability that i > j, but in our dataset, y = 1 means j > i.
            # To align, we use 1 - y.
            delta = -((1 - y) * math.log(p) + (y) * math.log1p(-p))
            tot += delta
            n += 1
    return tot / n


EPS = 1e-12
def _vote_loss(model, i, j, y):
    p = model.predict_pairwise(i, j)
    p = min(max(p, EPS), 1 - EPS)
    if y == 1.0:
        return -math.log(p)
    elif y == 0.0:
        return  -math.log1p(-p)
    else:
        return  -0.5 *(math.log(p) + math.log1p(-p))
    
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


loss_global = logloss_single(test_votes_real)
loss_emb = logloss_p2l(test_votes_real, eps=1e-12)


ans = {
    "global" : loss_global,
    "clustering_emb_only" : loss_emb,
    "drop" : (loss_global - loss_emb)/loss_global,
}

print("drop:", (loss_global - loss_emb)/loss_global )

def default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.int32, np.int64)):
        return int(o)
    return str(o)

with open(f"p2l/{args.data}_p2l_{pairs}_{tp}_7b.json", "w+") as f:
    json.dump(ans, f, default=default)