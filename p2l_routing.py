from data import Data
from utils import *
from sklearn.model_selection import train_test_split
from model import get_p2l_model, get_tokenizer
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--pairs', type=int, default=7)
parser.add_argument('--data', type=str, default="sprout")
parser.add_argument('--tp', type=float, default=0.2)

args = parser.parse_args()
pairs = args.pairs
tp = args.tp 

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

performance_score = {}
for ex in old_data:
    idx = ex["index"]
    model_scores = {m: ex[m] for m in input_models}
    performance_score[idx] = model_scores

cols = ["index", "pair_models", "pair_winner", "prompt"]  
all_votes = list(zip(*(old_data[c] for c in cols)))
qids = sorted({q for q, _, _, _ in all_votes})
qidtoprompt = {q : prompt for q, _, _, prompt in all_votes}
all_votes = [item[:-1] for item in all_votes]
qids_tr, qids_te = train_test_split(qids, test_size=0.2, shuffle=True, random_state=42)
train_votes = [v for v in all_votes if v[0] in qids_tr]
test_raw = [v for v in all_votes if v[0] in qids_te]
query_ids = {q for q, _, _ in train_votes}
rng = random.Random(0)
test_votes = [
    (qid, [m], [y])
    for (qid, models, ys) in test_raw
    for (m, y) in [rng.choice(list(zip(models, ys)))]
]


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
    init_type="reset_params", 
)

tmodel = model_cls.from_pretrained(
    ckpt_dir,
    CLS_id=tokenizer.cls_token_id,
    num_models=len(model_lists),
).to(device)

tmodel.eval()

# p2l
p2l = 0.0
for q in qids_te:
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

    m_idx = int(np.argmax(betas))
    m_name = input_models[m_idx]
    p2l += performance_score[q][m_name]

ans = {
    "p2l" : p2l
}

def default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.int32, np.int64)):
        return int(o)
    return str(o)

with open(f"p2l/{args.data}_p2l_{pairs}_{tp}_routing.json", "w+") as f:
    json.dump(ans, f, default=default)