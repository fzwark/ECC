import os
import pickle
import requests
import random, json
import numpy as np
import yaml, torch
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, load_from_disk, concatenate_datasets
import pandas as pd
from functools import reduce
from sentence_transformers import SentenceTransformer


datasets = []

# sprout
if not os.path.exists("sprout_raw"):
    dataset = load_dataset("CARROT-LLM-Routing/SPROUT")
    dataset.save_to_disk("sprout_raw")
else:
    dataset = load_from_disk("sprout_raw")
n_train = len(dataset['train'])
n_test = len(dataset['validation']) + len(dataset['test'])
dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
datasets.append(dataset)

# get routerbench from https://huggingface.co/datasets/withmartian/routerbench
filename ="routerbench_0shot.pkl"
with open(filename, "rb") as f:
    dataset = pickle.load(f)
dataset = Dataset.from_pandas(dataset)
datasets.append(dataset)

# get raw data from: https://github.com/somerstep/CARROT/blob/main/data/open-llm-lb-v2/data_QA.json
if not os.path.exists("leaderboard"):
    local_path = "leaderboard_raw/data_QA.json"
    with open(local_path, "r") as f:
        data_QA = json.load(f)
    with open("config.yaml", "r") as f:
        ans = yaml.safe_load(f)
    benchmark = ans['OPEN_BENCHMARKS']
    raw_models = ans['RAW_MODELS']
    raw_models = [m.replace("open-llm-leaderboard/","").replace("__","/").replace("-details","") for m in raw_models]
    tokenizer_dict = {}
    for model in raw_models:
        tokenizer_dict[model] = AutoTokenizer.from_pretrained(model, use_fast=False)

    for sce in data_QA.keys():
        data_QA[sce]['tokens'] = np.array([[len(tokenizer_dict[model](q)['input_ids']) for q in data_QA[sce]['Ps']] for model in raw_models]).T
    
    bench = 'all'
    benchmark['all'] = benchmark['bbh']+benchmark['gpqa']+benchmark['math']+benchmark['mmlu_pro']+benchmark['musr']

    data_Y = np.load("leaderboard_raw/new_leaderboard_processed_20241205.pickle", allow_pickle=True)
    M = [data_Y[k]['models'] for k in benchmark[bench]]
    M = np.sort(list(reduce(set.intersection, map(set, M)))).tolist()
    Y = [data_Y[k]['correctness'][[int(np.argmax(np.array(data_Y[k]['models'])==m)) for m in M]] for k in benchmark[bench]]
    Y = np.hstack(Y)
    data_Y[bench] = {}
    data_Y[bench]['correctness'] = Y.T
    data_Y[bench]['models'] = [m.replace("open-llm-leaderboard/","").replace("__","/").replace("-details","") for m in M]

    def flatten(xss):
        return [x for xs in xss for x in xs]
    data_QA[bench] = {}
    data_QA[bench]['Qs'] = flatten([data_QA[k]['Qs'] for k in benchmark[bench]])
    data_QA[bench]['tokens'] = np.vstack([data_QA[k]['tokens'] for k in benchmark[bench]])

    records = []
    for q, token_row, correct_row in zip(
        data_QA[bench]['Qs'],
        data_QA[bench]['tokens'],
        data_Y[bench]['correctness']
    ):
        entry = {'prompt': q}
        for idx, model in enumerate(data_Y[bench]['models']):
            entry[f"{model}"] = correct_row[idx]
        records.append(entry)

    df = pd.DataFrame(records)
    dataset = Dataset.from_pandas(df)
    indices = list(range(len(dataset)))
    dataset = dataset.add_column("index", indices)
    dataset.save_to_disk("leaderboard")
else:
    dataset = load_from_disk("leaderboard")

datasets.append(dataset)

# generate embeddings for all benchmarks
for i, item in enumerate(datasets):
    if not os.path.exists(f"bge_embeddings{i}.npy"):
        sampled_dataset = item["prompt"]
        model = SentenceTransformer('BAAI/bge-base-en-v1.5', device='cuda')
        with torch.no_grad():
            embeds = model.encode(sampled_dataset, batch_size=512, convert_to_numpy=True, show_progress_bar=True)
        np.save(f"bge_embeddings{i}.npy", embeds)
    else:
        embeds = np.load(f"bge_embeddings{i}.npy")
