
from collections import defaultdict
import json, math
import numpy as np
import pandas as pd
import requests
import os
import random
import copy
from typing import Dict
from datasets import load_dataset, Dataset, load_from_disk, concatenate_datasets
from itertools import combinations
import pickle

class Data:
    def __init__(self, models=None):
        if models == None:
            self.models = [
                "aws-claude-3-5-sonnet-v1", "aws-titan-text-premier-v1", "openai-gpt-4o", "openai-gpt-4o-mini", "wxai-granite-3-2b-instruct-8k-max-tokens",
                "wxai-granite-3-8b-instruct-8k-max-tokens", "wxai-llama-3-1-70b-instruct", "wxai-llama-3-1-8b-instruct", "wxai-llama-3-2-1b-instruct", 
                "wxai-llama-3-2-3b-instruct", "wxai-llama-3-3-70b-instruct", "wxai-llama-3-405b-instruct", "wxai-mixtral-8x7b-instruct-v01"
            ]
        else:
            self.models = models
        

    def load_raw_data(self, data="sprout"):
        if data == "sprout":
            local_path = "./sprout_raw"
            if not os.path.exists(local_path):
                self.dataset = load_dataset("CARROT-LLM-Routing/SPROUT")
                self.dataset.save_to_disk(local_path)
            else:
                self.dataset = load_from_disk(local_path)
            self.n_train = len(self.dataset['train'])
            self.n_test = len(self.dataset['validation']) + len(self.dataset['test'])
            self.dataset = concatenate_datasets([self.dataset['train'], self.dataset['validation'], self.dataset['test']])
        elif data == "routerbench":
            filename ="routerbench_0shot.pkl"
            local_path = f"{filename}"
                
            with open(local_path, "rb") as f:
                self.dataset = pickle.load(f)

            self.dataset = Dataset.from_pandas(self.dataset)

        elif data == "leaderboard":
            self.dataset = load_from_disk("leaderboard")


    def load_data(self,  pairs=1,data = "sprout"):
        if data == "sprout":
            if not os.path.exists(f"./sprout_data_ranking"):
                from itertools import combinations
                rng = random.Random(42)
                all_pairs = list(combinations(self.models, 2))

                def compute_field(example):
                    for model in self.models:
                        example[model] = example[model]["score"]

                    non_tie_pairs = []
                    for m1, m2 in all_pairs:
                        s1, s2 = example[m1], example[m2]
                        if s1 != s2:
                            non_tie_pairs.append((m1, m2))

                    # prepare all pairs
                    k = len(non_tie_pairs)
                    sampled_pairs = rng.sample(non_tie_pairs, k) if k > 0 else []

                    pair_models, pair_scores, pair_winner = [], [], []
                    for m1, m2 in sampled_pairs:
                        s1, s2 = example[m1], example[m2]
                        pair_models.append([m1, m2])
                        pair_scores.append([s1, s2])
                        pair_winner.append(0.0 if s1 > s2 else 1.0)

                    example["pair_models"] = pair_models
                    example["pair_scores"] = pair_scores
                    example["pair_winner"] = pair_winner
                    example["effective_pairs"] = k 
                    return example
                dataset = self.dataset.map(compute_field) 
                indices = list(range(len(dataset)))
                dataset = dataset.add_column("index", indices)
                dataset.save_to_disk(f"./sprout_data_ranking")
            else:
                dataset = load_from_disk(f"./sprout_data_ranking/")

        elif data == "routerbench":
            if not os.path.exists(f"./routerbench_data_ranking"):
                from itertools import combinations
                rng = random.Random(42)
                all_pairs = list(combinations(self.models, 2))

                def compute_field(example):
                    non_tie_pairs = []
                    for m1, m2 in all_pairs:
                        s1, s2 = example[m1], example[m2]
                        if s1 != s2:
                            non_tie_pairs.append((m1, m2))

                    k = len(non_tie_pairs)
                    sampled_pairs = rng.sample(non_tie_pairs, k) if k > 0 else []

                    pair_models, pair_scores, pair_winner = [], [], []
                    for m1, m2 in sampled_pairs:
                        s1, s2 = example[m1], example[m2]
                        pair_models.append([m1, m2])
                        pair_scores.append([s1, s2])
                        pair_winner.append(0.0 if s1 > s2 else 1.0)

                    example["pair_models"] = pair_models
                    example["pair_scores"] = pair_scores
                    example["pair_winner"] = pair_winner
                    example["effective_pairs"] = k 
                    return example
                dataset = self.dataset.map(compute_field) 
                indices = list(range(len(dataset)))
                dataset = dataset.add_column("index", indices)
                dataset.save_to_disk(f"./routerbench_data_ranking/")
            else:
                dataset = load_from_disk(f"./routerbench_data_ranking/")
        
        elif data == "leaderboard":
            if not os.path.exists(f"./leaderboard_data_ranking/"):
                from itertools import combinations
                rng = random.Random(42)
                all_pairs = list(combinations(self.models, 2))

                def compute_field(example):
                    non_tie_pairs = []
                    for m1, m2 in all_pairs:
                        s1, s2 = example[m1], example[m2]
                        if s1 != s2:
                            non_tie_pairs.append((m1, m2))

                    k = min(pairs, len(non_tie_pairs))
                    sampled_pairs = rng.sample(non_tie_pairs, k) if k > 0 else []

                    pair_models, pair_scores, pair_winner = [], [], []
                    for m1, m2 in sampled_pairs:
                        s1, s2 = example[m1], example[m2]
                        pair_models.append([m1, m2])
                        pair_scores.append([s1, s2])
                        pair_winner.append(0.0 if s1 > s2 else 1.0)

                    example["pair_models"] = pair_models
                    example["pair_scores"] = pair_scores
                    example["pair_winner"] = pair_winner
                    example["effective_pairs"] = k 
                    return example
                dataset = self.dataset.map(compute_field) 
                dataset.save_to_disk(f"./leaderboard_data_ranking/")
            else:
                dataset = load_from_disk(f"./leaderboard_data_ranking/")
      
        dataset = dataset.filter(lambda ex: ex["effective_pairs"] >= 2)
        print(f"Len of data:", len(dataset))

        return dataset, self.models