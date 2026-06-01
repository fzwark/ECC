# Capturing LLM Capabilities via Evidence-Calibrated Query Clustering

![LLM Capabilities](https://img.shields.io/badge/LLM_Capabilities-yellow.svg) ![Query Clustering](https://img.shields.io/badge/Query_Clustering-blue.svg) ![Evidence-Calibrated Clustering](https://img.shields.io/badge/Evidence_Calibrated_Clustering-green.svg)

## Introduction

Query clustering organizes queries into groups that reflect shared latent capability demands, enabling capability-aware LLM evaluation.
Existing clustering methods, which primarily rely on semantic taxonomies or embeddings, often fail to capture such latent capability requirements due to a misalignment between surface-level semantics and actual model performance. 
We propose ECC, an algorithm that calibrates prior semantic embeddings using limited posterior model comparisons to bridge the gap between surface-level semantics and latent capability requirements. 
ECC characterizes each cluster through a capability profile parameterized by a Bradley-Terry model and uses trainable mixture weights 
to accommodate queries with mixed capability demands, jointly learning a flexible, capability-aware clustering structure that supports query-specific inference of LLM capabilities.
Extensive quantitative and qualitative evaluations demonstrate that ECC significantly improves LLM capability ranking quality, outperforming human-labeled and embedding-based baselines by an average of **17.64** and **18.02** percentage points, respectively, and proves effective in downstream tasks such as query routing.

## Updates

- [2026.05.31] 🚀 Our code is now released! 

## Setup

Install dependencies:

```bash
cd ECC
pip install -r requirements.txt
```

Download benchmark data and generate embeddings:
```bash
python data_gen.py
```

## Evaluations
### Evaluation on Benchmarks

To reproduce results on the three benchmarks, run:

```bash
# ECC
python clustering_ecc.py --cc=1 --ec=1 --k=30 --pairs=7 --lam=3 --data=sprout

# Comp-only
python clustering_ecc.py --cc=0 --ec=1 --k=30 --pairs=7 --lam=3 --data=sprout

# Emb-only
python clustering_emb.py --ec=1 --k=30 --pairs=7 --lam=3 --data=sprout
```
Key arguments:
- `k`: number of clusters
- `data`: benchmark dataset in `{sprout, routerbench, leaderboard}`
- `pairs`: number of model comparison pairs used per query during clustering
- `lam`: trade-off between prior embedding signal and posterior comparison signal
- `cc`: whether to use embeddings during clustering (`1` = yes, `0` = no)
- `ec`: whether to use embeddings during inference (`1` = yes, `0` = no)


### Compare with Human-Labeled Clustering

To compare with human-labeled clustering on datasets with human-labeled categories, run:

```bash
# ECC
python human_ecc.py --pairs=7 --lam=3 --data=mmlu

# Human category
python human.py --pairs=7 --lam=3 --data=mmlu

# Emb-only
python human_emb.py --pairs=7 --lam=3 --data=mmlu
```
- `data` in `{mmlu, mmlupro, math}`

### Query Routing

To evaluate the downstream task of optimal query routing, run:

```bash
# ECC
python routing_ecc.py --cc=1 --ec=1 --k=30 --pairs=7 --lam=3 --data=leaderboard

# Comp-only
python routing_ecc.py --cc=0 --ec=1 --k=30 --pairs=7 --lam=3 --data=leaderboard

# Emb-only
python routing_emb.py --ec=1 --k=30 --pairs=7 --lam=3 --data=leaderboard
```

### Sample Efficient New Model Ranking

To evaluate sample-efficient ranking for a new model with a limited comparison budget, run:

```bash
# ECC
python routing_ecc.py --cc=1 --ec=1 --k=30 --pairs=3 --lam=3 --data=leaderboard --comp_num=100 --rd=4

# Comp-only
python routing_ecc.py --cc=0 --ec=1 --k=30 --pairs=3 --lam=3 --data=leaderboard --comp_num=100 --rd=4

# Emb-only
python routing_emb.py --ec=1 --k=30 --pairs=3 --lam=3 --data=leaderboard --comp_num=100 --rd=4
```

Key arguments:
- `comp_num`: comparison budget for the new model
- `rd`: index of the model to hold out as the new model
    - Open LLM Leaderboard: `rd` in `[0, 16]`
    - RouterBench: `rd` in `[0, 11]`
    - SPROUT: `rd` in `[0, 13]`



## Citation

If you use this codebase, please consider citing our paper:

```bibtex
@misc{wu2026capturingllmcapabilitiesevidencecalibrated,
      title={Capturing LLM Capabilities via Evidence-Calibrated Query Clustering}, 
      author={Fangzhou Wu and Sandeep Silwal and Qiuyi Zhang},
      year={2026},
      eprint={2605.17110},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.17110}, 
}
```