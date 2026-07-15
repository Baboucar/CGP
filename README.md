# Graph propagation for collaborative filtering code, and data

Self-contained package for the paper *"Contrastive graph propagation for top-K collaborative filtering"* (prepared for ICT Express).

## Layout

```
VitRec_graph.ipynb     Main notebook. Trains the model and reports HR@10 / NDCG@10.
graphvit_model.py      The graph model (LightGCN + optional ViT + XSimGCL contrastive) + DirectAU.
graph_baselines.py     Recent graph CF baselines: NGCF, SGL, LightGCL, SCCF.
baselines.py           MF / neural CF baselines (BPR-MF, GMF, MLP, NeuMF, ConvNCF).
data/                  MovieLens-1M and Pinterest-20 (He et al. 2017 leave-one-out split).
figures/               All figures: architecture, comparison bars, sensitivity, convergence.
results/               One JSON per training run: config, best HR/NDCG, epoch, full history.
```

Every number in the paper's tables traces to a JSON in `results/` (the `args` field holds the full
configuration of the run that produced it).

## Results

Under a corrected leave-one-out protocol (each held-out item ranked against 99 sampled negatives,
ties split evenly), CGP (contrastive graph propagation, a LightGCN backbone with an XSimGCL-style contrastive
term) is the strongest of ten methods on both datasets, with paired McNemar tests confirming
the advantage against every graph baseline on both (results/significance.json):

| Method | HR@10 (ML-1M) | HR@10 (Pinterest-20) |
|--------|:---:|:---:|
| BPR-MF | 0.6798 | 0.8738 |
| NeuMF | 0.6728 | 0.8593 |
| NGCF (2019) | 0.6957 | 0.8687 |
| SGL (2021) | 0.7017 | 0.8748 |
| DirectAU (2022) | 0.5940 | 0.8799 |
| LightGCL (2023) | 0.6911 | 0.8657 |
| SCCF (2024) | 0.5733 | 0.8540 |
| **CGP (ours)** | **0.7164** | **0.8841** |

A vision-transformer interaction encoder on top of the graph model adds parameters but no accuracy
(ablation in the paper). DirectAU and SCCF rank low on MovieLens-1M but competitively on
Pinterest-20: their uniformity objective is tuned for full-ranking recall and transfers unevenly to
the sampled-negative protocol (see the paper's discussion and Krichene & Rendle, KDD 2020).

## How to run

Requirements: Python 3.10, PyTorch 2.x with CUDA (a 6 GB GPU suffices), `numpy`, `scipy`,
`matplotlib`. Run everything from this folder so the relative `data/` paths resolve.

```
jupyter notebook VitRec_graph.ipynb     # run top to bottom (~1.5–2 h on a 6 GB GPU)
```

Reproduce any table row from the command line (swap `data/ml-1m` for `data/pinterest-20`):

```
python graphvit_model.py --model lgn  --data data/ml-1m --epochs 160 --cl 0.005          # our model
python graphvit_model.py --model lgn  --data data/ml-1m --epochs 90  --loss directau --gamma 0.5
python graph_baselines.py --model ngcf     --data data/ml-1m --epochs 200 --reg 1e-5
python graph_baselines.py --model sgl      --data data/ml-1m --epochs 90  --cl 0.02 --in_batch_neg
python graph_baselines.py --model lightgcl --data data/ml-1m --epochs 120 --cl 0.005 --layers 2 --svd_q 5
python graph_baselines.py --model sccf     --data data/ml-1m --epochs 300 --tau 0.1 --batch 40000 --no_amp
python baselines.py       --model neumf    --data data/ml-1m --epochs 30
```

Every method uses the same embedding size, split, and evaluation, so the numbers are directly
comparable. The recent baselines were re-tuned on this protocol rather than copied from papers that
report under a different (full-ranking) metric.

## Evaluation note

The held-out item is ranked by the number of negatives scoring strictly higher, with ties split
evenly. This avoids the sort-order artifact that lets an untrained model report HR = NDCG = 1.0 (the
positive is always element zero of the candidate list and wins every tie under a naive `argsort`).
