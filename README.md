# GraFT: Spatial-as-Refinement for Spatio-Temporal Forecasting

<p align="center">
  <img src="assets/spatial_as_refinement.png" alt="Substrate models compound graph noise multiplicatively through every layer; GraFT applies alpha-gated graph refinements after a complete per-sensor forecast, so each gate controls its own noise" width="100%">
</p>

<p align="center">
  <img src="assets/graft_architecture.png" alt="GraFT architecture: a per-sensor temporal predictor produces a complete forecast, then a stack of alpha-gated graph refinement blocks corrects it" width="100%">
</p>

<p align="center">
  <img src="assets/main_results.png" alt="Main results: test MAE on 14 benchmarks; GraFT (PatchTST) has 8 wins and the best average rank, 1.50" width="100%">
  <br><em>Test MAE on 14 benchmarks (mean of 5 seeds). Red: best; blue, underlined: second best.</em>
</p>

Official implementation of **Spatial-as-Refinement: Gated Graph Operators over Strong Temporal Predictors for Scalable Spatio-Temporal Forecasting**.

For a decade, graph-based forecasters have improved *what* the graph operator computes, but not *where* it sits. Placed inside every layer, a noisy learned graph spreads its errors through the whole model, with no way to switch them off. GraFT reverses the order. A per-sensor temporal predictor first produces a complete forecast, and a small stack of graph blocks then refines it:

$$y \leftarrow y + \alpha \cdot \mathrm{LN}(\mathcal{G}(y)), \qquad \alpha \text{ initialized at } 0.$$

At initialization GraFT is exactly its temporal backbone. Each gate $\alpha$ learns from data how much the graph should help. Across 14 benchmarks in 4 domains (traffic, energy, meteorology, mobility), GraFT has the best average rank among strong published baselines. The trained gates also show *where* the graph actually helps.

- Works with **any per-sensor predictor**: the paper's MLP, PatchTST and TSMixer, plus additional backbones such as DLinear, TimeMixer and iTransformer.
- **Needs no adjacency matrix**: GraFT learns its own graph, and scales linearly in the number of sensors (8,600 sensors on CA).
- One command runs GraFT on the paper benchmarks **or on your own CSV**.

> 📄 **Paper:** [preprint (PDF)](https://vpham47.github.io/files/Spatial_as_refinement_Preprint%20%281%29.pdf) · Citation at the bottom of this page.

---

## Contents

1. [Quickstart](#1-quickstart)
2. [Choose your setup](#2-choose-your-setup)
3. [Bring your own data](#3-bring-your-own-data)
4. [Tuning by dataset size](#4-tuning-by-dataset-size)
5. [Reproduce the paper](#5-reproduce-the-paper)
6. [Backbones](#6-backbones)
7. [Repository layout](#7-repository-layout)
8. [Citation and acknowledgements](#8-citation-and-acknowledgements)

---

## 1. Quickstart

```bash
git clone <this repo> && cd <repo>
pip install -r requirements.txt          # tested with Python 3.11 and PyTorch 2.5

python tests/smoke_test.py --quick       # a minute or two on CPU

python tools/make_synthetic.py --out data/synthetic        # a tiny dataset with community structure
python run.py --backbone patchtst --data data/synthetic
```

Each run writes a log, the best checkpoint and a `result_s<seed>.json` (MAE, RMSE, MAPE, parameter count and the learned gates) to `experiments/<dataset>/graft_<backbone>/<experiment_name>/`.

---

## 2. Choose your setup

**Q1. How many sensors (N)?** This sets the hyperparameter preset; see [Section 4](#4-tuning-by-dataset-size).

| Size | Sensors N | Examples from the paper | Preset |
|---|---|---|---|
| Small | fewer than 500 | PEMS03/04/08, PEMS-Bay, UrbanEV, Electricity, KnowAir, ChinaAir | `--preset small` |
| Medium | 500 to 2,500 | SD, PEMS07, GBA | `--preset medium` |
| Large | more than 2,500 | GLA, CA, Shanghai Mobile | `--preset large` |

With `--data` the default is `--preset auto`, which picks the class from N for you. No adjacency matrix is needed: GraFT learns which sensors to share information with.

**Q2. Which backbone?**

| Backbone | When to use it |
|---|---|
| `patchtst` | Best overall in the paper (average rank 1.50). Default choice. |
| `mlp` | Fastest and lightest; best on Shanghai Mobile and China Air Quality. |
| `tsmixer` | Converges in far fewer epochs; best on KnowAir. |
| an additional backbone | To try another forecaster as the per-sensor predictor; see [Section 6](#6-backbones). |

Then run, for example:

```bash
python run.py --backbone patchtst --data data/my_city                 # auto preset
python run.py --backbone mlp      --data data/my_city --preset large
python run.py --backbone dlinear  --data data/my_city                 # an additional backbone
```

---

## 3. Bring your own data

**From a CSV.** One row per time step, one column per sensor, and optionally a timestamp in the first column:

```text
timestamp,sensor_a,sensor_b,sensor_c
2024-01-01 00:00,412,388,97
2024-01-01 00:05,405,391,102
```

```bash
python tools/csv_to_graft.py --csv my_city.csv --out data/my_city
# without timestamps:  --no_timestamp --steps_per_day 288
# window lengths:      --seq_len 12 --horizon 12    split: --split 0.6 0.2 0.2
```

The converter:
- infers the sampling interval from the timestamps;
- derives time-of-day and day-of-week features;
- standardizes values using the training split only;
- splits chronologically.

Missing values (empty cells) are stored as 0 and are excluded from the metrics.

**The folder format.** If you prefer to write it yourself, a dataset is a folder with:

| File | Content |
|---|---|
| `his.npz` | `data` [T, N, C] float32 (channel 0 = standardized target, 1 = time of day in [0, 1), 2 = day of week / 7 in [0, 1)), `mean`, `std` |
| `idx_train.npy`, `idx_val.npy`, `idx_test.npy` | the last input step of each window |
| `meta.json` | `num_nodes`, `steps_per_day`, `seq_len`, `horizon`, `tod_channel`, `dow_channel` |

---

## 4. Tuning by dataset size

The presets below are distilled from the 42 paper runs (3 backbones × 14 datasets; exact values in [`configs/paper_configs.json`](configs/paper_configs.json)). They are reliable starting points, not tuned optima for every dataset.

| | Small (N < 500) | Medium (500–2,500) | Large (N > 2,500) |
|---|---|---|---|
| Refinement depth `graft_num_layers` (L) | 3 | 3 | 1 |
| Graph modes `graft_graph_dG` | 64 | 128 | 128 |
| Batch size `bs` | 16 (32 also works) | 16 | 16 (4 if memory is tight) |
| Learning rate `lrate` | 2e-3 | 1e-3 | 2e-3 |
| Epochs: MLP / PatchTST | 60 (100 for hourly or coarser data) | 60 | 60 |
| Epochs: TSMixer | 30 | 30 | 60 |

Width of the per-sensor predictor (`graft_hidden` / `graft_pred_latent`): MLP 512 / 128, PatchTST 128 / 64, TSMixer 128 / 64.

Shared by every paper run:
- AdamW, weight decay 1e-4;
- 5 warmup epochs from 1e-4, then linear decay to 5e-5;
- gradient clipping 5, weight EMA 0.999, early-stopping patience 7.

**What each setting does, and how to adjust it.**

- **Refinement depth L.**
  - Each layer is one gated graph correction of the forecast.
  - Small and medium networks benefit from 3.
  - On very large networks one layer is enough and saves memory.
  - Exception: the paper used L = 3 on Shanghai Mobile (N = 3,042) with batch size 4.
- **Graph modes d_G.**
  - Each sensor reads from and writes to d_G shared modes. This is the rank of the learned graph, and cost grows linearly with it.
  - Raise it when N grows or the network has many distinct regions.
  - 64 is enough for most networks under 500 sensors.
- **Batch size.** The first knob to lower when you run out of GPU memory; quality is not very sensitive to it.
- **Learning rate.** The paper used 2e-3 on most datasets. It used 1e-3 on:
  - SD and Electricity;
  - GBA and PEMS08 with the MLP and PatchTST backbones.

  If training is unstable, halve it.
- **Epochs.**
  - Early stopping (patience 7) usually ends training before the epoch limit.
  - TSMixer needs far fewer epochs than the MLP and PatchTST backbones.
- **Sampling frequency.** `steps_per_day` sizes the time-of-day embedding: 288 for 5-minute data, 96 for 15-minute, 24 for hourly. The converter sets it for you.

**Read the gates.** At the end of training the log prints the learned gate |α| of every refinement layer. A mean gate near 0 means the graph is not helping on your data, and GraFT has fallen back to its temporal backbone. In the paper, across the four benchmarks where this was measured, the mean gate falls monotonically with network connectivity: from 1.18 on PEMS-Bay to 0.28 on PEMS08.

Override any setting on the command line, for example `--graft_num_layers 2 --graft_graph_dG 96 --lrate 0.001`. All settings are listed in [`graft/config.py`](graft/config.py).

---

## 5. Reproduce the paper

**Data.** The benchmarks come from their original sources:
- [LargeST](https://github.com/liuxu77/LargeST) for SD, GBA, GLA and CA (year 2019);
- [BasicTS](https://github.com/GestaltCogTeam/BasicTS) for PEMS03/04/07/08;
- [DCRNN](https://github.com/liyaguang/DCRNN) for PEMS-Bay;
- [UrbanEV](https://github.com/IntelligentSystemsLab/ST-EVCDP);
- [Electricity](https://github.com/decisionintelligence/TFB);
- [KnowAir](https://github.com/shuowang-ai/PM2.5-GNN);
- [China City Air Quality](https://github.com/Friger/GAGNN);
- [Shanghai Mobile](https://drive.google.com/file/d/1TWD3QDBrsn90zxbDom94BF4fR-NOp0Pi/view).

The expected folder layout is listed in [`graft/datasets.py`](graft/datasets.py). For example, `data/sd/2019/his.npz` sits under `--data_root`.

**Run.** With `--dataset`, the paper settings are used by default:

```bash
python run.py --backbone patchtst --dataset sd
python run.py --backbone mlp      --dataset pems08 --seed 1
```

Dataset names are: `sd gba gla ca pems03 pems04 pems07 pems08 bay urbanev ele sh knowair ccaq`. The paper reports the mean of 5 seeds.

**Main results (test MAE, mean of 5 seeds).** The full table, with RMSE and MAPE, is in the paper.

| | SD | GLA | GBA | CA | PEMS03 | PEMS04 | PEMS07 | PEMS08 | PEMS-Bay | Avg. rank |
|---|---|---|---|---|---|---|---|---|---|---|
| GraFT (PatchTST) | 16.64 | 19.23 | **19.09** | **17.78** | **14.34** | **18.14** | **18.90** | 13.79 | **1.54** | **1.50** |
| GraFT (MLP) | 16.72 | 19.37 | 19.51 | 17.87 | 14.53 | 18.20 | 19.22 | 13.87 | 1.57 | 2.21 |
| GraFT (TSMixer) | 17.02 | 20.12 | 20.35 | 18.74 | 14.87 | 18.38 | 19.54 | 14.26 | 1.58 | 4.29 |
| MAGE | **16.35** | 19.74 | 19.84 | 18.28 | 14.66 | 18.25 | 19.63 | 13.93 | 1.59 | 4.43 |
| PatchSTG | 16.98 | **19.19** | 19.67 | 17.99 | – | – | – | – | 1.61 | 5.44 |

Average rank is over all 14 benchmarks, including the energy, meteorology and mobility datasets not shown here.

---

## 6. Backbones

The per-sensor predictor is swappable. Every backbone maps one sensor's input window `[T, C]` to its forecast latent `[horizon, d]`, and the refinement stack in [`graft/models/core.py`](graft/models/core.py) is shared by all of them.

**Main backbones** ([`graft/backbones/main`](graft/backbones/main)): the three evaluated in the paper. Their paper settings are in `configs/paper_configs.json`.

| Backbone | Predictor |
|---|---|
| `mlp` | two-layer MLP over the flattened window |
| `patchtst` | PatchTST-style transformer over patches of the window |
| `tsmixer` | TSMixer-style time- and feature-mixing MLP |

**Additional backbones** ([`graft/backbones/additional`](graft/backbones/additional)): further forecasters anyone can plug in. They were not evaluated in the paper and use untuned defaults; start from the size presets and expect to tune.

| Backbone | Model | Source |
|---|---|---|
| `dlinear` | DLinear (AAAI 2023) | [Time-Series-Library](https://github.com/thuml/Time-Series-Library) (MIT) |
| `timemixer` | TimeMixer (ICLR 2024) | Time-Series-Library |
| `timesnet` | TimesNet (ICLR 2023); slow, use a GPU | Time-Series-Library |
| `itransformer` | iTransformer (ICLR 2024) | Time-Series-Library |
| `autoformer` | Autoformer (NeurIPS 2021) | Time-Series-Library |
| `segrnn` | SegRNN (2023) | Time-Series-Library |
| `olinear` | OLinear (2025); its orthogonal basis is computed from your training split | our re-implementation from the paper |

To add your own backbone, write a predictor module that maps `[B·N, T, C]` to `[B·N, horizon, d]` (see [`graft/backbones/main/mlp.py`](graft/backbones/main/mlp.py)) and register it in [`graft/registry.py`](graft/registry.py).

**Metrics** follow the LargeST convention: MAE, RMSE and MAPE are averaged over the horizon, and missing readings are masked.

---

## 7. Repository layout

```text
run.py                        one entry point: python run.py --backbone <name> --data <folder>
graft/models/core.py          the alpha-gated graph block and the shared GraFT model
graft/backbones/main/         the paper's backbones: MLP, PatchTST, TSMixer
graft/backbones/additional/   additional backbones (Time-Series-Library models, OLinear)
graft/registry.py             backbone name -> model
graft/                        training engine, data loading, presets, settings
configs/paper_configs.json    exact settings of the 42 paper runs
tools/                        csv_to_graft.py (your data), make_synthetic.py (demo data)
tests/smoke_test.py           end-to-end checks on tiny synthetic data
```

---

## 8. Citation and acknowledgements

```bibtex
@article{graft2026,
  title   = {Spatial-as-Refinement: Gated Graph Operators over Strong Temporal Predictors for Scalable Spatio-Temporal Forecasting},
  author  = {AUTHORS},
  year    = {2026}
}
```

**Contact.** Questions and bug reports: open an issue or email vpham47@gatech.edu.

The training engine, data pipeline and evaluation protocol build on [LargeST](https://github.com/liuxu77/LargeST) (MIT) and [MAGE](https://github.com/PoorOtterBob/MAGE) (MIT). The additional backbones come from [Time-Series-Library](https://github.com/thuml/Time-Series-Library) (MIT); its license and our small modifications are in [`graft/backbones/additional/tslib`](graft/backbones/additional/tslib).
