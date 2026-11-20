# Detecting Benchmark Label Errors with Truth Discovery over Thousands of Model Sources

Code, results, and data pointers for the PVLDB Vol. 20 Experiments, Analyses & Benchmarks paper of the same title.
The study treats the per-item answers that public leaderboards release for thousands of language models as
claims of many sources, runs truth discovery and crowdsourcing methods over them, and scores how well each method
ranks the known label errors of MMLU and GSM8K (expert-verified) and HellaSwag (re-annotated).

| Where | What |
| --- | --- |
| this repository | method implementations (`depstudy/`), every experiment script (`scripts/`), all result files behind the tables and figures (`results/`), small inputs (`data/`), tests |
| release [`data-v1`](https://github.com/benchmark-label-audit/depstudy/releases/tag/data-v1) of this repository | the extracted answer logs (about 50 million answers of 2,999 Open LLM Leaderboard v1 models and the HELM models), the cached per-item method scores, and the HellaSwag re-annotations |

## Layout

```
depstudy/    data access (data.py, hf.py), answer normalization (textnorm.py), panels (panels.py), methods (methods.py)
scripts/     s0 ... s58, one script per experiment step; each docstring names its inputs and outputs
results/     result files (CSV, JSON, Parquet), checkpoints of the panel runs, and the figures
data/        model manifest, declared lineage, truth tables, fetch order; data/external holds the MMLU-Redux 2.0 inputs
tests/       synthetic-data tests of the methods and the data layer
```

## Setup

Python 3.11 with the versions used for the paper:

```
pip install -r requirements.txt
```

CrowdFM runs in a separate environment with PyTorch 2.10 and the authors' code and checkpoint
(https://github.com/liiuhaao/CrowdFM, commit `b08f4e70`); `scripts/crowdfm_adapter.py` documents the setup and
checks the checkpoint hash. CrowdFM is not redistributed here.

Large inputs and caches live outside the repository in `$DEPSTUDY_DATA` (default: `./raw`).

## Data

Download the assets of release [`data-v1`](https://github.com/benchmark-label-audit/depstudy/releases/tag/data-v1)
(3.3 GB in seven archives) and unpack every archive into `$DEPSTUDY_DATA`:

| Archive | Unpacks to | Content |
| --- | --- | --- |
| `answers_v1_mmlu_gsm8k.zip` | `preds/` | parsed MMLU (57 subjects, 5-shot) and GSM8K (5-shot) answers, one Parquet file per model |
| `answers_v1_hellaswag.zip` | `preds_hellaswag/` | HellaSwag (10-shot) answers, one file per model |
| `answers_v1_gsm8k_raw.zip` | `preds_gsm8k_raw/`, `gsm8k_raw_log.jsonl` | raw GSM8K generations with strict and lenient answer extraction |
| `answers_helm.zip` | `helm/` | HELM per-instance answers on MMLU and GSM8K |
| `fetch_logs.zip` | `fetch_log*.jsonl`, `fetch_hellaswag_shard*.jsonl` | per-model fetch records, read by `depstudy/data.py` |
| `hellaswag_relabel.zip` | `hellaswag_relabel/` | the stratified 500-item HellaSwag sample and its re-annotations |
| `item_scores.zip` | `item_scores/` | cached per-item scores of every method on every panel |

`SHA256SUMS.txt` in the release lists the checksums, and `MANIFEST.json` the file counts and sizes. Answer matrices are rebuilt from `preds/` on first use and
cached in `$DEPSTUDY_DATA/matrix_cache/`.

The answers can also be fetched again from the public sources (`s1_manifest.py`, `s3_fetch.py`, `s3b_hellaswag.py`,
`s27_gsm8k_raw.py`, `s3c_helm.py`, `s13_eee_fetch.py`). The fetch order is the random permutation in
`data/fetch_order_first3000.txt` (seed 20261115).

## Reproducing the paper

With `item_scores/` in place, the statistics, tables, and figures take minutes:

```
python scripts/s39_stats_v2.py        # single-panel AP, key-error AP, every contrast with 95% interval and Holm p
python scripts/s48_rq4_holm.py        # dependence experiments with Holm correction
python scripts/s42_figures_v2.py      # Figures 2-6
python scripts/s43_tables.py          # LaTeX rows of the supplementary tables
```

Without the cache, rerun the method steps first (`s5_gate.py`, `s19_logodds.py`, `s28_newmethods.py`, `s29_dose.py`,
`s37_mmlu_splitdose.py`, `s46_depen_rerun.py`, `s30_sensitivity.py`, `s31_gsm8k_flex.py`, `s6_sources.py`,
and the CrowdFM scripts `s15*.py`, `s34_crowdfm_new.py`).

| Paper element | Scripts | Result files |
| --- | --- | --- |
| Table 3 (benchmarks, defects, populations), Figure 1 (workload) | `s0_truth.py`, `s0b_extra_truth.py`, `s16_workload.py` | `data/truth_stats.json`, `results/workload.json`, `results/fig_workload.pdf` |
| Example 3.1, ranks of the revised GSM8K keys, HELM comparison, top-k queues per method, CrowdFM ties (Sections 3 and 5) | `s57_extra_numbers.py` | `results/extra_numbers.txt` |
| Repeat of Zheng et al. (Section 5.1) | `s11_repeat_zheng.py` | `results/repeat_zheng.csv` |
| Repeat of Dong et al.'s copy detection example | `s45_copy_validation.py` | `results/copy_reference.json` |
| Land and Bikel's benchmarks (Table 4) | `s13_eee_fetch.py`, `s14_landbikel.py`, `s41_pref_audit.py`, `s55_pref_boot.py`, `s43_tables.py` | `results/landbikel_table.json`, `results/pref_audit.json`, `results/pref_boot.json` |
| Single panels and key-error AP (Tables 5 and 6) | `s39_stats_v2.py`, `s53_gsm8k_prob_nf10.py`, `s54_ds_full_prob.py` | `results/singles_v2.parquet`, `results/contrasts_v2.csv` |
| Size and capability (Figure 2), sources (Figure 3), budget (Figure 4), defect types (Figure 5) | `s10_size_capability.py`, `s6_sources.py`, `s24_budget.py`, `s32_errortypes.py`, `s42_figures_v2.py` | `results/fig_*.pdf`, `results/budget.csv`, `results/errortypes_auroc.parquet` |
| Hyperparameter sensitivity (Section 5.4) | `s30_sensitivity.py`, `s43_tables.py` | `results/sensitivity.parquet` |
| Dependence experiments (Figure 6) | `s29_dose.py`, `s35_dep_analysis.py`, `s37_mmlu_splitdose.py`, `s40_panel_dependence.py`, `s48_rq4_holm.py` | `results/dose_effects.csv`, `results/splitdose_effects.csv`, `results/rq4_holm.csv`, `results/panel_dependence.csv` |
| Copy detection on the strong panels (Table 7) | `s45_copy_validation.py`, `s47_depen_grid.py`, `s52_depen_rounds.py` | `results/copy_*.csv`, `results/depen_grid.csv`, `results/depen_rounds.csv` |
| Headline contrasts and robustness (Table 8) | `s39_stats_v2.py`, `s48_rq4_holm.py`, `s51_lenient_robust.py`, `s50_lobo.py` | `results/contrasts_v2.csv` (item and subject-cluster schemes), `results/rq4_holm.csv`, `results/lenient_robust.json`, `results/lobo_ensemble.csv` |
| Scalability | `s18_scalability.py`, `s26_scal_figure.py` | `results/scalability.csv`, `results/scalability_hw.json` |
| Downstream ranking effect and verification budget | `s33_rankshift.py`, `s49_downstream.py`, `s56_wrongkey_answers.py` | `results/rankshift.json`, `results/downstream_estimand.csv`, `results/wrongkey_answers.json` |
| HellaSwag re-annotation (Sections 4.1 and 7) | `s38_hellaswag_relabel.py` | `results/hellaswag_relabel.json` |
| Contamination diagnostic (Section 7) | `s58_contamination.py` | `results/contamination.txt` |

Every contrast the paper reports, with its 95% percentile interval and Holm-corrected bootstrap p-value, is in
`results/contrasts_v2.csv` (research questions 1 to 3) and `results/rq4_holm.csv` (dependence experiments). The
Holm family size $m$ in Table 8 is the number of rows that share the family and scheme in these files.

The re-annotation script calls a language model through `ANNOTATOR_CMD`, a shell command that reads the prompt on
stdin and prints the reply. The released annotations make this call unnecessary for reproduction.

## Seeds

Fetch order 20261115; HellaSwag re-annotation sample 20261123; bootstrap and panel seeds are fixed in each script.

## Sources

| Source | Use |
| --- | --- |
| Open LLM Leaderboard v1 details (`open-llm-leaderboard-old/details_*`) | answers of the v1 population |
| HELM (MMLU, HELM Lite GSM8K) | answers of the HELM population |
| MMLU-Redux 2.0 | MMLU defect labels (`data/external`) |
| GSM8K-Platinum | GSM8K defect labels |
| PleIAs/hellaswag-annotations (round 2) | HellaSwag defect labels |
| Every Eval Ever (`evaleval/EEE_datastore`), `Writer/IRT-mislabeled-items` | Land and Bikel's benchmarks and released scores |
| Zheng et al.'s crowdsourcing datasets | repeat of Zheng et al. (not redistributed; see `s11_repeat_zheng.py`) |

Model cards, third-party code, and the Every Eval Ever records are not redistributed; the scripts fetch them.

## License

Code: MIT (`LICENSE`). Results, extracted answers, and annotations: CC BY 4.0 (`LICENSE-DATA.md`).
