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
depstudy/         data access (data.py, hf.py), answer normalization (textnorm.py), panels (panels.py), methods (methods.py)
scripts/          s0 ... s62, one script per experiment step; each docstring names its inputs and outputs
results/          result files (CSV, JSON, Parquet), checkpoints of the panel runs, and the figures
results/archive/  superseded outputs of earlier stages, kept as a record (see its README); the paper does not use them
data/             model manifest, declared lineage, truth tables, fetch order; data/external holds the MMLU-Redux 2.0 inputs
tests/            synthetic-data tests of the methods and the data layer
```

`data/lineage_stats.json` counts families before case normalization (12 with at least five members);
`results/workload.json` holds the normalized counts that the paper reports (243 families, 11 with at least five).

## Setup

Python 3.11 with the versions used for the paper:

```
pip install -r requirements.txt
```

Large inputs and caches live outside the repository in `$DEPSTUDY_DATA` (default: `./raw`).

Two steps need inputs that are not redistributed here:

* **CrowdFM** (`s15*.py`, `s34_crowdfm_new.py`) runs in a separate environment with PyTorch 2.10 and the authors'
  code and checkpoint (https://github.com/liiuhaao/CrowdFM, commit `b08f4e70`). `scripts/crowdfm_adapter.py`
  documents the setup and checks the checkpoint hash. Its scores are in `item_scores/` and
  `results/crowdfm_scores.parquet`.
* **Zheng et al.'s crowdsourcing datasets** (`s11_repeat_zheng.py`): download `datasets.zip` from
  https://github.com/TsinghuaDatabaseGroup/CrowdTruthInference (SHA256
  `c68ee01613da6dd6e2405c3252b73bb1199bc263909588bf952f57fc7d84323c`) and unpack it into
  `$DEPSTUDY_DATA/zheng2017_datasets/raw/`, so that `raw/datasets/` holds one folder per dataset
  (`d_Duck Identification/`, `s4_Relevance/`, ...).

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

`SHA256SUMS.txt` in the release lists the checksums, and `MANIFEST.json` the file counts and sizes. The archives
hold per-model answer files; the item x model answer matrices are built from them on first use and cached in
`$DEPSTUDY_DATA/matrix_cache/`.

The answers can also be fetched again from the public sources (`s1_manifest.py`, `s3_fetch.py`, `s3b_hellaswag.py`,
`s27_gsm8k_raw.py`, `s3c_helm.py`). The fetch order is the random permutation in `data/fetch_order_first3000.txt`
(seed 20261115).

## Reproducing the paper

### From the cached scores

With every archive of the release unpacked, the statistics, tables, and figures take minutes. These scripts read
the cached scores and also the answer archives and fetch logs, from which they rebuild the item tables, the
populations, and the items shared with HELM.

```
python scripts/s39_stats_v2.py        # Tables 5, 6, 8: single-panel AP, key-error AP, every contrast with 95% interval and Holm p
python scripts/s48_rq4_holm.py        # dependence experiments with Holm correction (reads the s35/s37 bootstrap outputs in results/)
python scripts/s57_extra_numbers.py > results/extra_numbers.txt   # numbers stated in the text (see the table below)
python scripts/s16_workload.py        # Figure 1 and the workload statistics
python scripts/s42_figures_v2.py      # Figures 2-6
python scripts/s43_tables.py          # Table 4 and the sensitivity ranges; needs the Every Eval Ever records of s13 and s36
```

`s39_stats_v2.py` stops before writing when a row of the shipped `singles_v2.parquet` or `contrasts_v2.csv`
cannot be recomputed, since a missing score file would also change the Holm family sizes.

### Without the cache

Each method step resumes where it stopped: a panel counts as finished when its checkpoint and every item-score
file it writes exist, so a run with an empty `item_scores/` rescores every panel even though the checkpoints in
`results/` are shipped. In this order:

```
python scripts/s17_bootstrap.py                     # item scores of the baseline methods on the single and size panels
python scripts/s19_logodds.py --workers 6           # log-odds scores of DS-1, full DS, ACCU, AccuCopy
python scripts/s28_newmethods.py --workers 6        # GLAD, CATD, entropy, n_f = 10 DS-1, the off-benchmark strong panel
python scripts/s29_dose.py build                    # dependence ladders and capability-matched panels
python scripts/s29_dose.py run --workers 6
python scripts/s29_dose.py manip
python scripts/s37_mmlu_splitdose.py build          # MMLU split-half ladder, pool of 400 models
python scripts/s37_mmlu_splitdose.py run
python scripts/s37_mmlu_splitdose.py analyze
DOSE_POOL=200 python scripts/s37_mmlu_splitdose.py build    # the ladder plotted in Figure 6 (pool of 200); run and analyze likewise
python scripts/s46_depen_rerun.py run --workers 6   # AccuCopy and AccuCopy-NonUni on every panel
DEPEN_VARIANT=key python scripts/s46_depen_rerun.py run --workers 6   # the same with the shipped keys as truth
python scripts/s30_sensitivity.py --workers 6       # hyperparameter sensitivity
python scripts/s31_gsm8k_flex.py --workers 6        # lenient GSM8K extraction
python scripts/s53_gsm8k_prob_nf10.py               # DS-1 probability scores with n_f = 10 on GSM8K
python scripts/s54_ds_full_prob.py                  # full DS probability scores on MMLU
python scripts/s35_dep_analysis.py --B 500          # lineage x dependence cells, ladders, curation controls
python scripts/s40_panel_dependence.py
python scripts/s45_copy_validation.py ref           # Dong et al.'s example
python scripts/s45_copy_validation.py synth
python scripts/s45_copy_validation.py semi          # simulated independent sources (Table 7)
python scripts/s45_copy_validation.py realoracle    # real panels with the shipped keys as truth (Table 7)
python scripts/s47_depen_grid.py --workers 4        # the 32 copy-detection settings on the GSM8K ladder
python scripts/s52_depen_rounds.py                  # round-by-round copy detection diagnostics
python scripts/s59_ebcc.py run --workers 6          # EBCC and warm-started AccuCopy on the single panels and ladders
python scripts/s59_ebcc.py copyrates                # injected copiers at copy rates 0.3, 0.5, 0.8 (Section 5.5)
python scripts/s59_ebcc.py stats                    # EBCC rows of Tables 5, 6, 8; warm-start row of Table 7
python scripts/s61_precreccorr.py run --workers 6   # PrecRecCorr, and AccuCopy with Dong et al.'s vote order
python scripts/s61_precreccorr.py stats             # PrecRecCorr rows of Tables 5, 6, 8; the vote-order ablation
python scripts/s62_irt_gap.py                       # IRT-4PL replication diagnostics on Land and Bikel's records (Section 5.1)
python scripts/s32_errortypes.py                    # defect types (Figure 5)
python scripts/s33_rankshift.py                     # leaderboard effect of the corrected labels
python scripts/s49_downstream.py                    # accuracy error removed by verification
python scripts/s60_budget_ci.py                     # 95% intervals of the accuracy error removed
python scripts/s50_lobo.py                          # leave-one-benchmark-out choice of the rank average
python scripts/s51_lenient_robust.py                # lenient-extraction column of Table 8
python scripts/s56_wrongkey_answers.py
python scripts/s58_contamination.py > results/contamination.txt
```

CrowdFM (`s15_crowdfm.py`, `s15b_crowdfm_extra.py`, `s15c_crowdfm_sources.py`, `s34_crowdfm_new.py`) runs in its own
environment (see Setup). The prior-study repeats need their own inputs: `s11_repeat_zheng.py` the Zheng et al.
datasets; `s13_eee_fetch.py` and `s36_eee_pref.py` fetch the Every Eval Ever records for `s14_landbikel.py`,
`s41_pref_audit.py`, `s55_pref_boot.py`, and `s43_tables.py`. `s18_scalability.py` times every method and records
the hardware in `results/scalability_hw.json`. Then run the commands of the cached route.

### Where each number comes from

| Paper element | Scripts | Result files |
| --- | --- | --- |
| Table 3 (benchmarks, defects, populations), Figure 1 and Section 4.5 (workload) | `s0_truth.py`, `s0b_extra_truth.py`, `s16_workload.py` | `data/truth_stats.json`, `results/workload.json` (`share_items_majority_disagrees_parseable` for the shares in Section 4.5), `results/fig_workload.pdf` |
| Example 3.1, ranks of the revised GSM8K keys, HELM comparison and same-false-value rates (Section 5.2), top-k queues per method, log-odds scale, rank agreement of ACCU and DS-1, weak-source weights (Section 5.4), NonUni coincidence rates (Section 5.5), CrowdFM ties | `s57_extra_numbers.py` | `results/extra_numbers.txt` |
| Repeat of Zheng et al. (Section 5.1) | `s11_repeat_zheng.py` | `results/repeat_zheng.csv` |
| Repeat of Dong et al.'s copy detection example | `s45_copy_validation.py ref` | `results/copy_reference.json` |
| Land and Bikel's benchmarks (Table 4) | `s13_eee_fetch.py`, `s36_eee_pref.py`, `s14_landbikel.py`, `s41_pref_audit.py`, `s55_pref_boot.py`, `s43_tables.py` | `results/landbikel_repeat.json`, `results/landbikel_table.json`, `results/pref_audit.json`, `results/pref_boot.json` |
| Single panels and key-error AP (Tables 5 and 6) | `s39_stats_v2.py`, `s53_gsm8k_prob_nf10.py`, `s54_ds_full_prob.py` | `results/singles_v2.parquet`; the GSM8K DS-1 (probability) row with $n_f=10$ is in `results/gsm8k_prob_nf10.csv` (also `ds_onecoin_nf10` in `singles_v2.parquet`), full DS by probability in `results/ds_full_prob.csv` |
| Size and capability (Figure 2), sources (Figure 3), verification queue (Figure 4) | `s10_size_capability.py`, `s6_sources.py`, `s42_figures_v2.py` | `results/fig_sizecap.pdf`, `results/fig_sources.pdf`, `results/fig_budget.pdf`; contrasts in `results/contrasts_v2.csv` |
| Defect types (Figure 5) | `s32_errortypes.py`, `s56_wrongkey_answers.py`, `s42_figures_v2.py` | `results/errortypes_auroc.parquet`, `results/errortypes_concentration.parquet`, `results/wrongkey_answers.json` |
| Hyperparameter sensitivity (Section 5.4) | `s30_sensitivity.py`, `s43_tables.py` | `results/sensitivity.parquet` |
| Top-200 overlap of the methods (Section 5.4) | `s25_complement.py` | `results/complement.csv` |
| Dependence experiments (Figure 6) | `s29_dose.py`, `s35_dep_analysis.py`, `s37_mmlu_splitdose.py`, `s40_panel_dependence.py`, `s48_rq4_holm.py` | `results/dose_effects.csv`, `results/splitdose_effects_p200.csv`, `results/rq4_holm.csv`, `results/panel_dependence.csv` |
| Copy detection on the strong panels (Table 7) | `s45_copy_validation.py semi`, `s45_copy_validation.py realoracle`, `s46_depen_rerun.py`, `s47_depen_grid.py`, `s52_depen_rounds.py` | real panels at Dong et al.'s settings and NonUni: `results/singles_v2.parquet` (key-error AP, tag `small_strong`) and `results/depen_diag/<bench>__single_small_strong.json` (flags, `shareD50_uni` and `shareD50_nu`); the $c=0.95$ and $\varepsilon=0.05$ rows: `results/depen_grid.csv` (tag `single_small_strong`, `anchor` inferred, `nonuni` False); simulated rows: `results/copy_semisynthetic.csv`; same-family / different-family rows: `results/copy_real_given_truth.csv`; rounds: `results/depen_rounds.csv` |
| Headline contrasts and robustness (Table 8) | `s39_stats_v2.py`, `s48_rq4_holm.py`, `s51_lenient_robust.py`, `s50_lobo.py` | `results/contrasts_v2.csv` (item and subject-cluster schemes; the GSM8K dependence row in `results/rq4_holm.csv`), lenient column in `results/lenient_robust.json`, held-out rank average in `results/lobo_ensemble.csv` |
| Scalability | `s18_scalability.py`, `s26_scal_figure.py` | `results/scalability.csv`, `results/scalability_hw.json` |
| Leaderboard effect and verification budget (Section 5.2) | `s33_rankshift.py`, `s49_downstream.py` | `results/rankshift.json`, `results/downstream_estimand.csv` |
| HellaSwag re-annotation (Sections 4.1 and 7) | `s38_hellaswag_relabel.py` | `results/hellaswag_relabel.json` |
| Contamination diagnostic (Section 7) | `s58_contamination.py` | `results/contamination.txt` |

Every contrast the paper reports, with its 95% percentile interval and Holm-corrected bootstrap p-value, is in
`results/contrasts_v2.csv` (research questions 1 to 3) and `results/rq4_holm.csv` (dependence experiments). The
Holm family size $m$ in Table 8 is the number of rows that share the family, metric, and scheme in
`contrasts_v2.csv`; the `diag` family (log-odds versus probability ranking of DS-1) pools the three benchmarks. In
`rq4_holm.csv` a family is also split by design, benchmark, and effect. Bootstrap p-values have a floor of $1/B$
($B=2{,}000$ for items, 500 for the dependence experiments).

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
| Zheng et al.'s crowdsourcing datasets | repeat of Zheng et al. (not redistributed; see Setup) |

Model cards, third-party code, and the Every Eval Ever records are not redistributed; the scripts fetch them.

## License

Code: MIT (`LICENSE`). Results, extracted answers, and annotations: CC BY 4.0 (`LICENSE-DATA.md`).
