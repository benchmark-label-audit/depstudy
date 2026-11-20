# Superseded outputs

These files come from earlier stages of the analysis. The paper does not use them, and some of their numbers differ
from the paper (for example, `paper_numbers.md` predates the corrected copy detection and gives a GSM8K strong-panel
AccuCopy loss of about -0.014). They are kept as a record. The paper's numbers come from the files named in the main
README.

| File | Written by | Superseded by |
| --- | --- | --- |
| `paper_numbers.md` | an early summary of the contrasts | `results/contrasts_v2.csv`, `results/rq4_holm.csv` |
| `bootstrap_contrasts.csv`, `bootstrap_summary.json`, `bootstrap_recompute_check.csv` | `s17_bootstrap.py` (its item-score caches are still used) | `results/contrasts_v2.csv` (`s39_stats_v2.py`) |
| `singles_table.parquet` | `s20_singles.py` | `results/singles_v2.parquet` (`s39_stats_v2.py`) |
| `cells_effects.csv`, `cells_means.csv` | `s21_cells.py` | `results/cells_v2_effects.csv` (`s35_dep_analysis.py`), `results/rq4_holm.csv` |
| `budget.csv` | `s24_budget.py` (stable tie order, earlier GSM8K DS-1 variant) | Figure 4 from `s42_figures_v2.py`; queue counts in `results/extra_numbers.txt`; error removed in `results/downstream_estimand.csv` |
| `logs_s45_realoracle.txt` | an earlier run of `s45_copy_validation.py realoracle` | `results/copy_real_given_truth.csv` |

Rerunning the writing scripts puts new copies of these files into `results/`.
