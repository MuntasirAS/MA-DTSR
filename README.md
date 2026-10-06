# MA-DTSR

Simulation code, results and figures for the paper:

> **MA-DTSR: Multi-Agent Delay-Tolerant Semantic Routing for Resource Discovery in Disaster and Emergency Response Networks**
> Muntasir Al-Asfoor and Estabraq Makiyah
> School of Creative Arts, Technology and Engineering, Buckinghamshire New University, UK

MA-DTSR is a decentralised protocol for locating resources (medical supplies, transport, sensing assets) in networks with no fixed infrastructure. Each agent forwards a resource query to the neighbour whose advertised resource descriptor is semantically closest to the query, using only one-hop knowledge. This repository contains the simulator used in the paper, the raw results, and the scripts that produce the paper's tables and figures.

## Main findings

Across 36 settings of network size, hop budget and admissibility threshold, on 2,429 generated networks:

- Semantic forwarding raises discovery success by 9.9 percentage points over an uninformed random walk, at the same message cost.
- It uses 88% fewer messages than epidemic routing (12.9 against 106.7 per query); epidemic routing still finds more resources (43.6% against 24.8%).
- Mission utility favours semantic forwarding in dense networks (N = 300) and whenever a message costs more than roughly 0.4–0.9% of the value of a discovery.
- Greedy and softmin next-hop selection are statistically indistinguishable.

## Repository contents

### Simulator

| File | Purpose |
|---|---|
| `MA_DTSR_Step1_Mobility.py` | Random Waypoint mobility, link formation and contact database |
| `MA_DTSR_Step2_Descriptors.py` | 8-dimensional resource descriptors, query generation and similarity metrics (L1, Euclidean, weighted L1) |
| `MA_DTSR_Step3_Baselines.py` | Epidemic routing, random walk and the MA-DTSR semantic heuristic (greedy and softmin), plus the mission utility function |
| `MA_DTSR_Step4_RL.py` | Forward/wait Q-learning router with a linear function approximator |
| `MA_DTSR_Step5_Sweep.py` | Full parameter sweep and ablation studies |
| `WiSARD_Analysis.py` | Optional analysis of the WiSARD dataset annotations (requires a local copy of WiSARD) |

### Results

| File | Contents |
|---|---|
| `step5_full_results.csv` | Full sweep: 38,880 routed queries (protocol, success, hops, messages, match error, utility, N, TTL, epsilon, alpha, metric, seed) |
| `step5_ablation_A1_self_org.csv` … `A4_coop.csv` | Four ablation studies at N = 200, TTL = 20 |
| `step3_raw_results.csv`, `step3_summary.csv` | Baseline comparison on a single network (N = 100) |
| `step4_raw_results (1).csv`, `step4_summary (1).csv` | Baselines and RL routers on a single network (N = 100) |
| `step4_training_log_fwd.csv`, `step4_training_log_wait.csv` | Training logs of the forward-only and forward+wait routers |
| `wisard_stats.json` | Output of `WiSARD_Analysis.py` |
| `*.png` | Figures written by Steps 1–5 |

### Paper

| Path | Contents |
|---|---|
| `paper_v2/MA_DTSR_sn-article.pdf` | Manuscript in the Springer Nature template |
| `paper_v2/MA_DTSR_springer_nature_latex.zip` | LaTeX source of the manuscript |
| `paper_v2/figs/` | Figures used in the manuscript |
| `paper_v2/analysis/make_tables.py` | Rebuilds the paper's tables from `step5_full_results.csv` |
| `paper_v2/analysis/make_figures.py` | Rebuilds the paper's figures from `step5_full_results.csv` |
| `paper_v2/analysis/diagnostic_rl_equals_greedy.py` | Checks that the trained RL router, an untrained router and the greedy rule give identical outcomes |
| `paper_v2/analysis/diagnostic_wait_never_selected.py` | Checks that the Wait action is never selected |

## Requirements

Python 3.9 or later with:

```
numpy
pandas
matplotlib
tqdm
```

Install with `pip install numpy pandas matplotlib tqdm`.

## Reproducing the paper's tables and figures

These scripts read the stored results, so they run in seconds:

```bash
cd paper_v2/analysis
python make_tables.py     # writes tab_main.tex, tab_eps.tex, tab_metric.tex
python make_figures.py    # writes the four results figures to paper_v2/figs/
```

## Re-running the simulation

The step modules are written to be imported (they were developed in Google Colab). From the repository root:

```python
import MA_DTSR_Step1_Mobility as step1
import MA_DTSR_Step2_Descriptors as step2
import MA_DTSR_Step3_Baselines as step3
import MA_DTSR_Step4_RL as step4
import MA_DTSR_Step5_Sweep as step5

# Full sweep and ablations; writes step5_full_results.csv,
# step5_ablation_*.csv and fig15–fig21
df, ablations = step5.main_step5(step1, step2, step3, step4)
```

The individual steps can also be run on a single network:

```python
net = step1.main()                      # build and visualise one network
step2.main_step2(net)                   # populate descriptors
step3.main_step3(net, step2)            # baselines
step4.main_step4(net, step2, step3)     # RL routers
```

Notes:

- The full sweep covers N ∈ {100, 200, 300}, TTL ∈ {10, 20, 30, 40}, ε ∈ {0.5, 1.0, 1.5}, α ∈ {0, 0.01, 0.05} and three similarity metrics, with 30 seeds per combination. Expect roughly half an hour or more, depending on the machine.
- The scripts call `plt.show()`. When running outside a notebook, set `MPLBACKEND=Agg` so figures are saved without opening windows.
- Parts of the routing code draw from NumPy's global random state, so a re-run reproduces the stored results statistically, not number for number.

## Known limitations of the simulator

The paper reports these in its Section 5.7; they matter for anyone reusing the code.

- **Snapshot routing.** Each query is routed over a frozen connectivity snapshot. The simulation clock does not advance during routing, so descriptor age is always zero and the staleness parameter α has no effect.
- **The RL router behaves as the greedy rule.** The semantic feature is encoded as `score / (score + 1e-6)`, which is numerically indistinguishable from one for every candidate, so the learned weights cannot reorder forward actions. The `RL-MADTSR` rows in the result files are therefore reported in the paper as "MA-DTSR greedy".
- **The Wait action is never selected.** Exploration samples only Forward actions, so the wait weight never changes from its initial value.
- **Self-organisation is not implemented.** The contact-table bound K is not enforced and the neighbourhood replacement step (Algorithm 2 in the paper) is not called. Ablation A1 compares the semantic heuristic with a random walk.
- **The similarity metric affects only the softmin heuristic.** Epidemic routing, random walk and the RL router always use L1.
- **Descriptors are synthetic.** No dataset statistic enters the simulation.
- **Seeding.** Agent random streams are seeded as `seed + i`, so networks generated from nearby seeds share part of their node placement.

## Citation

If you use this code or data, please cite:

```bibtex
@article{alasfoor_makiyah_madtsr,
  author  = {Al-Asfoor, Muntasir and Makiyah, Estabraq},
  title   = {{MA-DTSR}: Multi-Agent Delay-Tolerant Semantic Routing for Resource
             Discovery in Disaster and Emergency Response Networks},
  note    = {Manuscript submitted for publication},
  year    = {2026}
}
```

<!-- TODO: replace with the journal reference and DOI once published. -->

## Licence

<!-- TODO: choose a licence (for example MIT for the code and CC BY 4.0 for the data and figures) and add a LICENSE file. -->

## Contact

Muntasir Al-Asfoor — muntasir.al-asfoor@bnu.ac.uk
