# A Regret-Based Approach For Selecting Recommender Systems Under Evaluation Uncertainty

Code and results for the paper

> M.H. Zolfagharnasab and J.S. Cardoso. *Selecting a recommender system when the evaluation
> protocol is uncertain: A regret-based approach.*

The repository has two parts.

- `results/` holds every number in the paper. All tables and figures can be rebuilt from it
  in about a minute, with no data and no GPU.
- The rest of the repository reruns the experiments from the public datasets.

## 1. Install

```bash
git clone https://github.com/MsainZn/candidate-sampling-regret.git
cd candidate-sampling-regret
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or newer. The figure scripts also need `pdflatex` (any TeX Live or MacTeX).

## 2. Rebuild the tables and figures from the published results

```bash
python analysis/datasets_table.py        # Table 1
python figures/f3_structure.py           # Figure 3
python analysis/retained_population.py   # Table 2
python figures/f9_training.py            # Figure 4 and Table 5
python figures/f8_decision.py            # Tables 6 and 7
python figures/f11_reversal.py           # Figure 5
python analysis/b_analysis.py            # Table 8
python analysis/popularity_terms.py      # Table 9
python figures/f10_ranges.py             # Figure 6
python analysis/numerical_checks.py      # Appendix D, about two minutes
python analysis/numerical_check_d.py     # Appendix D, item (d)
```

Figures go to `results/figures/` and tables to `results/derived/`. The copies already in
those folders are the ones in the paper, so running the commands should reproduce them.

| Paper | Script | Output |
|---|---|---|
| Table 1 | `analysis/datasets_table.py` | `results/derived/datasets.csv` |
| Figure 3 | `figures/f3_structure.py` | `results/figures/f3_structure.pdf` |
| Table 2 | `analysis/retained_population.py` | `results/derived/retained_population.csv` |
| Table 3 | settings of every run | `results/<dataset>/meta.json` |
| Figure 4, Table 5 | `figures/f9_training.py` | `f9_training.pdf`, `results_table.tex` |
| Tables 6 and 7 | `figures/f8_decision.py` | `payoff_merged.tex`, `summary_merged.tex` |
| Figure 5 | `figures/f11_reversal.py` | `results/figures/f11_reversal.pdf` |
| Table 8 | `analysis/b_analysis.py` | `results/derived/b_analysis.json` |
| Table 9 | `analysis/popularity_terms.py` | `results/derived/popularity_terms.csv` |
| Figure 6 | `figures/f10_ranges.py` | `results/figures/f10_ranges.pdf` |
| Appendix D | `analysis/numerical_checks.py` (item (d) in `analysis/numerical_check_d.py`) | printed |

## 3. Rerun the experiments

### 3.1 Check the installation

```bash
python run_experiments.py --selftest     # no data needed, a few seconds
```

### 3.2 Get the data

Put the files in `data/`. ML-20M, Amazon:Books and Yelp are read as published. The other four
need one conversion command each.

| Dataset | Tag | Source | File in `data/` |
|---|---|---|---|
| MovieLens-20M | `ml20m` | [grouplens.org](https://grouplens.org/datasets/movielens/20m/) | `ml-20m/ratings.csv` |
| Netflix Prize | `netflix` | Netflix Prize training data, `combined_data_1.txt` to `combined_data_4.txt`, sampled to 100,000 users | `netflix.csv` |
| Goodreads | `goodreads_all` | [UCSD Book Graph](https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/), the eight genre interaction files | `goodreads_all.csv` |
| Amazon:Books | `amazon_books` | [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), 5-core rating-only `Books.csv` | `Books.csv` |
| Yelp | `yelp` | [Yelp Open Dataset](https://www.yelp.com/dataset) | `yelp_academic_dataset_review.json` |
| Google:CT | `google_connecticut` | [Google Local 2021](https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/), `review-Connecticut.json.gz` | `google_connecticut.csv` |
| HotelRec | `hotelrec` | [HotelRec](https://github.com/diegoantognini/HotelRec), `HotelRec.txt` | `hotelrec.csv` |

```bash
python preprocess/netflix.py      --src-dir data/netflix --out data/netflix.csv
python preprocess/goodreads.py    prep --dir data/goodreads --out data/goodreads_all.csv
python preprocess/google_local.py prep --src data/review-Connecticut.json.gz --out data/google_connecticut.csv
python preprocess/hotelrec.py     prep --src data/HotelRec.txt --out data/hotelrec.csv
```

`python preprocess/goodreads.py list` and `python preprocess/google_local.py list` print the
download commands for those two. Besides writing one table of user, item, rating and time, the
converters do the following.

- **Netflix** keeps a fixed sample of 100,000 users, each with the complete rating history. The
  sample is chosen by a hash of the user id, so every run picks the same users.
- **Goodreads** merges the eight genre files, keeps one copy of an interaction that appears in
  more than one of them, and drops shelf additions that carry no rating (rating 0).
- **HotelRec** drops reviews whose author is the placeholder `/undefined` or `null`.
- **Google:CT** drops the 27,166 reviews that have no user or business id.

The rating cut, the removal of repeated pairs, the split and the five-core are all done by
`run_experiments.py`, the same way on every dataset.

### 3.3 Run

```bash
./scripts/ml20m.sh          # one dataset
./scripts/run_all.sh        # all seven, the quickest first
./scripts/ml20m.sh --quick  # a short test run; its numbers are not meaningful
```

Each run writes `runs/<dataset>/` with the same files as `results/<dataset>/`. The defaults
of `run_experiments.py` are the settings of the paper, so the scripts only name the input
file and its columns. A run that stops can be started again, since fitted models are saved
in `runs/<dataset>/ckpt/` and loaded instead of retrained.

To rebuild the tables and figures from your own runs

```bash
python analysis/make_aggregated.py --results runs
python figures/f9_training.py --runs runs --out runs/figures --tex runs/derived
```

and `--bundle runs/aggregated --out runs/figures` for the other figure scripts.

### 3.4 What to expect

A rerun reproduces the published numbers up to small differences that come from the hardware
and the PyTorch version (CUDA, Apple MPS and CPU add in different orders). The published runs
used PyTorch 2.14, Apple MPS for six datasets and CUDA for Goodreads. Stage `e0` uses no GPU and
reproduces exactly everywhere, and so do the split, the scored positives and every candidate
draw, which are made with numpy from fixed seeds.

Training all seven datasets takes several days on one GPU. Google:CT and Yelp are the
quickest and HotelRec the slowest.

## 4. What is in `results/`

```
results/
  <dataset>/             one folder per dataset
    console_<stamp>.log  the full run log, including the training loss of every epoch
    meta.json            the settings of the run and the size of the log
    e0_diagnostics.csv   D, Lambda_U, Lambda_V and I_+(U;V) of the published log
    e1_arms.csv          the four training configurations x 4 seeds, scored with uniform candidates
    e2_crosscert.csv     the same frozen models scored under the three candidate laws
    e3_itemonly.csv      how much of each model's score its item embeddings alone explain
    e3b_capacity.csv     the larger model (dimension 128, twelve epochs, free per-item scalar),
                         one seed each for U, P and Q
    e4_invariance.csv    the matched sweep, training and scoring law moved together over
                         alpha, 2 seeds, scored on a second sample of 110,000 held-out positives
    e7_zbar.csv          a user-centred residual check (not used in the paper)
    metrics_long.csv     every number above in one long table
  aggregated/            the same CSVs with all seven datasets in one file each, and DIGEST.txt
  derived/               the tables of the paper, built by the scripts in section 2
  figures/               the figures of the paper
```

Names used in the files

| In the files | In the paper |
|---|---|
| `uniform`, `popularity`, `inbatch`, `inbatch_logq` | training configurations U, P, B, Q |
| `cert_ref` = `uniform`, `sqrt_pop`, `popularity` | candidate laws with alpha = 0, 0.5, 1 |
| `L_hat`, `L_lcb`, `half_width` | contrastive score, its lower bound and half-width (nats) |
| `Lambda_R` | popularity term under that candidate law, from the item counts of all held-out positives |
| `honest` | surrogate popularity term, the score that the item-only function log(P_tr/R) reaches with 1024 candidates (Monte Carlo) |
| `oracle` | the same for the item-only function log(P_+/R) |
| `drift` | divergence between held-out and training item frequencies |
| `ndcg20`, `hr10` | NDCG@20 and HR@10 |
| `p`, `D` in `e1_arms.csv` and `e4_invariance.csv` | computed from the held-out positives alone, whereas Tables 1 and 2 count all pairs |

## 5. Implementation details

These details of the code complement the descriptions in the paper.

1. **Records in Table 1.** For Netflix, Goodreads and HotelRec the records are counted on the
   converted file (section 3.2): 20,997,882 ratings of the 100,000 sampled Netflix users,
   79,224,894 rated Goodreads interactions after the eight genre files are merged, and
   49,047,625 HotelRec reviews after the placeholder authors are dropped. For the other four
   they are the published file, so the Google:CT count includes the 27,166 rows without an id.
2. **Scored positives.** The held-out positives of a dataset are put in one fixed random order.
   The training runs, the cross-evaluation and the larger model score the first 110,000. The
   matched sweep (`e4`) scores positions 30,001 to 140,000 of the same order, because the first
   30,000 are kept for the `e7` check.
3. **Surrogate popularity term.** The surrogate in Table 9 and Figure 6 (`honest`) is the score
   that the item-only function log(P_tr/R) reaches with 1024 candidates, estimated over 200,000
   draws. It lies below the expectation sum_v P_+(v) log(P_tr(v)/R(v)) by 0.067 nats on
   Goodreads under uniform candidates (35.2 against 36.2 percent of log K) and by at most
   0.017 nats in every other setting. The sampling estimate (M-1)/(2N) + (M-1)/(2N_tr) of Section 6.5
   is compared with `drift`, the divergence between held-out and training item frequencies,
   of which it accounts for 93 to 96 percent on every dataset. The identity check of Section 6.6
   compares `oracle - honest` with `drift` (column `identity_dev`).
4. **Larger model.** It is trained once (seed 0) for each of U, P and Q.
5. **Seed variation.** The largest seed-to-seed standard deviation of NDCG@20 (Goodreads, Q,
   uniform candidates) is 0.13 points as a population standard deviation over the four seeds
   and 0.15 points as a sample standard deviation.

## 6. Layout

```
run_experiments.py   entry point, one dataset per call
certlab/kit/         the pipeline
  data.py            reading a log, the random 90/10 split, the train-only five-core
  core.py            contrastive score, half-width, candidate laws
  model.py           two-tower model, training, scoring
  stages.py          one function per experiment (e0, e1-e3, e3b, e4 and e7)
  report.py          CSV output and the summary at the end of the log
  selftest.py        checks that need no data
scripts/             one run script per dataset
preprocess/          converters for Netflix, Goodreads, Google:CT and HotelRec
analysis/            Tables 1, 2, 8 and 9, Appendix D, and the aggregation of runs
figures/             Figures 3 to 6 and Tables 5 to 7
results/             the published results
```

## Citation

If you use this code, please cite the paper above.
