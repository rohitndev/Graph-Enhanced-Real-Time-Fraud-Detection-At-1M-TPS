# Data

This project trains and evaluates on a **synthetic transaction graph** that
reproduces the structure of the public fraud datasets referenced in the design,
with realistic coordinated **fraud rings** planted on top.

## Generator

[`generate_data.py`](./generate_data.py) builds a transaction log where each
event links an **account**, a **device**, and a **merchant**. Legitimate traffic
spreads naturally across the graph (with a few shared "hub" devices such as
public terminals), while fraud rings funnel several accounts through one or two
shared devices and repeatedly target a small merchant pool — the coordinated
pattern that is only visible in the graph.

```bash
# Generate a standalone dataset
python -m data.generate_data --rows 50000 --rings 12 --fraud-ratio 0.012 \
    --out data/transactions.parquet
```

## Real datasets

To run on real data, drop either of the following into this folder and point the
pipeline at it (both share the account / device / merchant schema after light
mapping):

- **IEEE-CIS Fraud Detection** — https://www.kaggle.com/c/ieee-fraud-detection
- **PaySim Synthetic Transactions** (6M+ rows) — https://www.kaggle.com/datasets/ealaxi/paysim1

Generated and downloaded data files are git-ignored.
