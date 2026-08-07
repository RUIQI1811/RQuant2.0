# Third-party notices

## KunQuant 0.1.11

RQuant's local Alpha101 and Alpha158 formula modules contain modified portions
derived from KunQuant 0.1.11:

- `src/rquant/factors/libraries/alpha101.py`
- `src/rquant/factors/libraries/alpha158.py`

Upstream project: https://github.com/Menooker/KunQuant

Original installed source fingerprints:

- `KunQuant/predefined/Alpha101.py`: `14667961f60963f312165a58e045cb9d562672c651b00af3e9c2b39bb3c72c91`
- `KunQuant/predefined/Alpha158.py`: `bae7d8b1bea46c4b8426667227ff5a69dc6fb64d6b164a5f0c8a0f8301a535d9`

The source is licensed under Apache License 2.0. RQuant changed the files to
own their registration, metadata, implementation fingerprinting, and input
contracts. Alpha101 also applies RQuant's percentile time-series-rank semantics
and supplies the 19 formulas absent from KunQuant 0.1.11.

See `licenses/KunQuant-0.1.11-LICENSE.txt` for the complete license text.

