"""certkit -- contrastive certification for collaborative filtering.

This is the code that produced every number in the paper, split into readable modules.

Read the modules in this order:

    core.py      the contrastive score, its certificate, the reference family
    data.py      loading a rating log and splitting it without leakage
    model.py     the two-tower critic, its fitting and its certification pass
    stages.py    one function per experiment stage (e0, e1/e2/e3, e3b, e4/e7)
    report.py    CSV writing and the end-of-run summary
    selftest.py  dataset-free checks of the core identities
    cli.py       the command line; one invocation runs one dataset

Entry point: ``python run_experiments.py --dataset <name> --csv <path> ...``
The exact invocation used for each of the seven datasets is in ``scripts/``.
"""
__version__ = "1.0.0"
