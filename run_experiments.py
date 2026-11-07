#!/usr/bin/env python3
"""Run the full protocol on one dataset.

    python run_experiments.py --selftest
    python run_experiments.py --dataset ml20m --csv data/ml-20m/ratings.csv

Every flag is documented by ``--help``. The seven invocations used in the paper are
in ``scripts/``, one shell script per dataset.

Re-running is safe: an existing checkpoint is loaded rather than retrained, so an
interrupted run resumes where it stopped.
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# before torch is imported anywhere: cuBLAS reads its workspace setting when the CUDA
# context is created, and TF32 has to be off before the first matmul
from certlab.determinism import enable as enable_determinism
enable_determinism()

from certlab.kit.cli import main

if __name__ == "__main__":
    sys.exit(main())
