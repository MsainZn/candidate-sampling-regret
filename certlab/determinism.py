"""Pinning everything that can be pinned, and being explicit about what cannot.

Call ``enable()`` once, before torch is imported, and ``seed(n)`` before each model is
built. Together they make a run reproducible against itself on one machine at one torch
version. They do not make two backends agree, and nothing does: CUDA, MPS and CPU use
different kernels, those kernels reduce in different orders, and floating-point addition
is not associative. Pick one backend for a study and stay on it.

What is already exact everywhere, because it is computed in numpy and never touches a
GPU: ``p``, ``D``, the shift terms, the ceiling, the choice of certification positives and
every candidate draw. Stage ``e0`` must therefore reproduce bit for bit on any machine,
which makes it the integrity check to run first after moving.

The one setting here that changes numbers rather than merely fixing them is TF32. On
Ampere and later, PyTorch may compute an fp32 matmul in TF32, which carries ten bits of
mantissa instead of twenty-three. A CUDA run with it on and an MPS run are not even at the
same precision, so it is turned off.
"""
import os
import random


def enable(verbose=True):
    """Set every determinism switch. Must run before torch initialises CUDA."""
    # cuBLAS needs a fixed workspace for its reductions to be reproducible, and reads this
    # when the CUDA context is created, so it has to be set before the first CUDA call.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")

    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # fp32 stays fp32: no TF32 substitution in matmuls or convolutions
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    try:
        # warn_only: a handful of ops have no deterministic kernel, and failing the run
        # over one of them would be worse than recording that it was used
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    if verbose:
        print(f"determinism: cudnn deterministic, TF32 off, "
              f"CUBLAS_WORKSPACE_CONFIG={os.environ['CUBLAS_WORKSPACE_CONFIG']}", flush=True)


def seed(n):
    """Re-seed python, numpy and torch immediately before a model is built.

    Seeding once per run is not enough: fitting arms in sequence advances the global
    state, so the second arm would start from different weights and the arms would stop
    being a paired comparison.
    """
    import numpy as np
    import torch
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(n)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(n)


def environment(torch):
    """What this run ran on, for the manifest."""
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    info = {"device": dev, "torch": torch.__version__,
            "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "tf32_matmul": bool(getattr(getattr(torch.backends, "cuda", None), "matmul", None)
                                and torch.backends.cuda.matmul.allow_tf32)}
    if dev == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info
