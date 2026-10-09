import random
import numpy as np
import os

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def capture_environment() -> dict:
    import platform
    import sys
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "env_vars": {k: v for k, v in os.environ.items() if "OMP" in k or "MKL" in k}
    }
