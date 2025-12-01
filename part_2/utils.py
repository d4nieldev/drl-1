import numpy as np
import torch
import torch.nn as nn

import random

def set_seed(seed: int = 0) -> None:
    """Set as many RNG seeds as possible for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
