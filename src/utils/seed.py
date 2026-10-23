import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def worker_init_fn(worker_id: int):
    """Truyền vào DataLoader(worker_init_fn=...) để seed cả các worker."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)