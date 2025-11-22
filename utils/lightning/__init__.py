import gc
from .predict_utils import *


def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()


def seed_everything(seed=42, workers=True):
    # 固定所有的随机种子，确保实验可复现
    L.seed_everything(seed=seed, workers=workers)
