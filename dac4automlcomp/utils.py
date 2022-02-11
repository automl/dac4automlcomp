import numpy as np
import torch
from gym import spaces


def generate_seed(idx, rng):
    if rng is not generate_seed.rng:
        generate_seed.cache = []
        generate_seed.rng = rng
    while idx >= len(generate_seed.cache):
        seed = rng.randint(1, 4294967295, dtype=np.int64)
        generate_seed.cache.append(seed)
    return generate_seed.cache[idx]


generate_seed.rng = None


def get_instance(generator, idx, rng):
    seed = generate_seed(idx, rng)
    rng = np.random.RandomState(seed)
    instance = generator(rng)
    return instance, seed
