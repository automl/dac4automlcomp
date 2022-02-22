from abc import ABC, abstractmethod
from itertools import count, cycle
from typing import Generic, List, TypeVar, Union

import numpy as np

T = TypeVar("T")


class Generator(Generic[T], ABC):
    def __init__(self):
        self._internal_rng: np.random.RandomState = np.random.RandomState(None)
        self._instance_seeds: List[int] = []

    @abstractmethod
    def random_instance(self, rng: np.random.RandomState) -> T:
        pass

    def get_instance(self, instance_idx) -> T:
        while instance_idx >= len(self._instance_seeds):
            seed = self._internal_rng.randint(1, 4294967295, dtype=np.int64)
            self._instance_seeds.append(seed)
        seed = self._instance_seeds[instance_idx]
        rng = np.random.RandomState(seed)
        return self.random_instance(rng)

    def seed(self, seed):
        self._internal_rng = np.random.RandomState(seed)  # Do not use it!
        self._instance_seeds: List[int] = []


class GeneratorIterator(Generic[T]):
    def __init__(
        self, generator: Generator[T], n_instances: Union[int, float] = np.inf
    ):
        self.generator = generator
        self.n_instances = n_instances
        self.instance_count: Union[cycle[int], count[int]]
        if self.n_instances == np.inf:
            self.instance_count = count(start=0, step=1)
        else:
            assert isinstance(self.n_instances, int)
            self.instance_count = cycle(range(self.n_instances))

    def __iter__(self):
        return self

    def __next__(self):
        instance_idx = next(self.instance_count)
        return self.generator.get_instance(instance_idx)

    def __getitem__(self, instance_idx):
        return self.generator.get_instance(instance_idx)
