from abc import ABC, abstractmethod
from itertools import count, cycle
from typing import Generic, List, TypeVar, Union

import numpy as np

InstanceType = TypeVar("InstanceType")


class Generator(Generic[InstanceType], ABC):
    """Instance generator modeling a target problem distribution for DAC as a (possibly infinite) sequence of instances
    that are distributed accordingly"""

    def __init__(self):
        self.seed(None)

    @abstractmethod
    def random_instance(self, rng: np.random.RandomState) -> InstanceType:
        """Sample a random instance.
        This function should be implemented by subclasses and should be deterministic,
        meaning given the same random state it should always return the same instance."""
        raise NotImplementedError

    def get_instance(self, instance_idx: int) -> InstanceType:
        """Return `instance_idx`th instance from the instance sequence.
        Using this function does not affect deterministic behaviour of the generator"""
        while instance_idx >= len(self._instance_seeds):
            seed = self._internal_rng.randint(1, 4294967295, dtype=np.int64)
            self._instance_seeds.append(seed)
        seed = self._instance_seeds[instance_idx]
        rng = np.random.RandomState(seed)
        return self.random_instance(rng)

    def seed(self, seed):
        """Initialize the random state of the generator, fully determining the order in which instances are generated"""
        self._internal_rng = np.random.RandomState(seed)  # Do not use it!
        self._instance_seeds: List[int] = []


class GeneratorIterator(Generic[InstanceType]):
    """Generator iterator to cycle through the first n_instances generated instances by generator."""

    def __init__(
        self,
        generator: Generator[InstanceType],
        n_instances: Union[int, float] = np.inf,
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
