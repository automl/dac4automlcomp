import abc
from dataclasses import dataclass, field
from functools import singledispatchmethod
from itertools import count, cycle
from typing import Generic, List, TypeVar, Union

import gym
import numpy as np
import torch
from gym.utils import EzPickle, seeding

T = TypeVar("T")


@dataclass(init=False)  # type: ignore
class Generator(Generic[T], abc.ABC):
    _internal_rng: np.random.RandomState = np.random.RandomState(None)
    _instance_seeds: List[int] = field(default_factory=lambda: [])

    @abc.abstractmethod
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


class DACEnv(gym.Env, EzPickle):
    def __init__(
        self,
        generator: Generator,
        n_instances: Union[int, float] = np.inf,
        device: str = "cpu",
    ):
        self.generator = generator
        self.n_instances = n_instances
        self.device = device
        self.seed()

    @singledispatchmethod
    def get_instance(self, instance):
        if instance is None:
            return next(self.generator_iterator)
        else:
            raise NotImplementedError

    @get_instance.register
    def _(self, instance: int):
        return self.generator_iterator[instance]

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)
        self.generator.seed(seed)
        self.generator_iterator = GeneratorIterator(self.generator, self.n_instances)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        return [seed]
