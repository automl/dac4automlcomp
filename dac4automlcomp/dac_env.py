from functools import singledispatchmethod
from typing import Optional, TypeVar, Union

import gym
import numpy as np
from gym.utils import EzPickle, seeding

from dac4automlcomp.generator import Generator, GeneratorIterator

T = TypeVar("T")


class DACEnv(gym.Env, EzPickle):
    def __init__(
        self,
        generator: Generator[T],
        n_instances: Union[int, float] = np.inf,
    ):
        self.generator = generator
        self.n_instances = n_instances
        self.seed()

    def reset(self, instance: Optional[Union[int, T]]):
        pass

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
        return [seed]
