from abc import abstractmethod
from functools import singledispatchmethod
from typing import Optional, Union, Generic, Type

import gym
import numpy as np
from gym.utils import EzPickle, seeding

from dac4automlcomp.generator import Generator, GeneratorIterator, InstanceType


class DACEnv(gym.Env, EzPickle, Generic[InstanceType]):
    """The base Dynamic Algorithm Configuration environment.
    Handles generator initialization and instance update. Subclasses
    should implement:
        step
        reset
    """
    def __init__(
        self,
        generator: Generator[InstanceType],
        n_instances: Union[int, float] = np.inf,
    ):
        """

        Args:
            generator (Generator[InstanceType]): Instance of a generator for instance type `InstanceType`
            n_instances: Number of instances that environment cycles through
        """
        self._generator = generator
        self._n_instances = n_instances
        self._current_instance: InstanceType  # Must only be changed by `reset`
        self.seed()

    def __init_subclass__(cls, instance_type: InstanceType):
        """Automatically register instance_type to `_get_instance` method in subclasses."""
        cls._get_instance.register(instance_type, lambda x: x)

    @abstractmethod
    def step(self, action):
        raise NotImplementedError

    @abstractmethod
    def reset(self, instance: Union[int, InstanceType]):
        """Resets the environment to given instance using `self._generator`.
        This method should be called in subclasses using `super().reset(instance)`
        Args:
            instance (Union[int, InstanceType]): DAC environment instance that holds a problem.
        """
        self._current_instance = self._get_instance(instance)

    @property
    def current_instance(self) -> InstanceType:
        """Returns the current instance that environment uses in current episode."""
        if hasattr(self, "_current_instance"):
            return self._current_instance
        raise ValueError("Call super().reset(instance)")

    @singledispatchmethod
    def _get_instance(self, instance):
        """Generic method to parse `instance` to `InstanceType` type.
        New types can be registered using `DACEnv._get_instance.register`

        Example:

        @DACEnv._get_instance.register
        def _(self, instance: float):
            return self._get_instance(int(instance))
        """
        if instance is None:
            return next(self.generator_iterator)
        else:
            raise NotImplementedError

    @_get_instance.register
    def _(self, instance: int):
        return self.generator_iterator[instance]

    @property
    def generator_iterator(self):
        """`Iterator` of `self._generator`.
        Can be used to access any instance of the generator by indexing.
        If it is used with `next` it will change the next instance the environment
        is going to use when the `reset` is called without an instance argument."""
        return self._generator_iterator

    def seed(self, seed=None):
        """Initialize random state of internal generator."""
        self.np_random, _ = seeding.np_random(seed)
        self._generator.seed(seed)
        self._generator_iterator = GeneratorIterator(self._generator, self._n_instances)
        return [seed]
