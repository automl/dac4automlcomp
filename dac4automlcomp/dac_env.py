from abc import abstractmethod
from functools import singledispatchmethod
from typing import Generic, Union

import gym
import numpy as np
from gym.utils import EzPickle, seeding

from dac4automlcomp.generator import Generator, GeneratorIterator, InstanceType


class DACEnv(gym.Env, EzPickle, Generic[InstanceType]):
    """The base Dynamic Algorithm Configuration gym environment. This class provides a high-level interface for the
    step-wise reconfiguration of a target algorithm across a target problem instance distribution (~ generator).

    This base class provides defaults for handling generator initialization and instance update on reset.
    Subclasses should implement gym's
        action_space (to be the target algorithm's configuration space)
        observation_space (to expose execution information available to condition reconfiguration on
        step (to model step-wise reconfiguration of the algorithm and the negated associated step-wise cost as reward)
        reset (optional, extended to model execution prior to the first reconfiguration point)
    and register it's instance_type=InstanceType as subclass (see PEP 487)

    """

    def __init__(
        self,
        generator: Generator[InstanceType],
        n_instances: Union[int, float] = np.inf,
    ):
        """

        Args: generator (Generator[InstanceType]): The target problem distribution, specified as an instance of a
        generator for instance type `InstanceType`
        n_instances: Optionally limit the number of instances that environment cycles through
        """
        self._generator = generator
        self._n_instances = n_instances
        self._current_instance: InstanceType  # Must only be changed by `reset`
        self.seed()

    def __init_subclass__(cls, instance_type: InstanceType):
        """Automatically register instance_type to `_to_instance` method in subclasses."""
        cls._to_instance.register(instance_type, lambda _, x: x)

    @abstractmethod
    def step(self, action):
        """Execute a single step of the target algorithm using configuration specified as action"""
        raise NotImplementedError

    @abstractmethod
    def reset(self, instance: Union[int, InstanceType]):
        """Start execution of the target algorithm on given problem instance.

        The default implementation of gym's reset cycles through a (possibly infinite) ordered sample of instances
        drawn from the target distribution (`self._generator`). The optional instance argument supports execution on
        a specific problem instance.

        Subclasses should call this method on reset using `super().reset(instance)`

        Args:
            instance (Union[int, InstanceType]): The problem instance the target algorithm should solve. This can be any
            instance of InstanceType, or an integer ~ i'th instance generated from the target distribution
        """
        self._current_instance = self._to_instance(instance)

    @property
    def current_instance(self) -> InstanceType:
        """Returns the instance that the target algorithm is currently solving."""
        if hasattr(self, "_current_instance"):
            return self._current_instance
        raise ValueError("Call super().reset(instance)")

    @singledispatchmethod
    def _to_instance(self, instance) -> InstanceType:
        """Generic method to parse `instance` to `InstanceType` type.
        New types can be registered using `DACEnv._to_instance.register`

        Example:

        @DACEnv._to_instance.register
        def _(self, instance: float):
            return self._to_instance(int(instance))
        """
        if instance is None:
            return next(self._generator_iterator)
        else:
            raise NotImplementedError

    @_to_instance.register
    def _(self, instance: int):
        return self._generator_iterator[instance]

    def seed(self, seed=None):
        """Initializes random state of internal problem instance generator."""
        self.np_random, _ = seeding.np_random(seed)
        self._generator.seed(seed)
        self._generator_iterator = GeneratorIterator(self._generator, self._n_instances)
        return [seed]
