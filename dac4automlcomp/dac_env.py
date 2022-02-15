import gym
import sys
import numpy as np
from gym.utils import EzPickle, seeding
from typing import Optional, Union, List, Dict, Any
from collections import namedtuple
from itertools import count, cycle
if sys.version_info.minor >= 8:
    from typing import Protocol
else:
    from typing_extensions import Protocol # type: ignore

import dac4automlcomp.utils as utils


# Wraps named tuple in order to define instances with different fields
class Instance(object):
    def __init__(self, name: str, fields: List[str]):
        self.name = name
        self.i = namedtuple(name, fields)

    def __getattr__(self, item):
        if item == "name":
            return self.name
        else:
            return getattr(self.instance, item)


class Generator(Protocol):
    def __call__(self, rng: np.random.RandomState, **kwargs: int) -> Instance:
        ...


class DACEnv(gym.Env, EzPickle):
    def __init__(self,
        generator: Generator = None,
        instance_set: Dict[Any, Instance] = None,
        n_instances: Union[int, float] = np.inf,
        device: str = "cpu",
        cutoff: int = 10000
    ):
        if generator:
            self.generator = generator
        elif instance_set:
            self.instance_set = instance_set
        else:
            raise ValueError("Either instance set or instance generator is required")

        self.n_instances = n_instances
        self.device = device
        self.seed()
        self.n_step = None
        self.instance = None
        self.cutoff = cutoff

        if self.n_instances == np.inf:
            self.instance_count = count(start=0, step=1)
        else:
            self.instance_count = cycle(range(self.n_instances))

    def _step(self):
        self.n_step += 1
        done = self.n_step >= self.cutoff
        return done

    def _reset(self, instance: Optional[Union[Instance, int]] = None):
        self.n_step = 0

        if isinstance(instance, Instance):
            self.instance = instance
            seed = None
        else:
            if instance is None:
                instance_idx = next(self.instance_count)
            elif isinstance(instance, int):
                assert instance < self.n_instances
                instance_idx = instance
            else:
                raise ValueError("Invalid instance argument, either provide type 'Instance' or an 'int' ID.")

            if self.generator:
                self.instance, seed = utils.get_instance(
                    self.generator, instance_idx, self.np_random
                )
            else:
                self.instance = self.instance_set[instance_idx]
                seed = None

        assert isinstance(self.instance, Instance)
        # Get raw instance fields
        self.instance = self.instance.i
        return seed

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
