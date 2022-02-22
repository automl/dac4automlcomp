from abc import ABC, abstractmethod


class DACPolicy(ABC):
    @abstractmethod
    def act(self, state):
        ...

    @abstractmethod
    def reset(self, instance):
        pass

    @abstractmethod
    def save(self, f):
        ...

    @classmethod
    @abstractmethod
    def load(cls, f):
        ...
