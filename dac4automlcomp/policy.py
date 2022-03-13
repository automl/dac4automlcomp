from abc import ABC, abstractmethod
from pathlib import Path


class DACPolicy(ABC):
    """
    Base DAC policy class.

    Your submission should implement this interface (see examples in the respective track repos)

    """

    @abstractmethod
    def act(self, state):
        """
        Compute and return the configuration to be used next time step.

        This configuration may depend on
        - the current algorithm state (observations)
        - past algorithm states encountered and configurations used in the execution thus far
          (i.e., policies may be stateful between resets ~ within a single target algorithm execution)
        - random seed (i.e., policies may be pseudo-random)

        Args:
            state: The current algorithm state (observations)

        Returns:
            action: action to take
        """
        pass

    @abstractmethod
    def reset(self, instance):
        """Reset a policy's internal state.

        The reset method is there to support 'stateful' policies (e.g., LSTM),
        i.e., whose actions are a function not only of the current
        observations, but of the entire observation history from the
        current episode/execution. It is called at the beginning of the
        target algorithm execution (before the first call to act()) and also provides the policy
        with information about the target problem instance being solved.

        Args:
            instance: The problem instance the target algorithm to be configured is currently solving
        """
        pass

    @abstractmethod
    def seed(self, seed):
        """Sets random state of the policy.
        Subclasses should implement this method if their policy is stochastic
        """
        pass

    @abstractmethod
    def save(self, path: Path):
        """Saves the policy to given folder path."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path):
        """Loads the policy from given folder path."""
        pass


class DeterministicPolicy:
    """Base class to indicate a policy is deterministic."""

    def seed(self, seed):
        pass
