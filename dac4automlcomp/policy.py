from abc import ABC, abstractmethod


class DACPolicy(ABC):
    """Base policy class."""

    @abstractmethod
    def act(self, state):
        """
        Compute and return environment action

        Args:
            state: Environment state (observations)

        Returns:
            action: action to take
        """
        pass

    @abstractmethod
    def reset(self, instance):
        """Reset policy to the initial state.
        The reset method is there to support 'stateful' policies,
        i.e., whose actions are a function not only of the current
        observations, but of the entire observation history from the
        current episode, e.g., you would need it if you would wanted
        to learn an LSTM policy. It is called at the beginning of the
        episode (before the first call to act()). It is also used to
        set instance features (as these are anyway fixed throughout
        the episode)
        Args:
            instance: Current instance the policy is acting on.
        """
        pass

    @abstractmethod
    def save(self, f):
        pass

    @classmethod
    @abstractmethod
    def load(cls, f):
        pass
