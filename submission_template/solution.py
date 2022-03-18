from pathlib import Path

from dac4automlcomp.policy import DACPolicy


def load_solution(path: Path = Path(".")) -> DACPolicy:
    """
    Load Solution.

    Serves as an entry point for the competition evaluation.

    Place your code loading your policy here.

    Parameters
    ----------
    path: Path
        The path to your submission directory.

    Returns
    -------
    DACPolicy
    """
    raise NotImplementedError
