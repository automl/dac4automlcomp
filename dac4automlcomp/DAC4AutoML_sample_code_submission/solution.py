from dac4automlcomp.policy import DACPolicy


def load_solution() -> DACPolicy:
    """
    Load Solution.

    Serves as an entry point for the competition evaluation.

    Place your code loading your policy here.

    Returns
    -------
    DACPolicy
    """
    from sgd_env.policy.schedulers import CosineAnnealingLRPolicy
    
    return CosineAnnealingLRPolicy(0.01)
