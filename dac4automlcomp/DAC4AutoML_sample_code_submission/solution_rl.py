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

    ppo_action = {
        "algorithm": "PPO",
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "clip_range": 0.2,
    }
    
    return lambda x: ppo_action
