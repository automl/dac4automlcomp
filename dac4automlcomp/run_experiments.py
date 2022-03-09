import os
import time

import gym
import numpy as np


# Parts of the code inspired by the AutoML3 competition
from sys import argv, path
from os import getcwd
from os.path import join
verbose = True
root_dir = getcwd()     # e.g. '../' or pwd()
default_ingestion_dir = join(root_dir, "DAC4AutoML_ingestion_program")
default_input_dir = join(root_dir, "DAC4AutoML_sample_data")
default_output_dir = join(root_dir, "DAC4AutoML_sample_predictions")
default_hidden_dir = join(root_dir, "DAC4AutoML_sample_ref")
default_shared_dir = join(root_dir, "DAC4AutoML_shared")
default_submission_dir = join(root_dir, "DAC4AutoML_sample_code_submission")


def run_experiment(
        dac_policy_obj,
        max_steps=10_000,
        eval_every=1000,
        num_repetitions=5,
        num_eval_episodes=5,
):
    """
    This is the main experiment runner for the DAC4AutoML competition tracks.
    It takes an object of the DAC policy to be evaluated and runs the policy on
    a set of test environments with a training or test distribution of contexts
    for a fixed number of steps of the DACEnv. It repeats this a fixed number of
    times and returns the resulting performances as a Numpy array.

    #TODO Improve docstrings
    # defining train/test splits and the distribution for the contexts of the DACEnv.

    Parameters
    ----------
    dac_policy_obj : DACPolicy

    max_steps : int

    eval_every : int
        Evaluate every eval_every steps to be able to calculate the AUC of performance

    num_repetitions : int

    num_eval_episodes : int

    Returns
    ----------
    A Numpy array of performances

    """

    screen_output_width = os.get_terminal_size().columns
    repeat_equal_sign = (screen_output_width - 24) // 2
    set_ansi_escape = "\033[32;1m"
    reset_ansi_escape = "\033[0m"
    print(
        set_ansi_escape
        + "=" * repeat_equal_sign
        + "Running DAC4RL experiment"
        + "=" * repeat_equal_sign
        + reset_ansi_escape
    )
    print("\n\nLoaded object of type:", type(dac_policy_obj), "\n\n")
    print("Current working directory:", os.getcwd())

    # Get set of envs to run on:
    # Some people might want VecEnv?? #TODO
    envs = [
        gym.make("CartPole-v1")
    ]  # TODO Define training/test contexts here. Will the train and test splits
    # of the datasets in the ML track also need to be defined? For RL, just the contexts should be enough.
    # init_states = []
    for env in envs:
        # init_states.append()
        print("Running env:", str(env), "for the experiment.")
        tot_reward_list = []

        # Load policy?
        # Already done by submission.py

        for rep in range(num_repetitions):
            print("Executing repetiton number:", rep, "for the current env.")
            print("Evaluating the loaded policy for", max_steps, "steps.")

            state = env.reset()
            tot_reward = 0.0
            for iter in range(max_steps):
                # TODO should we evaluate for x episodes or x steps? What to do in case episode finishes in-between?
                action = env.action_space.sample()  # TODO dac_policy_obj.act(state)
                next_state, reward, done, info = env.step(action)
                tot_reward += reward

                state = next_state

                # if iter % eval_every == eval_every - 1:
                #     print("Evaluating policy for")

            print("Total reward for this repetition:", tot_reward)
            tot_reward_list.append(tot_reward)

        print(
            "Average total reward for env:", str(env), "is:", np.mean(tot_reward_list)
        )


def run_experiment_draft(
        dac_policy_obj,
        dac_env_obj,
        gen_seed,
        num_instances,
        policy_seed,
        time_limit_sec,
        **kwargs
):
    """
    This is the main experiment runner for the DAC4AutoML competition tracks.
    It takes an object of the DAC policy to be evaluated and tests the policy on
    a set of num_instances target problem instances and returns the resulting performances as a Numpy array.

    #TODO Improve docstrings
    # defining train/test splits and the distribution for the contexts of the DACEnv.   

    Parameters
    ----------
    dac_policy_obj: DACPolicy
    dac_env_obj: DACEnv
    gen_seed: int
    num_instances: int
    policy_seed: int
    time_limit_sec: int

    Returns
    -------
    A Numpy array of performances with shape (num_instances,)

    """
    # TODO: Exclude downloading the datasets in the evaluation time?
    start_time = time.time()  # TODO: Replace with a superior way of timing?

    total_rewards = np.zeros((num_instances,))
    dac_env_obj.seed(gen_seed)
    policy_seed_rng = np.random.RandomState(policy_seed)

    for i in range(num_instances):
        if time.time() - start_time < time_limit_sec:
            # TODO: To avoid stateful policies, we should actually re-load the policy here!
            # (and optionally pass a policy_loader and policy_loader_kwargs(?) as argument instead)
            dac_policy_obj.seed(policy_seed_rng.randint(1, 4294967295, dtype=np.int64))
            obs = dac_env_obj.reset()
            dac_policy_obj.reset(dac_env_obj.current_instance)
            done = False
            while not done:
                config = dac_policy_obj.act(obs)
                obs, reward, done, info = dac_env_obj.step(config)
                total_rewards[i] += reward
        else:
            # TODO: generate some warning that time budget has been exceeded
            total_rewards[i] = -np.inf

    return total_rewards


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="The experiment runner for the DAC4RL track.")
    # parser.add_argument(
    #     "--ray-dir", type=str, help="Location of ray_results/<expt> dir"
    # )
    # parser.add_argument(
    #     "--convert-dir",
    #     type=str,
    #     default="/tmp/ray_hpbandster/",
    #     help="Location to store converted Ray files in HpBandster format",
    # )

    # args = parser.parse_args()
    # TODO: Should be cmd arguments

    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        ingestion_dir = default_ingestion_dir
        input_dir = default_input_dir
        output_dir = default_output_dir
        hidden_dir = default_hidden_dir
        shared_dir = default_shared_dir
        submission_dir = default_submission_dir
    else:
        ingestion_dir = os.path.abspath(argv[1])
        input_dir = os.path.abspath(argv[2])
        output_dir = os.path.abspath(argv[3])
        hidden_dir = os.path.abspath(argv[4])
        shared_dir = os.path.abspath(argv[5])
        submission_dir = os.path.abspath(argv[6])
    if verbose:
        print("Using ingestion_dir: " + ingestion_dir)
        print("Using input_dir: " + input_dir)
        print("Using output_dir: " + output_dir)
        print("Using hidden_dir: " + hidden_dir)
        print("Using shared_dir: " + shared_dir)
        print("Using submission_dir: " + submission_dir)

    path.append(submission_dir)

    from solution import load_solution

    args = {'env_name': "sgd-v0", 'gen_seed': 666, 'policy_seed': 42, 'num_instances': 3, 'time_limit_sec': 10}

    # args = {'env_name': "dac4carl-v0", 'gen_seed': 666, 'policy_seed': 42, 'num_instances': 3, 'time_limit_sec': 86_400}

    policy = load_solution() #TODO assert it's a DACPolicy

    if args['env_name'] == "sgd-v0":
        import sgd_env
    else:  # "== 'dac4carl-v0'"
        import rlenv

    env = gym.make(args['env_name'])

    run_experiment_draft(policy, env, **args)
