import argparse
import os

import time
import gym
import numpy as np
import warnings


# Parts of the code inspired by the AutoML3 competition
from sys import argv, path
from os import getcwd
from os.path import join

verbose = True
root_dir = getcwd()  # e.g. '../' or pwd()
# default_ingestion_dir = join(root_dir, "DAC4AutoML_ingestion_program")
# default_input_dir = join(root_dir, "DAC4AutoML_sample_data")
# default_output_dir = join(root_dir, "DAC4AutoML_sample_predictions")
# default_hidden_dir = join(root_dir, "DAC4AutoML_sample_ref")
# default_shared_dir = join(root_dir, "DAC4AutoML_shared")
# default_submission_dir = join(root_dir, "DAC4AutoML_sample_code_submission")


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

    Args:
        dac_policy_obj (DACPolicy):

        max_steps (int):

        eval_every (int): Evaluate every eval_every steps to be able to calculate the AUC of performance

        num_repetitions (int):

        num_eval_episodes (int):

    Returns:
        A Numpy array of performances # TODO: This function returns nothing?

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
    **kwargs,
):
    """
    This is the main experiment runner for the DAC4AutoML competition tracks.
    It takes an object of the DAC policy to be evaluated and tests the policy on
    a set of num_instances target problem instances and returns the resulting performances as a Numpy array.

    #TODO Improve docstrings
    # defining train/test splits and the distribution for the contexts of the DACEnv.

    Args:
        dac_policy_obj (DACPolicy):
        dac_env_obj (DACEnv):
        gen_seed (int):
        num_instances (int):
        policy_seed (int):
        time_limit_sec (int):

    Returns:
        total_rewards: A Numpy array of performances with shape (num_instances,)

    """
    # TODO: Exclude downloading the datasets in the evaluation time?
    start_time = time.time()  # TODO: Replace with a superior way of timing?

    screen_output_width = 80 # os.get_terminal_size().columns
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

    total_rewards = np.zeros((num_instances,))
    dac_env_obj.seed(gen_seed)
    policy_seed_rng = np.random.RandomState(policy_seed)

    for i in range(num_instances):
        if time.time() - start_time < time_limit_sec:
            # TODO: To avoid stateful policies, we should actually re-load the policy here!
            # (and optionally pass a policy_loader and policy_loader_kwargs(?) as argument instead)
            dac_policy_obj.seed(
                policy_seed_rng.randint(1, np.iinfo(np.int64).max, dtype=np.int64)
            )
            obs = dac_env_obj.reset()
            print(
                set_ansi_escape
                + "\nInstance set to: "
                # + dac_env_obj.current_instance.dataset
                + reset_ansi_escape
            )
            dac_policy_obj.reset(dac_env_obj.current_instance)
            done = False
            while not done:
                config = dac_policy_obj.act(obs)
                obs, reward, done, info = dac_env_obj.step(config)
                total_rewards[i] += reward
                #TODO Make it AUC for RL track
        else:
            # TODO: generate some warning that time budget has been exceeded
            warnings.warn(
                "TIME LIMIT EXCEEDED. Setting total reward for instance "
                + str(i)
                + " to 0."
            )
            total_rewards[i] = -np.inf

    return total_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The experiment runner for the DAC4RL track."
    )
    parser.add_argument(
        "-t",
        "--competition-track", 
        choices=['dac4sgd', 'dac4rl'], 
        help="DAC4SGD or DAC4RL", 
        default="dac4rl",
    )
    parser.add_argument(
        "-s",
        "--submission-dir",
        type=str,
        help="Location of program submission",
        default="DAC4AutoML_sample_code_submission",
    )
    parser.add_argument(
        "--ingestion-dir",
        type=str,
        default="DAC4AutoML_ingestion_program",
        help="Location of ingestion program",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="",
        help="",
    )

    print("Working directory:", root_dir)
    args, unknown = parser.parse_known_args()
    # TODO: Should be cmd arguments

    # ingestion_dir = os.path.abspath(args.ingestion_dir)
    # input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    # hidden_dir = os.path.abspath(args.hidden_dir)
    # shared_dir = os.path.abspath(args.shared_dir)
    submission_dir = os.path.abspath(args.submission_dir)
    if verbose:
        # print("Using ingestion_dir: " + ingestion_dir)
        # print("Using input_dir: " + input_dir)
        print("Using output_dir: " + output_dir)
        # print("Using hidden_dir: " + hidden_dir)
        # print("Using shared_dir: " + shared_dir)
        print("Using submission_dir: " + submission_dir)

    path.append(args.submission_dir)
    # print("path:", path)

    from solution import load_solution

    args_ml = {
        "env_name": "sgd-v0",
        "gen_seed": 666,
        "policy_seed": 42,
        "num_instances": 3,
        "time_limit_sec": 10,
    }

    args_rl = {
        "env_name": "dac4carl-v0",
        "gen_seed": 666,
        "policy_seed": 42,
        "num_instances": 3,
        "time_limit_sec": 86_400,
    }

    if args.competition_track == "dac4sgd":
        env_args = args_ml
    elif args.competition_track == "dac4rl":
        env_args = args_rl

    from pathlib import Path    
    policy = load_solution(path=Path(args.submission_dir)) #TODO assert it's a DACPolicy

    if env_args['env_name'] == "sgd-v0":
        import sgd_env
    else:  # "== 'dac4carl-v0'"
        import rlenv

    env = gym.make(env_args["env_name"])

    total_rewards = run_experiment_draft(policy, env, **env_args)

    print("total_rewards:", total_rewards)
    np.savetxt("scores_np.txt", total_rewards, delimiter=",")

    # Write scores.txt
    fout = open(args.output_dir + 'scores.txt', 'a')
    # for rew in total_rewards:
    fout.write("DLSCORES: " + str(np.mean(total_rewards)) + '\n')
    fout.write("RLSCORES: " + str(np.mean(total_rewards)) + '\n')
    fout.close()
