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


def run_experiment(
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
    set_ansi_escape = "" # "\033[32;1m"
    reset_ansi_escape = "" # "\033[0m"
    print(
        set_ansi_escape
        + "=" * repeat_equal_sign
        + "Running DAC4AutoML experiment"
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
                + dac_env_obj.current_instance.dataset if args.competition_track == "dac4sgd" else dac_env_obj.current_instance.env_type
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
    verbose = True
    root_dir = getcwd()

    parser = argparse.ArgumentParser(
        description="The experiment runner for the DAC4RL track."
    )
    parser.add_argument(
        "-t",
        "--competition-track", 
        choices=['dac4sgd', 'dac4rl'], 
        help="DAC4SGD or DAC4RL", 
        default="dac4sgd",
    )
    parser.add_argument(
        "-s",
        "--submission-dir",
        type=str,
        help="Location of program submission",
        default="DAC4AutoML_sample_code_submission",
    )
    parser.add_argument(
        "-i",
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

    print("Working directory:", root_dir + "\n")
    args, unknown = parser.parse_known_args()
    # TODO: Should be cmd arguments


    # os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # print("os.environ:", os.environ)
    # import resource
    # print("resource.RLIMIT_NPROC, CPU:", resource.RLIMIT_NPROC, resource.RLIMIT_CPU)

    print("pip installing packages...\n")
    os.system("pip install -r " + args.submission_dir + "/requirements.txt")

    os.environ['HTTP_PROXY'] = "http://web-proxy.rrzn.uni-hannover.de:3128"
    os.environ['HTTPS_PROXY'] = "http://web-proxy.rrzn.uni-hannover.de:3128"
    # print("os.environ:", os.environ)
    # print(os.system("ls -R /app/codalab/dac4automl/"))

    # os.system("bash " + args.ingestion_dir + "/evaluate_submission.sh -i " + 
    #             args.ingestion_dir + " -d " + args.submission_dir + " -f solution.py -o " + 
    #             args.output_dir + " 2>&1")

    ingestion_dir = os.path.abspath(args.ingestion_dir)
    # input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    # hidden_dir = os.path.abspath(args.hidden_dir)
    # shared_dir = os.path.abspath(args.shared_dir)
    submission_dir = os.path.abspath(args.submission_dir)
    if verbose:
        print("\nUsing ingestion_dir: " + ingestion_dir)
        # print("Using input_dir: " + input_dir)
        print("Using output_dir: " + output_dir)
        # print("Using hidden_dir: " + hidden_dir)
        # print("Using shared_dir: " + shared_dir)
        print("Using submission_dir: " + submission_dir + "\n")


    #TODO Remove?
    with open(submission_dir + '/metadata', 'r') as fh:
        track = fh.readline()
        if "rltrack" in track:
            args.competition_track = "dac4rl"


    path.insert(0, args.submission_dir)
    # print("path:", path)

    from solution import load_solution

    num_instances = 10
    args_ml = {
        "env_name": "sgd-v0",
        "gen_seed": 666,
        "policy_seed": 42,
        "num_instances": num_instances,
        "time_limit_sec": 21_600,
    }

    args_rl = {
        "env_name": "dac4carl-v0",
        "gen_seed": 666,
        "policy_seed": 42,
        "num_instances": num_instances,
        "time_limit_sec": 7_200,
    }

    if args.competition_track == "dac4sgd":
        env_args = args_ml
        import sgd_env
    elif args.competition_track == "dac4rl":
        env_args = args_rl
        import rlenv

    from pathlib import Path    
    policy = load_solution(path=Path(args.submission_dir)) #TODO assert it's a DACPolicy

    env = gym.make(env_args["env_name"])

    total_rewards = run_experiment(policy, env, **env_args)

    print("total_rewards:", total_rewards)

    # Write scores.txt
    fout = open(args.output_dir + '/scores.txt', 'a')
    # for rew in total_rewards:
    if args.competition_track == "dac4sgd":
        fout.write("DLSCORES: " + str(np.mean(total_rewards)) + '\n')
        # fout.write("RLSCORES: " + str("NaN") + '\n')
    elif args.competition_track == "dac4rl":
        # fout.write("DLSCORES: " + str("NaN") + '\n')
        fout.write("RLSCORES: " + str(np.mean(total_rewards)) + '\n')

    fout.close()
