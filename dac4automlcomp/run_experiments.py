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
    dac_policy_loader,
    dac_env_obj,
    gen_seed,
    num_instances,
    policy_seed,
    time_limit_sec,
    comp_track,
    **kwargs,
):
    """
    This is the main experiment runner for the DAC4AutoML competition tracks.
    It takes a loader loading the DAC policy to be evaluated and tests the policy on
    a set of num_instances target problem instances and returns the resulting performances as a Numpy array.

    #TODO Improve docstrings
    # defining train/test splits and the distribution for the contexts of the DACEnv.

    Args:
        dac_policy_loader (Callable[[], DACPolicy]): Loads the policy to be evaluated (reloaded for every instance)
        dac_env_obj (DACEnv): The evaluation environment
        gen_seed (int): The seed determining the order of the infinite sequence of evaluation instances
        num_instances (int): The prefix of instances to evaluate the loaded policy on
        policy_seed (int): The seed determining the random seed for the policy's rng
        time_limit_sec (int): The wall-clock time limit for the entire evaluation

    Returns:
        total_rewards: A Numpy array of performances with shape (num_instances,)
        duration (float): Duration of the evaluation in sec

    """

    screen_output_width = 80 # os.get_terminal_size().columns
    repeat_equal_sign = (screen_output_width - 24) // 2
    set_ansi_escape = "" # "\033[32;1m"
    reset_ansi_escape = "" # "\033[0m"
    print(
        set_ansi_escape
        + "=" * repeat_equal_sign
        + "Running Evaluation"
        + "=" * repeat_equal_sign
        + reset_ansi_escape
    )

    print("Current working directory:", os.getcwd())

    total_rewards = np.zeros((num_instances,))
    if comp_track == "dac4rl":
        per_env_stats = {}
        curr_env_type = None
        all_envs = ["CARLPendulumEnv", "CARLAcrobotEnv", "CARLMountainCarContinuousEnv", "CARLLunarLanderEnv", "CARLCartPoleEnv"]
        for env in all_envs:
            per_env_stats[env] = {}
            per_env_stats[env]["reward"] = 0.0
            per_env_stats[env]["num_instances"] = 0

    dac_env_obj.seed(gen_seed)
    policy_seed_rng = np.random.RandomState(policy_seed)

    start_time = time.time()  # TODO: Replace with a superior way of timing?
    duration = 0
    for i in range(num_instances):
        print("> Start evaluation on instance {}:".format(i))
        if duration < time_limit_sec:
            dac_policy_obj = dac_policy_loader()
            print("- Loaded policy object of type:", type(dac_policy_obj))
            dac_policy_obj.seed(
                policy_seed_rng.randint(1, np.iinfo(np.int64).max, dtype=np.int64)
            )
            obs = dac_env_obj.reset()
            if comp_track == "dac4rl":
                curr_env_type = dac_env_obj.current_instance.env_type

                # If an env has hit its limit of num_instances, keep resetting until another env instance is sampled
                while per_env_stats[curr_env_type]["num_instances"] == num_instances / 5:
                    obs = dac_env_obj.reset()
                    curr_env_type = dac_env_obj.current_instance.env_type

                per_env_stats[curr_env_type]["num_instances"] += 1

            if comp_track == "dac4rl":
                print(
                    set_ansi_escape
                    + "\nInstance set to: "
                    + (dac_env_obj.current_instance.dataset if comp_track == "dac4sgd" else dac_env_obj.current_instance.env_type)
                    + reset_ansi_escape
                )

            dac_policy_obj.reset(dac_env_obj.current_instance)
            done = False
            while not done:
                config = dac_policy_obj.act(obs)
                obs, reward, done, info = dac_env_obj.step(config)
                total_rewards[i] += reward

        duration = time.time() - start_time
        if duration > time_limit_sec:
            limit_exceed_penalty = -1e6 if comp_track == "dac4rl" else -np.log(10)

            warnings.warn(
                "TIME LIMIT EXCEEDED. Setting total reward for instance "
                + str(i)
                + " to {}.".format(limit_exceed_penalty)
            )
            total_rewards[i] = limit_exceed_penalty
        
        if comp_track == "dac4rl":
            per_env_stats[curr_env_type]["reward"] += total_rewards[i]

        print("\n- Total reward for instance: {}".format(total_rewards[i]))
        print("- Time elapsed: {} / {} sec".format(duration, time_limit_sec))

    return total_rewards, duration, per_env_stats if comp_track == "dac4rl" else None


if __name__ == "__main__":
    verbose = True
    root_dir = getcwd()

    parser = argparse.ArgumentParser(
        description="The experiment runner for the DAC4AutoML competition."
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
        "-i",
        "--ingestion-dir",
        type=str,
        default="dac4automlcomp/",
        help="Location of ingestion program",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="tmp_output",
        help="",
    )
    parser.add_argument(
        "-g",
        "--gen-seed",
        type=int,
        default=42,
        help="The generator seed (determining the instance order)",
    )
    parser.add_argument(
        "-n",
        "--n_instances",
        type=int,
        default=5,
        help="The number of instances to evaluate on, when set overrides the track defaults",
    )

    print("Working directory:", root_dir + "\n")
    args, unknown = parser.parse_known_args()
    print("Track:", args.competition_track + "\n")

    # os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # print("os.environ:", os.environ)
    # import resource
    # print("resource.RLIMIT_NPROC, CPU:", resource.RLIMIT_NPROC, resource.RLIMIT_CPU)

    print("pip installing packages...\n")
    os.system("pip install -r " + args.submission_dir + "/requirements.txt")
    # from dotenv import load_dotenv
    # load_dotenv("/app/codalab/dac4automl/.env")

    # os.environ['HTTP_PROXY'] = "http://web-proxy.rrzn.uni-hannover.de:3128"
    # os.environ['HTTPS_PROXY'] = "http://web-proxy.rrzn.uni-hannover.de:3128"
    print("os.environ:", os.environ)
    print(os.system("cat /proc/cpuinfo"))
    print(os.system("cat /proc/meminfo"))
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


    path.insert(0, args.submission_dir)
    # print("path:", path)

    from solution import load_solution

    args_ml = {
        "env_name": "sgd-v0",
        "gen_seed": args.gen_seed,
        "policy_seed": args.gen_seed,
        "num_instances": args.n_instances,
        "time_limit_sec": 18_000,
    }

    args_rl = {
        "env_name": "dac4carl-v0",
        "gen_seed": args.gen_seed,
        "policy_seed": args.gen_seed,
        "num_instances": args.n_instances,
        "time_limit_sec": 10_800,
    }

    if args.competition_track == "dac4sgd":
        env_args = args_ml
        import sgd_env
    elif args.competition_track == "dac4rl":
        env_args = args_rl
        import rlenv

    from pathlib import Path
    policy_loader = lambda : load_solution(path=Path(args.submission_dir))  # TODO assert it's a DACPolicy

    env = gym.make(env_args["env_name"])

    print("num_instances:", args.n_instances)

    total_rewards, duration, per_env_stats = run_experiment(
        policy_loader, env, comp_track=args.competition_track, **env_args)

    print("total_rewards:", total_rewards)
    print("duration:", duration)

    # Write scores.txt
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    fout = open(Path(args.output_dir) / Path('scores.txt'), 'w')
    # for rew in total_rewards:
    if args.competition_track == "dac4sgd":
        for i, r in enumerate(total_rewards):
            fout.write("cid_{}: {} \n".format(i, -r))
        fout.write("duration: {} \n".format(duration))
    elif args.competition_track == "dac4rl":
        print("per_env_stats:", per_env_stats)
        for i, r in enumerate(per_env_stats):
            all_envs_full = ["CARLPendulumEnv", "CARLAcrobotEnv", "CARLMountainCarContinuousEnv", "CARLLunarLanderEnv", "CARLCartPoleEnv"]
            all_envs = ["Pendulum", "Acrobot", "MountainCarContinuous", "LunarLander", "CartPole"]

            if per_env_stats[all_envs_full[i]]["num_instances"] != 0:
                fout.write("{}: {} \n".format(all_envs[i], per_env_stats[all_envs_full[i]]["reward"]/per_env_stats[all_envs_full[i]]["num_instances"]))
            else:
                warnings.warn("No instances of env " + all_envs_full[i] + " sampled.")

        fout.write("Duration: {} \n".format(duration))

    fout.close()
