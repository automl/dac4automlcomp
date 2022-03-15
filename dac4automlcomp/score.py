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
        "-i",
        "--input-dir",
        type=str,
        default="",
        help="",
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

    root_dir = getcwd()
    print("Working directory:", root_dir)
    args, unknown = parser.parse_known_args()
    # TODO: Should be cmd arguments

    # ingestion_dir = os.path.abspath(args.ingestion_dir)
    # input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    # hidden_dir = os.path.abspath(args.hidden_dir)
    # shared_dir = os.path.abspath(args.shared_dir)
    # submission_dir = os.path.abspath(args.submission_dir)
    if verbose:
        # print("Using ingestion_dir: " + ingestion_dir)
        # print("Using input_dir: " + input_dir)
        print("Using output_dir: " + output_dir)
        # print("Using hidden_dir: " + hidden_dir)
        # print("Using shared_dir: " + shared_dir)
        # print("Using submission_dir: " + submission_dir)

    # path.append(args.submission_dir)
    # print("path:", path)


    # if args['env_name'] == "sgd-v0":
    #     import sgd_env
    # else:  # "== 'dac4carl-v0'"
    #     import rlenv

    # env = gym.make(args["env_name"])

    # total_rewards = run_experiment_draft(policy, env, **args)

    # print("total_rewards:", total_rewards)
    # np.savetxt("scores_1.txt", total_rewards, delimiter=",")

    # Write scores.txt
    fout = open('scores_2.txt', 'a')
    # for rew in total_rewards:
    fout.write("DLSCORES: " + str(np.mean(0)) + ' ')
    fout.write("RLSCORES: " + str(np.mean(0)) + ' ')
    fout.close()
