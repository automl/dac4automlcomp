import argparse
import os
import time
import gym
import warnings

# Parts of the code are inspired by the AutoML3 competition
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
        "-o",
        "--output-dir",
        type=str,
        default="",
        help="",
    )

    root_dir = getcwd()
    print("Working directory:", root_dir)
    args, unknown = parser.parse_known_args()

    output_dir = os.path.abspath(args.output_dir)
    if verbose:
        print("Using output_dir: " + output_dir)

    if not os.path.exists(args.output_dir):
        print("Path not found:", args.output_dir)
        os.makedirs(args.output_dir)


    if os.path.exists(args.output_dir):
        print("Output directory contents:")
        os.system("ls -lR " + args.output_dir)

    if os.path.exists(args.input_dir):
        os.system("cp " + args.input_dir + "/res/scores.txt " + args.output_dir)
    else:
        print("No results from ingestion!")

    with open(args.output_dir + '/scores.txt', 'r') as fh:
        print(fh.readlines())


