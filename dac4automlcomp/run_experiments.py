import os
import gym
import numpy as np

def run_experiment(dac_opt_obj, max_steps=10_000, eval_every=1000, num_repetitions=5, num_eval_episodes=5):
    '''

    '''

    screen_output_width = os.get_terminal_size().columns
    repeat_equal_sign = (screen_output_width - 24) // 2
    set_ansi_escape = "\033[32;1m"
    reset_ansi_escape = "\033[0m"
    print(set_ansi_escape
        + "=" * repeat_equal_sign
        + "Running DAC4RL experiment"
        + "=" * repeat_equal_sign
        + reset_ansi_escape
    )
    print("\n\nLoaded object of type:", type(dac_opt_obj), "\n\n")
    print("Current working directory:", os.getcwd())

    # Get set of envs to run on:
    # Some people might want VecEnv?? #TODO
    envs = [gym.make("CartPole-v1")]
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
                #TODO should we evaluate for x episodes or x steps? What to do in case episode finishes in-between?
                action = env.action_space.sample() #TODO dac_opt_obj.act(state)
                next_state, reward, done, info = env.step(action)
                tot_reward += reward

                state = next_state

                # if iter % eval_every == eval_every - 1:
                #     print("Evaluating policy for")

            print("Total reward for this repetition:", tot_reward)
            tot_reward_list.append(tot_reward)
        
        print("Average total reward for env:", str(env), "is:", np.mean(tot_reward_list))



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
    run_experiment(str("a"))
