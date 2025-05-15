import random
import numpy as np
import os
import sys
from datetime import datetime

# usage: python run_ppo_qualia.py python run_experiment.py [environment] [results_directory] [method] [omega] [num_trials] [norm_mode]
def run_experiment(environment, results_dir, method, omega, num_trials, norm_mode):
    
    methods = ["method1", "method2", "method3", "control"]
    if method not in methods:
        raise ValueError(f"Method must be one of {methods}")

    if environment not in ["halfcheetah", "cartpole", "pong"]:
        raise ValueError("Environment must be one of ['halfcheetah', 'cartpole', 'pong']")

    if environment == "halfcheetah":
        import ppo_implementations.mujoco_ppo_qualia as ppo
    elif environment == "cartpole":
        import ppo_implementations.cartpole_ppo_qualia as ppo
    elif environment == "pong":
        import ppo_implementations.atari_ppo_qualia as ppo

    if not os.path.exists(f"{results_dir}"):
        os.mkdir(f"{results_dir}")

    omega = float(omega)
    NUM_TRIALS = int(num_trials)
    seeds = [random.randint(0, 2**32 - 1) for _ in range(NUM_TRIALS)]

    args = ppo.Args()

    if method != "control":
        args.qualia_method = method
        args.qualia_omega = omega

    if norm_mode == "nonorm":
        args.norm_adv = False
        print("Not normalizing advantages", flush=True)
    elif norm_mode == "norm":
        args.norm_adv = True
        print("Normalizing advantages", flush=True)
    else:
        raise ValueError("norm_mode must be either 'norm' or 'nonorm'")


    if not os.path.exists(f"{results_dir}"):
        os.mkdir(f"{results_dir}")

    if not os.path.exists(f"{results_dir}/{method}"):
        os.mkdir(f"{results_dir}/{method}")

    if not os.path.exists(f"{results_dir}/{method}/omega_{omega}"):
        os.mkdir(f"{results_dir}/{method}/omega_{omega}")

    if not os.path.exists(f"{results_dir}/{method}/omega_{omega}/timestep_returns"):
        os.mkdir(f"{results_dir}/{method}/omega_{omega}/timestep_returns")

    if not os.path.exists(f"{results_dir}/{method}/omega_{omega}/ep_returns"):
        os.mkdir(f"{results_dir}/{method}/omega_{omega}/ep_returns")

    if not os.path.exists(f"{results_dir}/{method}/omega_{omega}/mean_ratios"):
        os.mkdir(f"{results_dir}/{method}/omega_{omega}/mean_ratios")

    if not os.path.exists(f"{results_dir}/{method}/omega_{omega}/cumulative_ratios"):
        os.mkdir(f"{results_dir}/{method}/omega_{omega}/cumulative_ratios")


    print("Running PPO trials on device: ", ppo.cuda(), flush=True)
    print("Method: ", method, flush=True)
    print("Omega: ", omega, flush=True)
    print(f"Results will be saved to {results_dir}/{method}/omega_{omega}", flush=True)
    print("Start Time: ", datetime.now().strftime("%H:%M"), flush=True)


    for i, seed in enumerate(seeds):
        print(f"Beginning Trial {i+1} of {NUM_TRIALS}", flush=True)
        args.seed = seed

        ret_by_ep, ret_by_ts, sum_ratios, mean_ratios = ppo.learn(args)
        time = datetime.now().strftime("%H%M%S")
        np.savetxt(f"{results_dir}/{method}/omega_{omega}/timestep_returns/{i}_{time}.csv", np.array(ret_by_ts), delimiter=",")
        np.savetxt(f"{results_dir}/{method}/omega_{omega}/cumulative_ratios/{i}_{time}.csv", np.array(sum_ratios), delimiter=",")
        np.savetxt(f"{results_dir}/{method}/omega_{omega}/ep_lengths/{i}_{time}.csv", np.array(ret_by_ep), delimiter=",")
        np.savetxt(f"{results_dir}/{method}/omega_{omega}/mean_ratios/{i}_{time}.csv", np.array(mean_ratios), delimiter=",")

    print("End Time: ", datetime.now().strftime("%H:%M"), flush=True)




if __name__ == "__main__":
    if len(sys.argv) != 7:
        raise ValueError("Usage: python run_ppo_qualia.py [environment] [results_directory] [method] [omega] [num_trials] [norm_mode]")

    _, environment, results_dir, method, omega, num_trials, norm_mode = sys.argv  
    run_experiment(environment, results_dir, method, omega, num_trials, norm_mode)  