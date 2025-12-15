from utils.plotting_utils import plot_curves
import sys

# Usage: python plot_results.py [arg] 
# arg: 'halfcheetah', 'cartpole', 'pong', or 'all'. If no arg is provided, 'all' is used by default.

# This script is used to plot the results of the experiments and to print tex tables for final RQO and performance

arg = sys.argv[1] if len(sys.argv) > 1 else 'all'
if arg not in ['halfcheetah', 'cartpole', 'pong', 'all']:
    print("Invalid argument. Please use 'halfcheetah', 'cartpole', 'pong', or 'all'.")
    sys.exit(1)

# HalfCheetah
if arg == 'halfcheetah' or arg == 'all':
    norms = ["norm", "nonorm"]

    for norm in norms:

        if norm == "nonorm":
            omegas = [
                ["0.01", "0.1", "0.3", "0.5", "1.0"],
                ["0.8", "0.4", "0.2", "0.1", "0.0"],
                ["0.01", "0.1", "0.3", "0.5", "1.0"]
            ]

        if norm == "norm":
            omegas = [
                ["0.5", "1.0", "2.0", "4.0", "8.0"],
                ["0.95", "0.9", "0.5", "0.1", "0.0"],
                ["0.1", "0.5", "1.0", "2.0", "4.0"]
            ]

        
        line_colors = ['C1', 'C2', 'C3', 'C4', 'C5']
        names = ["PPO - Advantage Inflation", "PPO - Asymmetric Clipping", "PPO - Additive Regularizer"]
        methods = ['method1', 'method2', 'method3']

        per_update_qualia_limits = [0.95 - 1, 1.25 - 1]

        plot_curves(methods, names, omegas, norm, line_colors, f'experiment_results/halfcheetah_{norm}', 'halfcheetah', save_dir="plots/", qe_lims=per_update_qualia_limits, NUM_TRIALS=100, NUM_BINS=100)


# CartPole
if arg == 'cartpole' or arg == 'all':
    
    methods = ['method1', 'method2', 'method3']
    omegas = [
        ["0.5", "1.0", "2.0", "5.0", "10.0"],
        ["0.8", "0.4", "0.2", "0.1", "0.0"],
        ["0.5", "1.0", "2.0", "5.0", "10.0"]
    ]

    norm = "norm"
    line_colors = ['C1', 'C2', 'C3', 'C4', 'C5']
    names = ["PPO - Advantage Inflation", "PPO - Asymmetric Clipping", "PPO - Additive Regularizer"]


    qe_lims = [0.9995 - 1, .0020]

    plot_curves(methods, names, omegas, norm, line_colors, 'experiment_results/cartpole_norm', 'cartpole', save_dir="plots/", qe_lims=qe_lims, NUM_TRIALS=100, NUM_BINS=100)


# Pong
if arg == 'pong' or arg == 'all':
   
    methods = ['method1', 'method2', 'method3']

    omegas = [
        ['0.01', '0.05', '0.1', '0.2', '0.3'],
        ['0.0', '0.2', '0.4', '0.6', '0.8'],
        ['0.05', '0.1', '0.3', '0.5', '0.75']
    ]

    norm = "norm"
    line_colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    names = ["PPO - Advantage Inflation", "PPO - Asymmetric Clipping", "PPO - Additive Regularizer"]

    qe_lims = [-0.001, 0.04]


    plot_curves(methods, names, omegas, norm, line_colors, 'experiment_results/pong_norm', 'pong', save_dir="plots/", qe_lims=qe_lims, NUM_TRIALS=100, NUM_BINS=100)

