from run_experiment import run_experiment
import sys

task = sys.argv[1]


# RUN HALFCHEETAH EXPERIMENTS
# 300 trials, with and without advantage normalization
if task == "halfcheetah":
    for norm in ["norm", "nonorm"]:

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

        for i, method in enumerate(['method1', 'method2', 'method3']):
            for j, omega in enumerate(omegas[i]):
                run_experiment("halfcheetah", f"reproduced_results/halfcheetah_{norm}", method, omega, 300, norm)
    



# RUN CARTPOLE EXPERIMENTS
# 300 trials
elif task == "cartpole":
    omegas = [
        ["0.5", "1.0", "2.0", "5.0", "10.0"],
        ["0.8", "0.4", "0.2", "0.1", "0.0"],
        ["0.5", "1.0", "2.0", "5.0", "10.0"]
    ]

    for i, method in enumerate(['method1', 'method2', 'method3']):
        for j, omega in enumerate(omegas[i]):
            run_experiment("cartpole", f"reproduced_results/cartpole_norm", method, omega, 300, "norm")



# Pong
# 100 trials
elif task == "pong":
    omegas = [
        ['0.01', '0.05', '0.1', '0.2', '0.3'],
        ['0.0', '0.2', '0.4', '0.6', '0.8'],
        ['0.05', '0.1', '0.3', '0.5', '0.75']
    ]

    for i, method in enumerate(['method1', 'method2', 'method3']):
        for j, omega in enumerate(omegas[i]):
            run_experiment("pong", f"reproduced_results/pong_norm", method, omega, 100, "norm")

else:
    raise ValueError("Must provide a task: halfcheetah, cartpole, or pong")