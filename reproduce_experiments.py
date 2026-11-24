from run_experiment import run_experiment
import sys

task = sys.argv[1]


# Every grid includes standard PPO (control, omega = 0.0), which the plots and tables use as the baseline.
NUM_TRIALS = 100  # trials per setting, as reported in the paper


# RUN HALFCHEETAH EXPERIMENTS
# With and without advantage normalization
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

        run_experiment("halfcheetah", f"reproduced_results/halfcheetah_{norm}", "control", "0.0", NUM_TRIALS, norm)
        for i, method in enumerate(['method1', 'method2', 'method3']):
            for j, omega in enumerate(omegas[i]):
                run_experiment("halfcheetah", f"reproduced_results/halfcheetah_{norm}", method, omega, NUM_TRIALS, norm)
    



# RUN CARTPOLE EXPERIMENTS
elif task == "cartpole":
    omegas = [
        ["0.5", "1.0", "2.0", "5.0", "10.0"],
        ["0.8", "0.4", "0.2", "0.1", "0.0"],
        ["0.5", "1.0", "2.0", "5.0", "10.0"]
    ]

    run_experiment("cartpole", f"reproduced_results/cartpole_norm", "control", "0.0", NUM_TRIALS, "norm")
    for i, method in enumerate(['method1', 'method2', 'method3']):
        for j, omega in enumerate(omegas[i]):
            run_experiment("cartpole", f"reproduced_results/cartpole_norm", method, omega, NUM_TRIALS, "norm")



# RUN PONG EXPERIMENTS
elif task == "pong":
    omegas = [
        ['0.01', '0.05', '0.1', '0.2', '0.3'],
        ['0.0', '0.2', '0.4', '0.6', '0.8'],
        ['0.05', '0.1', '0.3', '0.5', '0.75']
    ]

    run_experiment("pong", f"reproduced_results/pong_norm", "control", "0.0", NUM_TRIALS, "norm")
    for i, method in enumerate(['method1', 'method2', 'method3']):
        for j, omega in enumerate(omegas[i]):
            run_experiment("pong", f"reproduced_results/pong_norm", method, omega, NUM_TRIALS, "norm")

else:
    raise ValueError("Must provide a task: halfcheetah, cartpole, or pong")