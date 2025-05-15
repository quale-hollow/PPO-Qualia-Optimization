# PPO-Qualia-Optimization
This repository contains the code and results for the paper **"Qualia Optimization in Reinforcement Learning: Balancing Agent Experience and Performance"**. It provides implementations of the Proximal Policy Optimization (PPO) algorithm for qualia optimization in three reinforcement learning (RL) tasks: **CartPole**, **HalfCheetah**, and **Pong**.

This README includes detailed instructions for:
- Setting up the necessary environments.
- Extracting existing results.
- Running experiments.
- Plotting results to visualize performance and qualia.


## Repository Structure

```
PPO-Qualia-Optimization/
├── experiment_results.zip  # Zip file containing experiment results
├── README.md              # Project documentation
├── reproduce_experiments.py # Script to replicate all experiments
├── run_experiment.py      # Script to run individual experiments
├── plot_paper_results.py  # Script to generate plots for paper results
├── plot_example.py        # Example script for plotting specific results
├── requirements.txt       # Dependencies for the project
├── .gitignore             # Git ignore file
├── .gitattributes         # Git (large file storage) LFS attributes 
├── utils/                 
│   └── plotting_utils.py  # Helper functions for plotting results
├── ppo_implementations/   # PPO qualia optimization implementations for different environments
    ├── cartpole_ppo_qualia.py
    ├── atari_ppo_qualia.py
    └── mujoco_ppo_qualia.py
```

## Requirements

To avoid conflicts between dependencies for the experiments in each task, we recommend creating separate environments for each task. Below are the requirements for the environments:

### CartPole Environment
- `gymnasium==1.0.0`
- `numpy==2.2.2`
- `python==3.12`

### HalfCheetah Environment
- `gymnasium==0.29.1`
- `numpy==2.2.2`
- `python==3.12`

### Pong Environment
- `gymnasium==1.0.0`
- `numpy<2.0` (we use version 1.26.0)
- `python==3.11`

Additionally, at least one of these environments should include the following packages for generating results:
- `texlive`
- `matplotlib`


## Extracting Existing Results

To extract the results of the experiments used in the paper, unzip the file `experiment_results.zip`. This will save four directories `halfcheetah_norm`, `halfcheetah_nonorm`, `cartpole_norm`, and `pong_norm` to the `experiment_results/` directory.

After extracting the results, you can run `python plot_paper_results.py` to generate and save plots as well as print LaTeX tables summarizing the results. This script processes the data in the extracted directories and outputs the visualizations and tables in the `plots/` directory.


# Running Experiments

To run experiments for a specific task, activate the corresponding environment and run the `run_experiment.py` script.

To use the `run_experiment.py` script, follow these steps:

Use the following command to execute the script:
    `python run_experiment.py [environment] [results_directory] [method] [omega] [num_trials] [norm_mode]`

    Replace the placeholders with the appropriate values:

    - `[environment]`:  
        The environment to run the experiment on (`cartpole`, `halfcheetah`, or `pong`).

    - `[results_directory]`:  
        The directory where results will be saved.

    - `[method]`:  
        The qualia optimization method to use (`method1`, `method2`, `method3`, or `control`).

    - `[omega]`:  
        The omega value for qualia optimization (a float). Use `0.0` when using the `control` method.

    - `[num_trials]`:  
        The number of trials to run (an integer).

    - `[norm_mode]`:  
        Whether to normalize advantages (`norm` or `nonorm`).


 **Example Command**:
    ```
    python run_experiment.py cartpole test_dir method1 0.1 10 norm
    ```
    This command runs 10 trials of the CartPole environment using `method1` with an omega value of 0.1 and normalized advantages, saving results to the `test_dir/` directory.


## Replicating Paper Experiments 
To replicate the experiments presented in the paper, you can use the `reproduce_experiments.py` script. This script is designed to sequentially run all the experiments for the three environments: **HalfCheetah**, **CartPole**, and **Pong**. It executes each trial in order for every environment and method combination.

To use the `reproduce_experiments.py` script, activate any environment with the required dependencies installed and run the following command:

```
python reproduce_experiments.py [environment]
```

This will the experiments described in the paper for the given environment, saving the results in the appropriate directories. The script is configured to run trials sequentially. While the script provides a straightforward way to replicate the experiments, running all trials sequentially can be time-consuming. To speed up the process, consider parallelizing the trials across multiple machines or using a distributed computing framework. This is especially important for environments like **HalfCheetah** and **Pong**, which require more computational resources.

### Output

The results of the experiments will be saved in the `experiment_results/` directory, organized by environment and method. Once the script completes, you can use the provided plotting scripts to visualize the results.


## Plotting Results
This script provides utilities for plotting results and generating visualizations for experiments.

- `plot_paper_results.py`: 
    This script plots all the results presented in the paper (saved to experiment_results/) and saves the generated plots to the `plots/` directory. 
    It does not require any arguments to execute.

- `plot_example.py`: 
    This script demonstrates how to plot results for trials of method 1 with omega=0.1 in the CartPole environment. 
    This script can be executed after running the above experiment example: `python run_experiment.py cartpole test_dir method1 0.1 10 norm'

**Note**: 
The provided plotting utility function requires 'control' trials to be saved in the specified directory. 


