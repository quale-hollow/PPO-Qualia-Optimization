# PPO-Qualia-Optimization

This repository contains PPO experiments for the paper **"Qualia Optimization in Reinforcement Learning: Balancing Agent Experience and Performance"**.

This repo includes:
- PPO experiments for `cartpole`, `halfcheetah`, and `pong`
- scripts to reproduce runs and generate paper plots/tables
- archived and extracted PPO result data


## Repository Structure

```
PPO-Qualia-Optimization/
├── experiment_results/          # Extracted PPO result directories
├── experiment_results.zip       # Archived PPO results used for plotting/reproduction
├── ppo_implementations/
│   ├── cartpole_ppo_qualia.py
│   ├── mujoco_ppo_qualia.py
│   └── atari_ppo_qualia.py
├── utils/
│   └── plotting_utils.py
├── ppo_plots/                   # Generated PDFs from plot_paper_results.py
├── run_experiment.py            # Run a single PPO setting
├── reproduce_experiments.py     # Re-run predefined PPO grids for one environment
├── plot_paper_results.py        # Generate paper plots + print LaTeX table rows
├── plot_example.py              # Minimal plotting example
└── requirements.txt
```

## Environment Notes

Different tasks require different dependency combinations. Use separate environments per task.

- CartPole:
  - `python==3.12`
  - `gymnasium==1.0.0`
  - `numpy==2.2.2`
- HalfCheetah:
  - `python==3.12`
  - `gymnasium==0.29.1`
  - `numpy==2.2.2`
- Pong:
  - `python==3.11`
  - `gymnasium==1.0.0`
  - `numpy<2.0` (used: `1.26.0`)
  - `envpool`
- Plotting (any environment):
  - `matplotlib`
  - `scipy`
  - LaTeX install for `text.usetex=True` (for example TeX Live)

Also install `torch` in each environment (CPU or CUDA build as appropriate for your machine).

## Results Data

If needed, extract archived results:

```bash
unzip experiment_results.zip -d .
```

This populates `experiment_results/` with:
- `halfcheetah_norm`
- `halfcheetah_nonorm`
- `cartpole_norm`
- `pong_norm`

## Run Single Experiments

```bash
python run_experiment.py [environment] [results_directory] [method] [omega] [num_trials] [norm_mode]
```

Arguments:
- `environment`: `cartpole`, `halfcheetah`, or `pong`
- `results_directory`: base output directory for this run
- `method`: `method1`, `method2`, `method3`, or `control`
- `omega`: float (`0.0` for `control`)
- `num_trials`: integer
- `norm_mode`: `norm` or `nonorm`, specifies if advantage normalization is used

Example:

```bash
python run_experiment.py cartpole test_dir method1 0.1 10 norm
```

## Reproduce Predefined PPO Grids

`reproduce_experiments.py` runs the hard-coded paper grids for one environment at a time:

```bash
python reproduce_experiments.py [environment]
```

Where `[environment]` is one of:
- `halfcheetah`
- `cartpole`
- `pong`

Output is written to `reproduced_results/...`.

## Plotting

Generate full paper plots from `experiment_results/`:

```bash
python plot_paper_results.py [halfcheetah|cartpole|pong|all]
```

If no argument is provided, default is `all`.

Generated PDFs are saved to `ppo_plots/`.

Minimal plotting example:

```bash
python plot_example.py
```
