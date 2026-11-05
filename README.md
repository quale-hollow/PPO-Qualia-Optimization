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

Each environment folder uses this structure:

```text
experiment_results/<env>_<norm_mode>/
  <method>/                      # control, method1 (advantage inflation), method2 (asymmetric clipping), method3 (additive regularization)
    omega_<omega>/
      timestep_returns/
      ep_returns/
      mean_ratios/
      cumulative_ratios/
```

Each csv corresponds to a single trial.

CSV contents:
- `timestep_returns/*.csv`: 2 columns, `[global_step, episodic_return]`
- `ep_returns/*.csv`: 1 column, episodic return sequence for that trial
- `mean_ratios/*.csv`: 2 columns, `[global_step, average per-update likelihood ratio: mean(exp(new_logprob - old_logprob))]`
- `cumulative_ratios/*.csv`: 2 columns, `[global_step, running sum of exp(new_logprob - old_logprob)]` (unused)

Notes:
- `new_logprob` is evaluated after PPO policy updates for the current rollout batch; `old_logprob` is the stored rollout policy log-probability from data collection time.
- The plot labeled per-update qualia valence (`Q_k`) is computed from `mean_ratios` as:
  - `Q_k = mean_ratio_k - 1`

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

Plotting/statistical details (`utils/plotting_utils.py`):
- 95% confidence intervals use Student's t distribution:
  - `CI_95 half-width = t.ppf(0.975, df=n-1) * (sample_std(ddof=1) / sqrt(n))`
- Each condition uses the first `NUM_TRIALS` file names (sorted) present in all three metric folders, so curves and final values come from the same trials.
- Time-step curves are built by:
  - dividing training into `NUM_BINS` equal-width bins over `total_steps`,
  - averaging each trial's rows within each bin to get one value per trial,
  - using the bin center as the x-value,
  - plotting the mean across trials with t-based 95% CI (`n` = trials with data in that bin).
- For `mean_ratios`, plotting converts to per-update qualia valence by subtracting `1.0` before computing bin means/CIs.

Minimal plotting example:

```bash
python plot_example.py
```
