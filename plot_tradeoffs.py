from utils.plotting_utils import plot_tradeoff
from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
import os

# Usage: python plot_tradeoffs.py
# Plots final return (last episode) vs RQO for each method in HalfCheetah-v4, CartPole-v1, and Pong-v5,
# one panel per environment, saved to ppo_plots/all_envs_tradeoff.pdf (the paper's trade-off figure).

names = ["Advantage Inflation", "Asymmetric Clipping", "Additive Regularizer"]
methods = ["method1", "method2", "method3"]
save_dir = "ppo_plots/"

# Omegas are listed in order of increasing reinforcement bias (asymmetric clipping decreases omega)
envs = [
    {
        "key": "halfcheetah_norm",
        "title": "HalfCheetah-v4",
        "path": "experiment_results/halfcheetah_norm",
        "omegas": [
            ["0.5", "1.0", "2.0", "4.0", "8.0"],
            ["0.95", "0.9", "0.5", "0.1", "0.0"],
            ["0.1", "0.5", "1.0", "2.0", "4.0"],
        ],
        "yscale": "log",
    },
    {
        "key": "cartpole_norm",
        "title": "CartPole-v1",
        "path": "experiment_results/cartpole_norm",
        "omegas": [
            ["0.5", "1.0", "2.0", "5.0", "10.0"],
            ["0.8", "0.4", "0.2", "0.1", "0.0"],
            ["0.5", "1.0", "2.0", "5.0", "10.0"],
        ],
        "yscale": "linear",
    },
    {
        "key": "pong_norm",
        "title": "Pong-v5",
        "path": "experiment_results/pong_norm",
        "omegas": [
            ["0.01", "0.05", "0.1", "0.2", "0.3"],
            ["0.8", "0.6", "0.4", "0.2", "0.0"],
            ["0.05", "0.1", "0.3", "0.5", "0.75"],
        ],
        "yscale": "symlog",
    },
]

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 8,
    "font.size": 8,
    "legend.fontsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
})
plt.style.use("seaborn-v0_8-paper")
os.makedirs(save_dir, exist_ok=True)

width_in = 397.48499 / 72.27


def add_legend(fig, ax, ncol):
    handles, labels = ax.get_legend_handles_labels()
    # One "Standard PPO" entry showing both its marker and its dashed return line
    i = labels.index("Standard PPO")
    dashed = plt.Line2D([], [], color="C0", linestyle="--", linewidth=0.6, alpha=0.6)
    handles[i] = (handles[i], dashed)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=ncol, frameon=False, handletextpad=0.4, columnspacing=1.0,
               handlelength=1.6, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)})


# Combined figure, one row with one panel per environment
fig, axes = plt.subplots(1, len(envs), figsize=(width_in, width_in * 0.28), constrained_layout=True)
for ax, env in zip(axes.flat, envs):
    plot_tradeoff(ax, env["path"], methods, names, env["omegas"], env["title"],
                  yscale=env["yscale"], NUM_TRIALS=100, last_n_episodes=1)
for ax in axes.flat[1:]:
    ax.set_ylabel("")
# A one-line legend narrower than the panel row keeps the saved PDF at \textwidth,
# so the panels fill the figure when included at \textwidth.
add_legend(fig, axes.flat[0], ncol=4)
fig.savefig(f"{save_dir}/all_envs_tradeoff.pdf", format="pdf", bbox_inches="tight")
plt.close(fig)
