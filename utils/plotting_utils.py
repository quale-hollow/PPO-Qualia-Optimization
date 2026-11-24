import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import t


# Fixed panel geometry shared by every per-method figure (instead of constrained_layout and
# tight bounding boxes), so all figures have the same size and panel positions and line up
# when stacked in the paper. Margins are fractions of the figure; YLABEL_X places every
# y-axis label at the same offset from its axis (in axes coordinates).
FIG_HEIGHT_RATIO = 0.28
PANEL_MARGINS = dict(left=0.095, right=0.99, bottom=0.33, top=0.88, wspace=0.75)
YLABEL_X = -0.46
LEGEND_PAD = 0.03


# ----------------------------
# 95% CI (t-based) half-width
# ----------------------------
def ci95_halfwidth(vals):
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n <= 1:
        return 0.0
    sem = np.std(vals, ddof=1) / np.sqrt(n)
    return t.ppf(0.975, df=n - 1) * sem


def pm(mean, ci, decimals=2):
    """Table entry 'mean ± ci' fully in math mode, so negatives get a true minus sign."""
    return f"${mean:.{decimals}f} \\pm {ci:.{decimals}f}$"


def bin_trials(trials, total_steps, num_bins=100, value_offset=0.0):
    """
    Trial-level learning curve over equal-width time step bins.

    Each trial contributes one value per bin (the mean of its rows in that bin),
    and the bin's mean and 95% CI are computed across those per-trial values.

    Args:
        trials: list of [T, 2] arrays with columns [global_step, value], one per trial.
        value_offset: subtracted from each value (1.0 converts ratios to Q_k).

    Returns:
        bin_centers, mean_bins, ci95_bins
    """
    bin_width = total_steps / num_bins
    per_trial = np.full((len(trials), num_bins), np.nan)

    for i, rows in enumerate(trials):
        rows = np.asarray(rows, dtype=np.float64)
        rows = rows[np.isfinite(rows).all(axis=1)]
        idx = np.clip((rows[:, 0] / bin_width).astype(int), 0, num_bins - 1)
        counts = np.bincount(idx, minlength=num_bins)
        sums = np.bincount(idx, weights=rows[:, 1] - value_offset, minlength=num_bins)
        nonempty = counts > 0
        per_trial[i, nonempty] = sums[nonempty] / counts[nonempty]

    trials_per_bin = np.isfinite(per_trial).sum(axis=0)

    bin_centers = (np.arange(num_bins) + 0.5) * bin_width
    mean_bins = np.full(num_bins, np.nan)
    ci95_bins = np.full(num_bins, np.nan)
    for b in range(num_bins):
        if trials_per_bin[b] > 0:
            mean_bins[b] = np.nanmean(per_trial[:, b])
            ci95_bins[b] = ci95_halfwidth(per_trial[:, b])

    return bin_centers, mean_bins, ci95_bins


def get_trial_files(condition_dir, max_trials=None):
    """
    File names present in both metric folders, sorted so every metric uses
    the same trials regardless of filesystem listing order.
    """
    metric_dirs = ["timestep_returns", "mean_ratios"]
    shared = set.intersection(*(set(os.listdir(f"{condition_dir}/{m}")) for m in metric_dirs))
    files = sorted(shared)
    if max_trials is not None:
        files = files[:max_trials]
    return files


def get_returns_data(filepath, files, total_steps, num_bins=100):
    """
    Returns:
        bin_centers, mean_returns_bins, ci95_returns_bins, count
    """
    trials = [
        np.genfromtxt(os.path.join(filepath, fname), delimiter=",").reshape(-1, 2)
        for fname in files
    ]
    bin_centers, mean_bins, ci95_bins = bin_trials(trials, total_steps, num_bins=num_bins)
    return bin_centers, mean_bins, ci95_bins, len(trials)


def get_binned_reinforcement_data(filepath, files, total_steps, num_bins=100):
    """
    Returns:
        bin_centers, mean_bins, ci95_bins, count
    """
    trials = [
        np.genfromtxt(os.path.join(filepath, fname), delimiter=",").reshape(-1, 2)
        for fname in files
    ]
    bin_centers, mean_bins, ci95_bins = bin_trials(trials, total_steps, num_bins=num_bins, value_offset=1.0)
    return bin_centers, mean_bins, ci95_bins, len(trials)


def get_episode_returns(filepath, fname):
    """
    A trial's sequence of episodic returns, in episode order, read from its
    timestep_returns csv (columns: [global_step, episodic_return]).
    """
    return np.genfromtxt(os.path.join(filepath, fname), delimiter=",").reshape(-1, 2)[:, 1]


def get_final_return(filepath, files):
    """
    Final return is each trial's last episodic return (from timestep_returns).

    Returns:
        mean_final_return, ci95_final_return
    """
    final_returns = [get_episode_returns(filepath, fname)[-1] for fname in files]

    final_returns = np.asarray(final_returns, dtype=np.float64)
    mean_final_return = float(np.mean(final_returns))
    ci95_final_return = float(ci95_halfwidth(final_returns))

    return mean_final_return, ci95_final_return


def get_final_qualia_return(filepath, files):
    """
    Returns:
        mean_qualia_return, ci95_qualia_return
    """
    qualia_returns = []

    for fname in files:
        data = np.genfromtxt(os.path.join(filepath, fname), delimiter=",")
        qualia_experiences = data[:, 1] - 1.0
        qualia_returns.append(np.sum(qualia_experiences))

    qualia_returns = np.asarray(qualia_returns, dtype=np.float64)
    mean_qualia_return = float(np.mean(qualia_returns))
    ci95_qualia_return = float(ci95_halfwidth(qualia_returns))

    return mean_qualia_return, ci95_qualia_return


# graphing functions from: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
def set_size(width, fraction=1, subplots=(1, 1)):
    fig_width_pt = width * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**0.5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in)


def axes_labels(x, pos):
    """Compact tick labels, e.g. -500, 0.25, 1.5K, 40M."""
    for scale, suffix in ((1e6, "M"), (1e3, "K")):
        if abs(x) >= scale:
            return f"{x / scale:g}{suffix}"
    return f"{x:g}"


def plot_curves(
    methods,
    names,
    omegas,
    norm,
    line_colors,
    path,
    env,
    total_steps,
    save_dir="plots",
    qe_lims=None,
    NUM_TRIALS=None,
    NUM_BINS=100,
    rqo_decimals=2,
):
    width = 397.48499
    inches_per_pt = 1 / 72.27

    fig_width = width * inches_per_pt
    fig_height = fig_width * FIG_HEIGHT_RATIO
    layout = (1, 4)
    fig_dim = (fig_width, fig_height)

    print("Plotting results for ", env, " with ", norm)

    tex_fonts = {
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 8,
        "font.size": 8,
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
    }

    plt.rcParams.update(tex_fonts)
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams["xtick.major.pad"] = 0.5
    plt.rcParams["ytick.major.pad"] = 0.5

    # Ensure save dir exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # ---------- BASELINE (CONTROL) ----------
    bl_files = get_trial_files(f"{path}/control/omega_0.0", max_trials=NUM_TRIALS)

    bl_ret_x, bl_ret_y, bl_ret_ci, bl_ret_count = get_returns_data(
        f"{path}/control/omega_0.0/timestep_returns/",
        bl_files,
        total_steps,
        num_bins=NUM_BINS,
    )
    print("Count for baseline: ", bl_ret_count)

    bl_rf_x, bl_rf_y, bl_rf_ci, _ = get_binned_reinforcement_data(
        f"{path}/control/omega_0.0/mean_ratios/",
        bl_files,
        total_steps,
        num_bins=NUM_BINS,
    )

    bl_final_return, bl_final_return_ci = get_final_return(
        f"{path}/control/omega_0.0/timestep_returns/",
        bl_files,
    )
    bl_qr, bl_qr_ci = get_final_qualia_return(
        f"{path}/control/omega_0.0/mean_ratios/",
        bl_files,
    )

    table1 = (
        f"Standard PPO & N/A & "
        f"{pm(bl_final_return, bl_final_return_ci)} & "
        f"{pm(bl_qr, bl_qr_ci, rqo_decimals)} \\\\ \\midrule\n"
    )

    # ---------- METHODS ----------
    for i, method in enumerate(methods):
        baseline_omega = "1.0" if method == "method2" else "0.0"
        control_x = 1 if method == "method2" else 0

        fig, ax = plt.subplots(layout[0], layout[1], figsize=fig_dim)
        fig.subplots_adjust(**PANEL_MARGINS)
        fig.suptitle(names[i], fontsize=9, y=0.995)
        label_pad = 0

        ax[0].set_xlabel("Time Step", labelpad=label_pad)
        ax[0].set_ylabel(r"$J(\theta)$", labelpad=label_pad)
        ax[0].xaxis.set_major_formatter(plt.FuncFormatter(axes_labels))
        ax[0].yaxis.set_major_formatter(plt.FuncFormatter(axes_labels))

        ax[1].set_xlabel("Time Step", labelpad=label_pad)
        ax[1].set_ylabel(r"$Q_k$", labelpad=label_pad)
        ax[1].xaxis.set_major_formatter(plt.FuncFormatter(axes_labels))
        if qe_lims is not None:
            ax[1].set_ylim(*qe_lims)

        ax[2].set_xlabel(r"$\omega$", labelpad=label_pad)
        ax[2].set_ylabel(r"Final $J(\theta)$", labelpad=label_pad)
        ax[2].yaxis.set_major_formatter(plt.FuncFormatter(axes_labels))

        ax[3].set_xlabel(r"$\omega$", labelpad=label_pad)
        ax[3].set_ylabel("RQO", labelpad=label_pad)
        ax[3].yaxis.set_major_formatter(plt.FuncFormatter(axes_labels))

        # ---- plot baseline (control) with 95% CI ----
        ax[0].plot(bl_ret_x, bl_ret_y, label=fr"$\omega={baseline_omega}$ (standard PPO)", color="C0")
        ax[0].fill_between(bl_ret_x, bl_ret_y - bl_ret_ci, bl_ret_y + bl_ret_ci, alpha=0.3, color="C0")

        ax[1].plot(bl_rf_x, bl_rf_y, color="C0")
        ax[1].fill_between(bl_rf_x, bl_rf_y - bl_rf_ci, bl_rf_y + bl_rf_ci, alpha=0.3, color="C0")

        ax[2].errorbar(control_x, bl_final_return, yerr=bl_final_return_ci, fmt="o", color="C0")
        ax[3].errorbar(control_x, bl_qr, yerr=bl_qr_ci, fmt="o", color="C0")

        # ---- plot each omega curve ----
        for j, omega in enumerate(omegas[i]):
            files = get_trial_files(f"{path}/{method}/omega_{omega}", max_trials=NUM_TRIALS)

            x, y, ci, count = get_returns_data(
                f"{path}/{method}/omega_{omega}/timestep_returns/",
                files,
                total_steps,
                num_bins=NUM_BINS,
            )
            print("Count for ", method, omega, norm, ": ", count)

            ax[0].plot(x, y, label=fr"$\omega={omega}$", color=line_colors[j])
            ax[0].fill_between(x, y - ci, y + ci, alpha=0.3, color=line_colors[j])

            x, y, ci, _ = get_binned_reinforcement_data(
                f"{path}/{method}/omega_{omega}/mean_ratios/",
                files,
                total_steps,
                num_bins=NUM_BINS,
            )
            ax[1].plot(x, y, color=line_colors[j])
            ax[1].fill_between(x, y - ci, y + ci, alpha=0.3, color=line_colors[j])

            # Final metrics + table
            ret, ret_ci = get_final_return(f"{path}/{method}/omega_{omega}/timestep_returns/", files)
            QR, QR_ci = get_final_qualia_return(f"{path}/{method}/omega_{omega}/mean_ratios/", files)

            table1 += (f"{names[i]} & ") if j == 0 else (f"\t & ")
            table1 += (
                f"{omega} & {pm(ret, ret_ci)} & "
                f"{pm(QR, QR_ci, rqo_decimals)} \\\\"
            )
            if j == len(omegas[i]) - 1:
                table1 += f"{chr(92)}midrule"
            table1 += "\n"

            ax[2].errorbar(float(omega), ret, yerr=ret_ci, fmt="o", color=line_colors[j])
            ax[3].errorbar(float(omega), QR, yerr=QR_ci, fmt="o", color=line_colors[j])

        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            mode="expand",
            # span the panel row, so the legend lines up with the panels; LEGEND_PAD
            # leaves a gap below it so stacked figures don't touch
            bbox_to_anchor=(PANEL_MARGINS["left"], LEGEND_PAD,
                            PANEL_MARGINS["right"] - PANEL_MARGINS["left"], 1),
            ncol=6,
            columnspacing=1,
            handletextpad=0.2,
            borderpad=0,
            borderaxespad=0,
            frameon=False,
        )
        for a in ax:
            a.yaxis.set_label_coords(YLABEL_X, 0.5)

        plt.savefig(f"{save_dir}/{env}_{method}_{norm}.pdf", format="pdf")
        plt.clf()

    print("\n\n")
    print(table1)


def get_condition_finals(condition_dir, NUM_TRIALS=None, last_n_episodes=1):
    """
    Final return is each trial's mean over its last `last_n_episodes` training
    episodes, then the mean and 95% CI across trials.

    Returns:
        mean_final_return, ci95_final_return, mean_rqo, ci95_rqo
    """
    files = get_trial_files(condition_dir, max_trials=NUM_TRIALS)
    final_returns = np.asarray([
        np.mean(get_episode_returns(f"{condition_dir}/timestep_returns/", fname)[-last_n_episodes:])
        for fname in files
    ])
    QR, QR_ci = get_final_qualia_return(f"{condition_dir}/mean_ratios/", files)
    return float(np.mean(final_returns)), float(ci95_halfwidth(final_returns)), QR, QR_ci


def plot_tradeoff(ax, path, methods, names, omegas, title, yscale="linear", NUM_TRIALS=None,
                  last_n_episodes=1):
    """
    Final return vs RQO for each method, removing omega from the comparison.
    Each method's line starts at standard PPO and follows its omegas in the order
    given (increasing reinforcement bias). Curves that climb higher while staying
    near standard PPO's return (dashed line) give the better trade-off.
    """
    method_styles = [("C1", "s"), ("C2", "^"), ("C3", "D")]

    bl = get_condition_finals(f"{path}/control/omega_0.0", NUM_TRIALS, last_n_episodes)

    for i, method in enumerate(methods):
        color, marker = method_styles[i]
        points = [bl] + [
            get_condition_finals(f"{path}/{method}/omega_{omega}", NUM_TRIALS, last_n_episodes)
            for omega in omegas[i]
        ]
        ret, ret_ci, QR, QR_ci = map(np.array, zip(*points))
        # The baseline point's error bars are drawn once, in blue, below
        ret_ci[0], QR_ci[0] = 0.0, 0.0
        ax.errorbar(ret, QR, xerr=ret_ci, yerr=QR_ci, fmt=f"-{marker}", color=color,
                    linewidth=0.8, markersize=3, elinewidth=0.6, capsize=1, label=names[i])

    ax.errorbar(bl[0], bl[2], xerr=bl[1], yerr=bl[3], fmt="o", color="C0",
                markersize=4, elinewidth=0.6, capsize=1, label="Standard PPO", zorder=3)

    if yscale == "symlog":
        ax.set_yscale("symlog", linthresh=1)
    else:
        ax.set_yscale(yscale)
    # Standard PPO's return: curves that rise while staying near this line lose the least performance
    ax.axvline(bl[0], color="C0", linestyle="--", linewidth=0.6, alpha=0.6, zorder=0)

    ax.set_title(title, fontsize=8)
    ax.set_xlabel(r"Final $J(\theta)$", labelpad=0)
    ax.set_ylabel("RQO", labelpad=2)