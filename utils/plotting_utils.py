import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import t


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


def get_returns_data(filepath, num_bins=100, max_trials=None):
    """
    Returns:
        grouped_timesteps, mean_returns_bins, ci95_returns_bins, count
    """
    returns_combined = np.empty((0, 2))
    count = 0

    files = os.listdir(filepath)
    if max_trials is not None:
        files = files[:max_trials]

    for fname in files:
        data = np.genfromtxt(os.path.join(filepath, fname), delimiter=",")
        returns_combined = np.vstack((returns_combined, data))
        count += 1

    returns_combined = returns_combined[returns_combined[:, 0].argsort()]

    bin_size = max(1, len(returns_combined) // num_bins)

    mean_returns_bins = np.zeros(num_bins)
    ci95_returns_bins = np.zeros(num_bins)
    grouped_timesteps = np.zeros(num_bins)

    for i in range(num_bins):
        start = i * bin_size
        if start >= len(returns_combined):
            grouped_timesteps[i] = grouped_timesteps[i - 1] if i > 0 else 0.0
            mean_returns_bins[i] = mean_returns_bins[i - 1] if i > 0 else 0.0
            ci95_returns_bins[i] = 0.0
            continue

        end = len(returns_combined) if i == num_bins - 1 else min(len(returns_combined), (i + 1) * bin_size)

        grouped_timesteps[i] = np.mean(returns_combined[start:end, 0])
        group_values = returns_combined[start:end, 1]

        mean_returns_bins[i] = np.mean(group_values)
        ci95_returns_bins[i] = ci95_halfwidth(group_values)

    return grouped_timesteps, mean_returns_bins, ci95_returns_bins, count


def get_binned_reinforcement_data(filepath, num_bins=100, max_trials=None):
    """
    Returns:
        grouped_timesteps, mean_bins, ci95_bins, count
    """
    reinforcement_combined = np.empty((0, 2))
    count = 0

    files = os.listdir(filepath)
    if max_trials is not None:
        files = files[:max_trials]

    for fname in files:
        data = np.genfromtxt(os.path.join(filepath, fname), delimiter=",")
        reinforcement_combined = np.vstack((reinforcement_combined, data))
        count += 1

    reinforcement_combined = reinforcement_combined[reinforcement_combined[:, 0].argsort()]

    bin_size = max(1, len(reinforcement_combined) // num_bins)

    mean_bins = np.zeros(num_bins)
    ci95_bins = np.zeros(num_bins)
    grouped_timesteps = np.zeros(num_bins)

    for i in range(num_bins):
        start = i * bin_size
        if start >= len(reinforcement_combined):
            grouped_timesteps[i] = grouped_timesteps[i - 1] if i > 0 else 0.0
            mean_bins[i] = mean_bins[i - 1] if i > 0 else 0.0
            ci95_bins[i] = 0.0
            continue

        end = len(reinforcement_combined) if i == num_bins - 1 else min(len(reinforcement_combined), (i + 1) * bin_size)

        grouped_timesteps[i] = np.mean(reinforcement_combined[start:end, 0])
        group_values = reinforcement_combined[start:end, 1] - 1.0

        mean_bins[i] = np.mean(group_values)
        ci95_bins[i] = ci95_halfwidth(group_values)

    return grouped_timesteps, mean_bins, ci95_bins, count


def get_final_return(filepath, max_trials=None):
    """
    Returns:
        mean_final_return, ci95_final_return
    """
    final_returns = []

    files = os.listdir(filepath)
    if max_trials is not None:
        files = files[:max_trials]

    for fname in files:
        data = np.genfromtxt(os.path.join(filepath, fname), delimiter=",")
        final_returns.append(data[-1])

    final_returns = np.asarray(final_returns, dtype=np.float64)
    mean_final_return = float(np.mean(final_returns))
    ci95_final_return = float(ci95_halfwidth(final_returns))

    return mean_final_return, ci95_final_return


def get_final_qualia_return(filepath, max_trials=None):
    """
    Returns:
        mean_qualia_return, ci95_qualia_return
    """
    qualia_returns = []

    files = os.listdir(filepath)
    if max_trials is not None:
        files = files[:max_trials]

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
    if x >= 1e6:
        return f"{x*1e-6:.0f}M"
    elif x >= 1e3:
        if x % 1000 == 0:
            return f"{x*1e-3:.0f}K"
        else:
            return f"{x*1e-3:.1f}K"
    elif x < 1:
        return f"{x:.2f}"
    else:
        return f"{x:.0f}"


def plot_curves(
    methods,
    names,
    omegas,
    norm,
    line_colors,
    path,
    env,
    save_dir="plots",
    qe_lims=None,
    NUM_TRIALS=None,
    NUM_BINS=100,
):
    width = 397.48499
    inches_per_pt = 1 / 72.27

    fig_width = width * inches_per_pt
    fig_height = fig_width * (1 / 4)
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
    bl_ret_x, bl_ret_y, bl_ret_ci, bl_ret_count = get_returns_data(
        f"{path}/control/omega_0.0/timestep_returns/",
        num_bins=NUM_BINS,
        max_trials=NUM_TRIALS,
    )
    print("Count for baseline: ", bl_ret_count)

    bl_rf_x, bl_rf_y, bl_rf_ci, _ = get_binned_reinforcement_data(
        f"{path}/control/omega_0.0/mean_ratios/",
        num_bins=NUM_BINS,
        max_trials=NUM_TRIALS,
    )

    bl_final_return, bl_final_return_ci = get_final_return(
        f"{path}/control/omega_0.0/ep_returns/",
        max_trials=NUM_TRIALS,
    )
    bl_qr, bl_qr_ci = get_final_qualia_return(
        f"{path}/control/omega_0.0/mean_ratios/",
        max_trials=NUM_TRIALS,
    )

    table1 = (
        f"Standard PPO & N/A & "
        f"{bl_final_return:.2f}$\\pm${bl_final_return_ci:.2f} & "
        f"{bl_qr:.2f}$\\pm${bl_qr_ci:.2f} {chr(92)}{chr(92)} {chr(92)}midrule\n"
    )

    # ---------- METHODS ----------
    for i, method in enumerate(methods):
        baseline_omega = "1.0" if method == "method2" else "0.0"
        control_x = 1 if method == "method2" else 0

        fig, ax = plt.subplots(layout[0], layout[1], figsize=fig_dim, constrained_layout=True)
        fig.suptitle(names[i], fontsize=9)
        label_pad = 0

        ax[0].set_xlabel("Time Step", labelpad=label_pad)
        ax[0].set_ylabel(r"Return", labelpad=label_pad)
        ax[0].xaxis.set_major_formatter(plt.FuncFormatter(axes_labels))
        ax[0].yaxis.set_major_formatter(plt.FuncFormatter(axes_labels))

        ax[1].set_xlabel("Time Step", labelpad=label_pad)
        ax[1].set_ylabel(r"Per-Update $Q_i$", labelpad=label_pad)
        ax[1].xaxis.set_major_formatter(plt.FuncFormatter(axes_labels))
        if qe_lims is not None:
            ax[1].set_ylim(*qe_lims)

        ax[2].set_xlabel(r"$\omega$", labelpad=label_pad)
        ax[2].set_ylabel("Final Return", labelpad=label_pad)
        ax[2].yaxis.set_major_formatter(plt.FuncFormatter(axes_labels))

        ax[3].set_xlabel(r"$\omega$", labelpad=label_pad)
        ax[3].set_ylabel("Final RQO", labelpad=label_pad)
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
            x, y, ci, count = get_returns_data(
                f"{path}/{method}/omega_{omega}/timestep_returns/",
                num_bins=NUM_BINS,
                max_trials=NUM_TRIALS,
            )
            print("Count for ", method, omega, norm, ": ", count)

            ax[0].plot(x, y, label=fr"$\omega={omega}$", color=line_colors[j])
            ax[0].fill_between(x, y - ci, y + ci, alpha=0.3, color=line_colors[j])

            x, y, ci, _ = get_binned_reinforcement_data(
                f"{path}/{method}/omega_{omega}/mean_ratios/",
                num_bins=NUM_BINS,
                max_trials=NUM_TRIALS,
            )
            ax[1].plot(x, y, color=line_colors[j])
            ax[1].fill_between(x, y - ci, y + ci, alpha=0.3, color=line_colors[j])

            # Final metrics + table
            ret, ret_ci = get_final_return(f"{path}/{method}/omega_{omega}/ep_returns/", max_trials=NUM_TRIALS)
            QR, QR_ci = get_final_qualia_return(f"{path}/{method}/omega_{omega}/mean_ratios/", max_trials=NUM_TRIALS)

            table1 += (f"{names[i]} & ") if j == 0 else (f"\t & ")
            table1 += (
                f"{omega} & {ret:.2f}$\\pm${ret_ci:.2f} & "
                f"{QR:.2f}$\\pm${QR_ci:.2f} {chr(92)}{chr(92)}"
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
            ncol=6,
            columnspacing=1,
            handletextpad=0.2,
            borderpad=0,
            borderaxespad=0,
            frameon=False,
        )
        fig.get_layout_engine().set(h_pad=0.01)

        plt.savefig(f"{save_dir}/{env}_{method}_{norm}.pdf", format="pdf", bbox_inches="tight")
        plt.clf()

    print("\n\n")
    print(table1)