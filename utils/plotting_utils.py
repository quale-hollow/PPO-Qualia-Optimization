import numpy as np
import os
import matplotlib.pyplot as plt

def get_returns_data(filepath, num_bins=100, max_trials=None):

    returns_combined = np.empty((0, 2))
    count = 0

    files = os.listdir(filepath)
    if max_trials is not None:
        files = files[:max_trials]

    for fname in files:
        data = np.genfromtxt(filepath + fname, delimiter=',')
        returns_combined = np.vstack((returns_combined, data))
        count += 1

    returns_combined = returns_combined[returns_combined[:, 0].argsort()]

    bin_size = len(returns_combined) // num_bins

    mean_returns_bins = np.zeros(num_bins)
    stderr_returns_bins = np.zeros(num_bins)
    grouped_timesteps = np.zeros(num_bins)
    for i in range(num_bins):
        grouped_timesteps[i] = np.mean(returns_combined[(i * bin_size):((i+1) * bin_size), 0])

        if i == num_bins - 1:
            group_values = returns_combined[i * bin_size:, 1]
        else:
            group_values = returns_combined[i * bin_size:(i + 1) * bin_size, 1]
        mean_returns_bins[i] = np.mean(group_values)
        stderr_returns_bins[i] = np.std(group_values) / np.sqrt(len(group_values))

    
    return grouped_timesteps, mean_returns_bins, stderr_returns_bins, count


def get_binned_reinforcement_data(filepath, num_bins=100, max_trials=None):
    reinforcement_combined = np.empty((0, 2))
    count = 0
    files = os.listdir(filepath)
    if max_trials is not None:
        files = files[:max_trials]

    for fname in files:
        data = np.genfromtxt(filepath + fname, delimiter=',')
        reinforcement_combined = np.vstack((reinforcement_combined, data))
        count += 1

    reinforcement_combined = reinforcement_combined[reinforcement_combined[:, 0].argsort()]

    bin_size = len(reinforcement_combined) // num_bins

    mean_returns_bins = np.zeros(num_bins)
    stderr_returns_bins = np.zeros(num_bins)
    grouped_timesteps = np.zeros(num_bins)
    for i in range(num_bins):
        grouped_timesteps[i] = np.mean(reinforcement_combined[(i * bin_size):((i+1) * bin_size), 0])

        if i == num_bins - 1:
            group_values = reinforcement_combined[i * bin_size:, 1]
        else:
            group_values = reinforcement_combined[i * bin_size:(i + 1) * bin_size, 1]

        group_values -= 1
        mean_returns_bins[i] = np.mean(group_values)
        stderr_returns_bins[i] = np.std(group_values) / np.sqrt(len(group_values))

    
    return grouped_timesteps, mean_returns_bins, stderr_returns_bins, count



def get_final_return(filepath, max_trials=None):
    final_returns = []
    files = os.listdir(filepath)
    if max_trials is not None:
        files = files[:max_trials]

    for fname in files:
        data = np.genfromtxt(filepath + fname, delimiter=',')
        final_return = data[-1]
        final_returns.append(final_return)

    count = len(final_returns)
    mean_final_return = np.mean(final_returns)
    stderr_final_return = np.std(final_returns) / np.sqrt(count)

    return mean_final_return, stderr_final_return



def get_final_qualia_return(filepath, max_trials=None):
    qualia_returns = []
    files = os.listdir(filepath)
    if max_trials is not None:
        files = files[:max_trials]

    for fname in files:
        data = np.genfromtxt(filepath + fname, delimiter=',')
        qualia_experiences = data[:, 1] - 1
        qualia_return = np.sum(qualia_experiences)
        qualia_returns.append(qualia_return)

    mean_qualia_return = np.mean(qualia_returns)
    stderr_qualia_return = np.std(qualia_returns) / np.sqrt(len(files))

    return mean_qualia_return, stderr_qualia_return


# graphing functions from: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
def set_size(width, fraction=1, subplots=(1, 1)):
    # Width of figure (in pts)
    fig_width_pt = width * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def axes_labels(x, pos):
    if x >= 1e6:
        return f'{x*1e-6:.0f}M'
    elif x >= 1e3:
        if x % 1000 == 0:
            return f'{x*1e-3:.0f}K'
        else:
            return f'{x*1e-3:.1f}K'
    elif x < 1:
        return f'{x:.2f}'
    else:
        return f'{x:.0f}'
    


def plot_curves(methods, names, omegas, norm, line_colors, path, env, save_dir="plots", qe_lims=None, NUM_TRIALS=None, NUM_BINS=100):

    width =  397.48499
    inches_per_pt = 1 / 72.27

    fig_width = width * inches_per_pt
    fig_height = fig_width * (1/4)
    layout = (1, 4)
    fig_dim = (fig_width, fig_height)

    # fig_dim = set_size(width, fraction=1, subplots=layout)

    print("Plotting results for ", env, " with ", norm)

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "font.size": 8,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6
    }

    plt.rcParams.update(tex_fonts)
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['xtick.major.pad'] = 0.5
    plt.rcParams['ytick.major.pad'] = 0.5

    
    bl_ret_x, bl_ret_y, bl_ret_stderr, bl_ret_count = get_returns_data(f'{path}/control/omega_0.0/timestep_returns/', num_bins=NUM_BINS, max_trials=NUM_TRIALS)
    print("Count for baseline: ", bl_ret_count)
    bl_rf_x, bl_rf_y, bl_rf_stderr, _ = get_binned_reinforcement_data(f'{path}/control/omega_0.0/mean_ratios/', num_bins=NUM_BINS, max_trials=NUM_TRIALS)

    bl_final_return, bl_final_return_std = get_final_return(f'{path}/control/omega_0.0/ep_returns/')
    bl_qr, bl_qr_std = get_final_qualia_return(f'{path}/control/omega_0.0/mean_ratios/')
    
    table1 = f"Standard PPO & N/A & {bl_final_return:.2f}$\pm${bl_final_return_std:.2f} & {bl_qr:.2f}$\pm${bl_qr_std:.2f} {chr(92)}{chr(92)} {chr(92)}midrule\n"

    for i, method in enumerate(methods):

        baseline_omega = "1.0" if method == "method2" else "0.0"
        control_x = 1 if method == 'method2' else 0


        fig, ax = plt.subplots(layout[0], layout[1], figsize=fig_dim, constrained_layout=True)
        fig.suptitle(names[i], fontsize=9)
        label_pad = 0

        ax[0].set_xlabel('Time Step', labelpad=label_pad)
        ax[0].set_ylabel(r'Return', labelpad=label_pad)
        ax[0].xaxis.set_major_formatter(plt.FuncFormatter(axes_labels))
        ax[0].yaxis.set_major_formatter(plt.FuncFormatter(axes_labels))

        ax[1].set_xlabel('Time Step', labelpad=label_pad)
        ax[1].set_ylabel(r'Per-Update $Q_i$', labelpad=label_pad)
        ax[1].xaxis.set_major_formatter(plt.FuncFormatter(axes_labels))
        if qe_lims is not None:
            ax[1].set_ylim(*qe_lims)

        ax[2].set_xlabel(r'$\omega$',labelpad=label_pad)
        ax[2].set_ylabel('Final Return',labelpad=label_pad)
        ax[2].yaxis.set_major_formatter(plt.FuncFormatter(axes_labels))
        

        ax[3].set_xlabel(r'$\omega$', labelpad=label_pad)
        ax[3].set_ylabel('Final RQO', labelpad=label_pad)
        ax[3].yaxis.set_major_formatter(plt.FuncFormatter(axes_labels))


        # plot control data
        ax[0].plot(bl_ret_x, bl_ret_y, label=fr'$\omega=${baseline_omega} (standard PPO)', color='C0')
        ax[0].fill_between(bl_ret_x, bl_ret_y - bl_ret_stderr, bl_ret_y + bl_ret_stderr, alpha=0.3, color='C0')
        
        ax[1].plot(bl_rf_x, bl_rf_y, color='C0')
        ax[1].fill_between(bl_rf_x, bl_rf_y - bl_rf_stderr, bl_rf_y + bl_rf_stderr, alpha=0.3, color='C0')

        ax[2].errorbar(control_x, bl_final_return, yerr=bl_final_return_std, fmt='o', color='C0')
        ax[3].errorbar(control_x, bl_qr, yerr=bl_qr_std, fmt='o', color='C0')


        # PLOT LEARNING CURVES
        for j, omega in enumerate(omegas[i]):
            x, y, stderr, count = get_returns_data(f'{path}/' + method + '/' + 'omega_' + omega + '/timestep_returns/', num_bins=NUM_BINS, max_trials=NUM_TRIALS)
            print("Count for ", method, omega, norm, ": ", count)

            ax[0].plot(x, y, label=fr"$\omega=${omega}", color=line_colors[j])
            ax[0].fill_between(x, y-stderr, y+stderr, alpha=0.3, color=line_colors[j])
            
            x, y, stderr, count = get_binned_reinforcement_data(f'{path}/' + method + '/' + 'omega_' + omega + '/mean_ratios/', num_bins=NUM_BINS, max_trials=NUM_TRIALS)
            
            ax[1].plot(x, y, color=line_colors[j])
            ax[1].fill_between(x, y-stderr, y+stderr, alpha=0.3, color=line_colors[j])

            # OMEGA PLOTS AND LATEX TABLES
            ret, ret_std = get_final_return(f'{path}/{method}/omega_{omega}/ep_returns/')
            QR, QR_std = get_final_qualia_return(f'{path}/{method}/omega_{omega}/mean_ratios/')
            
            # add to tex table
            table1 += (f"{names[i]} & ") if j == 0 else (f"\t & ")
            table1 += (f"{omega} & {ret:.2f}$\pm${ret_std:.2f} & {QR:.2f}$\pm${QR_std:.2f} {chr(92)}{chr(92)}")
            if j == len(omegas[i]) - 1:
                table1 += f"{chr(92)}midrule"
            table1 += "\n"

            ax[2].errorbar(float(omega), ret, yerr=ret_std, fmt='o', color=line_colors[j])
            ax[3].errorbar(float(omega), QR, yerr=QR_std, fmt='o', color=line_colors[j])


        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', mode="expand", ncol=6, columnspacing=1, handletextpad=0.2, borderpad=0, borderaxespad=0, frameon=False)
        fig.get_layout_engine().set(h_pad=0.01)

        plt.savefig(f'{save_dir}/{env}_{method}_{norm}.pdf', format='pdf', bbox_inches='tight')
        plt.clf()


    print("\n\n")
    print(table1)