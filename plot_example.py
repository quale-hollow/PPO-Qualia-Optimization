from utils.plotting_utils import plot_curves


environment = 'cartpole'
method = 'method1'
norm = 'norm' 

# Change this to the values you have results for and want to plot
omegas = [
    ["0.1"],
    [],
    []
]


line_colors = ['C1', 'C2', 'C3', 'C4', 'C5']
names = ["Method 1 - Advantage Inflation", "Method 2 - Asymmetric Clipping", "Method 3 - Additive Regularizer"]
methods = ['method1', 'method2', 'method3']


plot_curves(methods, names, omegas, norm, line_colors, 'test_dir', 'cartpole', save_dir="test_dir", qe_lims=None, NUM_TRIALS=300, NUM_BINS=100)
