import numpy as np
import matplotlib.pyplot as plt
import models
from matplotlib.colors import Normalize
from numpy.linalg import solve
from matplotlib.animation import ArtistAnimation
import matplotlib.gridspec as gridspec

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 0.75
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

r_e_list = np.zeros((3, 50000))
r_i_list = np.zeros((3, 50000))
adapt_list = np.zeros((3, 50000))

# W_ee = 3.5, W_ee = 3.75 sample wave
np.random.seed(6)

plot_s = 50000
plot_f = 100000

for i in range(6):
    test = models.LocalEI(w_ee = [1.5, 3.5, 4.0, 3.75, 1.75, 2.0][i], total_t=100000)
    test.simulate()
    if i == 1:
        r_e_list[0] = test.r_e[plot_s : plot_f]
        r_i_list[0] = test.r_i[plot_s : plot_f]
        adapt_list[0] = test.adapt[plot_s : plot_f]
    elif i == 3:
        r_e_list[1] = test.r_e[plot_s : plot_f]
        r_i_list[1] = test.r_i[plot_s : plot_f]
        adapt_list[1] = test.adapt[plot_s : plot_f]

# W_ee = 3.5, beta_adapt = 1.75 sample wave
test = models.LocalEI(w_ee=3.5, beta_adapt=1.75, total_t=100000)
test.simulate()
r_e_list[2] = test.r_e[plot_s : plot_f]
r_i_list[2] = test.r_i[plot_s : plot_f]
adapt_list[2] = test.adapt[plot_s : plot_f]

ymax = 24.9
ymin = -3.0

def diff(r_e, r_i, w_ee, adapt, I_e, I_i):
    """
    Calculate the differentiation of excitatory and inhibitory firing rates
    """
    para = models.LocalEI(w_ee = w_ee)
    para.set_params()
    dredt = 1/para.tau_e * (- r_e + para.g(para.w_ee * r_e - para.w_ei * r_i - adapt + I_e, para.g_e, para.thre_e))
    dridt = 1/para.tau_i * (- r_i + para.g(para.w_ie * r_e - para.w_ii * r_i + I_i, para.g_i, para.thre_i))
    return dredt, dridt

def calculate_flow(delta_x, delta_y, w_ee, adapt, I_e, I_i):
    """
    Calculate the differentiation of excitatory and inhibitory firing rates at each point
    """
    xmin, xmax = 0, 7.8
    x_range = np.arange(xmin, xmax, delta_x)
    y_range = np.arange(ymin, ymax, delta_y)
    x, y = np.meshgrid(x_range, y_range)
    dx, dy = diff(x, y, w_ee, adapt, I_e, I_i)
    return x, y, dx, dy

default_para = models.LocalEI()
default_para.set_params()

vector_max = 7.9
vector = True
color = True
w_ei = default_para.w_ei
w_ie = default_para.w_ie
w_ii = default_para.w_ii
I_e = default_para.I_e
I_i = default_para.I_i

time_series = np.arange(0, 5000, 0.1)

for block in range(5):
    fig = plt.figure(figsize=(10, 12))
    gs = gridspec.GridSpec(5, 2, height_ratios=[1, 0.1, 1, 0.1, 1])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[4, 0]), fig.add_subplot(gs[4, 1])]

    movie = []
    for plot_t in range(10000*block, 10000*(block+1), 10): 
        movie_one_t = []
        for mode in range(3):
            w_ee = [3.5, 3.75, 3.5][mode]

            ax1 = axes[2 * mode]
            ax2 = axes[2 * mode + 1]

            fig.text(0.5, 0.88-0.28*mode, '$W_{\mathrm{EE}}$' + ' = {}, '.format(w_ee) + r'$\beta$' + ' = {}'.format([0.75, 0.75, 1.75][mode]), ha='center', va='bottom', fontsize=16)
            ax1.set_aspect(250.0)
            ax1.set_xlabel('Time[ms]', labelpad=0)
            ax1.set_ylabel('$r_{\mathrm{E}}$', rotation=0, labelpad=5)
            ax2.set_xlabel('$r_{\mathrm{E}}$', labelpad=-2)
            ax2.set_ylabel('$r_{\mathrm{I}}$', rotation=0, labelpad=5)
            plot5, = ax1.plot(time_series, r_e_list[mode], color='orange')
            ax1.set_ylim(-0.3, 8.0)

            plot6, = ax1.plot(time_series[plot_t], r_e_list[mode, plot_t], marker='o', markersize=5, color='green')

            r_e_set, r_i_set = r_e_list[mode, plot_t], r_i_list[mode, plot_t]
            adapt_fix = adapt_list[mode, plot_t]
                
            x = np.array([0.05* i for i in range(154)])
            e_null = np.zeros(np.size(x))
            i_null = np.zeros(np.size(x))
            for i in range(np.size(x)):
                e_null[i] = 1 / (default_para.g_e * w_ei) * (-x[i] + default_para.g_e * w_ee * x[i] - default_para.thre_e - default_para.g_e * adapt_fix +  default_para.g_e * I_e)
                i_null[i] = default_para.g(w_ie * x[i] +  I_i, default_para.g_i, default_para.thre_i) / (1+default_para.g_i * w_ii)
            plot1, = ax2.plot(x, e_null, color='orange', linewidth=1.0)
            plot2, = ax2.plot(x, i_null, color='navy', linewidth=1.0)
            plot3 = ax2.vlines(x=0, ymin=(- default_para.thre_e - default_para.g_e * adapt_fix +  default_para.g_e * I_e)/(default_para.g_e * w_ei), ymax=i_null[-1], color='orange', linewidth=1.0)

            # plot vector field
            if vector:
                x_sparse, y_sparse, dx_sparse, dy_sparse = calculate_flow(0.4, 1.2, w_ee, adapt_fix, I_e, I_i)
                flow_norm = np.hypot(dx_sparse, dy_sparse)
                dx_sparse /= flow_norm  # normalize each arrow                                    
                dy_sparse /= flow_norm  # normalize each arrow
                if np.max(flow_norm) > vector_max:
                    print('error')
                    
                vectorfield = ax2.quiver(x_sparse, y_sparse, dx_sparse, dy_sparse, flow_norm, norm = Normalize(vmin=0.0, vmax=vector_max), cmap='viridis', alpha=0.6)
                if color and plot_t == 10000*block:
                    fig.colorbar(vectorfield, ax=ax2)

            plot4, = ax2.plot(r_e_set, r_i_set, marker='o', markersize=5, color='green')
            movie_one_t.append([plot1, plot2, plot3, vectorfield, plot4, plot5, plot6])
        movie.append(sum(movie_one_t, []))


    ani = ArtistAnimation(fig, movie, interval=1, blit=True)
    ani.save('../figure/phaseplane_animation_block{}.gif'.format(block), writer='pillow')


