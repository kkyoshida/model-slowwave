import numpy as np
import matplotlib.pyplot as plt
import models
from matplotlib.colors import Normalize
from numpy.linalg import solve

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 0.75
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

np.random.seed(2)

ymax = 12.9
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
    xmin, xmax = 0, 4.81
    x_range = np.arange(xmin, xmax, delta_x)
    y_range = np.arange(ymin, ymax, delta_y)
    x, y = np.meshgrid(x_range, y_range)
    dx, dy = diff(x, y, w_ee, adapt, I_e, I_i)
    return x, y, dx, dy

default_para = models.LocalEI()
default_para.set_params()

vector_max = 4.9
vector_max_a = 0.16

def calculate_re_ri(w_ee, w_ei=default_para.w_ei, w_ie=default_para.w_ie, w_ii=default_para.w_ii):
    #calculate r_e and r_i when I_e=0, I_i=0
    left  = [[default_para.g_e * w_ee - 1 - default_para.beta_adapt * default_para.g_e, - default_para.g_e * w_ei], [default_para.g_i * w_ie, -(default_para.g_i * w_ii + 1)]]
    right = [default_para.thre_e, default_para.thre_i]
    up_r_e, up_r_i = solve(left, right)
    return up_r_e, up_r_i

def calculate_ie_ii(w_ee, up_r_e, up_r_i, w_ei=default_para.w_ei, w_ie=default_para.w_ie, w_ii=default_para.w_ii):
    I_e = -((default_para.g_e * w_ee - 1 - default_para.beta_adapt * default_para.g_e) * up_r_e - default_para.g_e * w_ei * up_r_i - default_para.thre_e) / default_para.g_e
    I_i = -(default_para.g_i * w_ie * up_r_e - (default_para.g_i * w_ii + 1) * up_r_i - default_para.thre_i) / default_para.g_i
    return I_e, I_i

# plot phase plane
def plot_phaseplane(ax, w_ee, adapt_fix, vector=False, color=False, w_ei=default_para.w_ei, w_ie=default_para.w_ie, w_ii=default_para.w_ii, I_e=default_para.I_e, I_i=default_para.I_i):
    x = np.array([0.03* i for i in range(161)])
    e_null = np.zeros(np.size(x))
    i_null = np.zeros(np.size(x))
    for i in range(np.size(x)):
        e_null[i] = 1 / (default_para.g_e * w_ei) * (-x[i] + default_para.g_e * w_ee * x[i] - default_para.thre_e - default_para.g_e * adapt_fix +  default_para.g_e * I_e)
        i_null[i] = default_para.g(w_ie * x[i] +  I_i, default_para.g_i, default_para.thre_i) / (1+default_para.g_i * w_ii)
    ax.plot(x, e_null, color='orange', linewidth=1.0)
    ax.plot(x, i_null, color='navy', linewidth=1.0)
    ax.vlines(x=0, ymin=(- default_para.thre_e - default_para.g_e * adapt_fix +  default_para.g_e * I_e)/(default_para.g_e * w_ei), ymax=i_null[-1], color='orange', linewidth=1.0)
    
    # plot vector field
    if vector:
        x_sparse, y_sparse, dx_sparse, dy_sparse = calculate_flow(0.4, 1.2, w_ee, adapt_fix, I_e, I_i)
        flow_norm = np.hypot(dx_sparse, dy_sparse)
        dx_sparse /= flow_norm  # normalize each arrow                                    
        dy_sparse /= flow_norm  # normalize each arrow
        if np.max(flow_norm) > vector_max:
            print('error')
        vectorfield = ax.quiver(x_sparse, y_sparse, dx_sparse, dy_sparse, flow_norm, norm = Normalize(vmin=0.0, vmax=vector_max), cmap='viridis', alpha=0.6)
        if color:
            fig.colorbar(vectorfield, ax=ax)

# phase plane: adaptation induced, noise-induced
class LocalEINullcline(models.LocalEI):
    def __init__(self, w_ee, adapt_init, r_e_initial, r_i_initial, perturb_strength, w_ei=default_para.w_ei, w_ie=default_para.w_ie, w_ii=default_para.w_ii, I_e=0, I_i=0):
        super().__init__(w_ee=w_ee, w_ei=w_ei, w_ie=w_ie, w_ii=w_ii, noise_strength_e=0.0, noise_strength_i=0.0, I_e=I_e, I_i=I_i)
        self.adapt_init = adapt_init
        self.r_e = r_e_initial * np.ones(self.total_t)
        self.r_i = r_i_initial * np.ones(self.total_t)
        self.perturb_strength = perturb_strength
        
    def simulate(self):
        self.set_params()
        self.I_ext_e[50000] += self.perturb_strength
        self.adapt = self.adapt_init * np.ones(self.total_t)
        for t in range(self.total_t -1):
            self.r_e[t+1] = self.r_e[t] + self.dt * 1/self.tau_e * (-self.r_e[t] + self.g(self.w_ee * self.r_e[t] - self.w_ei * self.r_i[t] - self.adapt[t] + self.I_ext_e[t], self.g_e, self.thre_e))
            self.r_i[t+1] = self.r_i[t] + self.dt * 1/self.tau_i * (-self.r_i[t] + self.g(self.w_ie * self.r_e[t] - self.w_ii * self.r_i[t] + self.I_ext_i[t], self.g_i, self.thre_i))



class LocalEINullcline_re_a(models.LocalEI):
    def __init__(self, w_ee, beta_adapt, r_e_initial, r_i_initial, a_initial):
        super().__init__(w_ee=w_ee, beta_adapt=beta_adapt, noise_strength_e=0.0, noise_strength_i=0.0)
        self.r_e = r_e_initial * np.ones(self.total_t)
        self.r_i = r_i_initial * np.ones(self.total_t)
        self.a_initial = a_initial

    def set_params(self):
        super().set_params()
        self.adapt = self.a_initial * np.ones(self.total_t)


def diff_re_a(r_e, adapt, w_ee, beta_adapt):
    r_i = np.where(default_para.g(default_para.w_ie * r_e, default_para.g_i, default_para.thre_i) > 0, (default_para.g_i * default_para.w_ie * r_e - default_para.thre_i) / (1 + default_para.g_i * default_para.w_ii), 0)
    dredt = 1 / default_para.tau_e * (- r_e + default_para.g(w_ee * r_e - default_para.w_ei * r_i - adapt, default_para.g_e, default_para.thre_e))
    dadt = 1 / default_para.tau_adapt * (- adapt + beta_adapt * r_e)
    return dredt, dadt

def calculate_flow_re_a(w_ee, beta_adapt):
    xmin, xmax = 0.05, 1.65
    amin, amax = -0.05, 1.2
    x_range = np.arange(xmin, xmax, 0.15)
    y_range = np.arange(amin, amax, 0.15)
    x, y = np.meshgrid(x_range, y_range)
    dx, dy = diff_re_a(x, y, w_ee, beta_adapt)
    return x, y, dx, dy
        
# plot phase plane
def plot_phaseplane_re_a(ax, w_ee, beta_adapt=default_para.beta_adapt, vector=False, color=False, w_ei=default_para.w_ei, w_ie=default_para.w_ie, w_ii=default_para.w_ii):
    y = np.array([0.01* i for i in range(121)])
    e_null = np.zeros(np.size(y))
    e_null_2 = np.zeros(np.size(y))
    a_null = np.zeros(np.size(y))
    for i in range(np.size(y)):
        a_null[i] = 1/beta_adapt * y[i]

        #in the case of r_e>0 and r_i>0
        left  = [[default_para.g_e * w_ee - 1, - default_para.g_e * w_ei], [default_para.g_i * w_ie, -(default_para.g_i * w_ii + 1)]]
        right = [default_para.thre_e + default_para.g_e * y[i], default_para.thre_i]
        up_r_e, up_r_i = solve(left, right)
        if default_para.g(w_ie * up_r_e - w_ii * up_r_i, default_para.g_i, default_para.thre_i) > 0 and default_para.g(w_ee * up_r_e - w_ei * up_r_i - y[i], default_para.g_e, default_para.thre_e) > 0: 
            e_null[i] = up_r_e

        #in the case of r_e>0 and r_i=0
        up_r_e = (y[i] * default_para.g_e + default_para.thre_e) / (default_para.g_e * w_ee - 1)
        up_r_i = 0.0
        if default_para.g(w_ie * up_r_e - w_ii * up_r_i, default_para.g_i, default_para.thre_i) == 0 and default_para.g(w_ee * up_r_e - w_ei * up_r_i - y[i], default_para.g_e, default_para.thre_e) > 0:
            e_null_2[i] = up_r_e

    ax.plot(e_null[e_null!=0], y[e_null!=0], color='orange', linewidth=1.0)
    ax.plot(e_null_2[e_null_2!=0], y[e_null_2!=0], color='orange', linewidth=1.0)
    ax.plot(a_null, y, color='cyan', linewidth=1.0)
    #in the case of r_e=0, and thus r_i=0
    ax.vlines(x=0, ymin=-default_para.thre_e, ymax=1.2, color='orange', linewidth=1.0)
    ax.set_xlim(-0.05, 1.65)
    
   
    # plot vector field
    if vector:
        x_sparse, y_sparse, dx_sparse, dy_sparse = calculate_flow_re_a(w_ee, beta_adapt)
        flow_norm = np.hypot(dx_sparse, dy_sparse)
        dx_sparse /= flow_norm  # normalize each arrow                                    
        dy_sparse /= flow_norm  # normalize each arrow
        if np.max(flow_norm) > vector_max_a:
            print('error')
        vectorfield = ax.quiver(x_sparse, y_sparse, dx_sparse, dy_sparse, flow_norm, norm=Normalize(vmin=0.0, vmax=vector_max_a), cmap='viridis', alpha=0.6)
        if color:
            fig.colorbar(vectorfield, ax=ax)

    sw_trajectory = LocalEINullcline_re_a(w_ee=w_ee, beta_adapt=beta_adapt, r_e_initial=0.0, r_i_initial=0.0, a_initial=-sw.thre_e)
    sw_trajectory.simulate()
    ax.plot(sw_trajectory.r_e[0:100000], sw_trajectory.adapt[0:100000], color='green', zorder=-1, linewidth=2.0)


fig = plt.figure(figsize = (7, 14))

#r_e, r_i phase plane
pert_strength = 40.0

w_ee_set = 3.5
r_e_set, r_i_set = calculate_re_ri(w_ee=w_ee_set)
adapt_set = default_para.beta_adapt * r_e_set
sw = LocalEINullcline(w_ee=w_ee_set, adapt_init=adapt_set, r_e_initial=r_e_set, r_i_initial=r_i_set, perturb_strength=pert_strength)
sw.simulate()
ax = fig.add_subplot(4, 2, 1)
plot_phaseplane(ax, w_ee=w_ee_set, adapt_fix=adapt_set, vector=True)
ax.plot(sw.r_e[50000:100000], sw.r_i[50000:100000], color='green', linewidth=1.0)
ax.plot(0.0, 0.0, marker='s', markersize=7, color='blue')
ax.plot(r_e_set, r_i_set, marker='s', markersize=7, color='red')
ax.set_ylim(ymin, ymax)

ax = fig.add_subplot(4, 2, 4)
plot_phaseplane(ax, w_ee=w_ee_set, adapt_fix=adapt_set, vector=True, color=True)
ax.set_ylim(ymin, ymax)

# w_ee increase
w_ee_increase = 4.0
r_e_set, r_i_set = calculate_re_ri(w_ee=w_ee_increase)
adapt_set = default_para.beta_adapt * r_e_set
sw = LocalEINullcline(w_ee=w_ee_increase, adapt_init=adapt_set, r_e_initial=r_e_set, r_i_initial=r_i_set, perturb_strength=pert_strength)
sw.simulate()
ax = fig.add_subplot(4, 2, 2)
plot_phaseplane(ax, w_ee=w_ee_increase, adapt_fix=adapt_set, vector=True)
ax.plot(sw.r_e[50000:100000], sw.r_i[50000:100000], color='green', linewidth=1.0)
ax.plot(0.0, 0.0, marker='s', markersize=7, color='blue')
ax.plot(r_e_set, r_i_set, marker='D', markersize=7, color='red')
ax.set_ylim(ymin, ymax)

# I_e increase
I_e_set, I_i_set = calculate_ie_ii(w_ee_set, r_e_set, r_i_set)
sw = LocalEINullcline(w_ee=w_ee_set, adapt_init=adapt_set, r_e_initial=r_e_set, r_i_initial=r_i_set, perturb_strength=pert_strength, I_e=I_e_set, I_i=I_i_set)
sw.simulate()
ax = fig.add_subplot(4, 2, 3)
plot_phaseplane(ax, w_ee=w_ee_set, adapt_fix=adapt_set, I_e=I_e_set, I_i=I_i_set, vector=True)
print(I_e_set, I_i_set)
ax.plot(sw.r_e[50000:100000], sw.r_i[50000:100000], color='green', linewidth=1.0)
ax.plot(0.0, 0.0, marker='s', markersize=7, color='blue')
ax.plot(r_e_set, r_i_set, marker='s', markersize=7, color='red')
ax.set_ylim(ymin, ymax)

#r_e, a phase plane
ax = fig.add_subplot(4, 2, 5)
plot_phaseplane_re_a(ax, w_ee=1.5, vector=True)
ax = fig.add_subplot(4, 2, 6)
plot_phaseplane_re_a(ax, w_ee=1.5, vector=True, color=True)
ax = fig.add_subplot(4, 2, 7)
plot_phaseplane_re_a(ax, w_ee=1.75, vector=True)
ax = fig.add_subplot(4, 2, 8)
plot_phaseplane_re_a(ax, w_ee=2.0, vector=True)
plt.savefig('../figure/phaseplane.pdf')
