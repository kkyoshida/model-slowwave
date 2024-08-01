import numpy as np
import matplotlib.pyplot as plt
import models

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 0.75
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

np.random.seed(6)

#change the W_ee
fig = plt.figure(figsize = (18, 6))
plot_s = 50000
plot_f = 100000

for i in range(6):
    test = models.LocalEI(w_ee = [1.5, 3.5, 4.0, 3.75, 1.75, 2.0][i], total_t=100000)
    test.simulate()
    time_series = np.arange(0, 5000, 0.1)
    ax = fig.add_subplot(3, 6, 1 + i)
    ax.plot(time_series, test.r_e[plot_s : plot_f], color='orange')
    ax.set_ylim(-0.3, 13.0)
    ax = fig.add_subplot(3, 6, 7 + i)
    ax.plot(time_series, test.r_i[plot_s : plot_f], color='navy')
    ax.set_ylim(-0.3, 40.0)
    ax = fig.add_subplot(3, 6, 13 + i)
    ax.plot(time_series, test.adapt[plot_s : plot_f])
    ax.set_ylim(0, 2.2)
    
plt.savefig("../figure/samplewave.pdf")
