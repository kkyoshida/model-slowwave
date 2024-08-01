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
plot_s = 50000 
plot_f = 250000
result = np.zeros(251)
w_ee_list = np.array([3.5+0.002*i for i in range(251)])

for i in range(251):
    test = models.LocalEI(w_ee=w_ee_list[i], total_t=250000)
    test.simulate()
    
    if test.judgeslowwave(test.r_e) == 0:
        result[i] = -1 # not physiological
    else:
        f_power = np.abs(np.fft.fft(test.r_e[plot_s : plot_f])/(np.size(test.r_e[plot_s : plot_f])/2)) ** 2 # power spectrum
        result[i] = np.sum(f_power[int(1.0*(plot_f-plot_s)/10000) : int(4.0*(plot_f-plot_s)/10000)+1]) # 1.0-4.0Hz

f_range = np.fft.fftfreq(n=np.size(test.r_e[plot_s : plot_f]), d=0.0001)[int(1.0*(plot_f-plot_s)/10000) : int(4.0*(plot_f-plot_s)/10000)+1]
print(f_range[0], f_range[-1])

fig = plt.figure(figsize = (3, 3))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(w_ee_list, result, s=1)
plt.savefig('../figure/deltapower.pdf')
