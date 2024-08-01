import numpy as np
import matplotlib.pyplot as plt
import params

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 0.75
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

result = np.load('../data/twodbifurcationach.npy')

div_n = 101
combination_n = 1

w_ee_list = np.linspace(3.5, 4.0, div_n)
I_e_list = np.linspace(-0.2, 0.2, div_n)
beta_adapt_list = np.linspace(0.50, 1.00, div_n)
alpha_list = np.linspace(0.95, 1.05, div_n)

fig = plt.figure(figsize = (16, 40))

for combination in range(combination_n):
    for i in range(2):
        ax = fig.add_subplot(10, 4, 4*combination+i+1)
        img = ax.imshow(result[combination].T, interpolation='none', origin='lower', cmap='PiYG', vmin=-1, vmax=1)
        ax.set_xticks([0, div_n-1])
        ax.set_yticks([0, div_n-1])
 
        ax.set_ylabel('$W_{\mathrm{EE}}$', rotation=0, labelpad=-5)
        ax.set_yticklabels([w_ee_list[0], w_ee_list[div_n-1]]) 
        
        labelpad_x = -10
        
        ax.set_xlabel('neurotransmitter', labelpad=labelpad_x)
        ax.set_xticklabels([-1, 1]) 

        if i == 1:
            fig.colorbar(img)

plt.savefig('../figure/wee_2dclearach.pdf')
