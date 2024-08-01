import numpy as np
import matplotlib.pyplot as plt
import params

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 0.75
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

result = np.load('../data/twodbifurcationinhibitory.npy')

div_n = 101
combination_n = 4

# parameters
para = params.Para(div_n)
beta_adapt_list, w_ee_list, w_ee_list_wide, w_ei_list, w_ie_list, w_ii_list, I_e_list, I_i_list, noise_e_list, noise_i_list = para.giveparams()

w_ei_list = np.linspace(0.5, 2.0, div_n) 
w_ie_list = np.linspace(0.8, 2.0, div_n)

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
        
        if combination == 0:
            ax.set_xlabel('$W_{\mathrm{EI}}$', labelpad=labelpad_x)
            ax.set_xticklabels([w_ei_list[0], w_ei_list[div_n-1]]) 
        elif combination == 1:
            ax.set_xlabel('$W_{\mathrm{IE}}$', labelpad=labelpad_x)
            ax.set_xticklabels([w_ie_list[0], w_ie_list[div_n-1]]) 
        elif combination == 2:
            ax.set_xlabel('$W_{\mathrm{EI}}$', labelpad=labelpad_x)
            ax.set_xticklabels([w_ei_list[0], w_ei_list[div_n-1]]) 
        elif combination == 3:
            ax.set_xlabel('$W_{\mathrm{IE}}$', labelpad=labelpad_x)
            ax.set_xticklabels([w_ie_list[0], w_ie_list[div_n-1]])      

        if i == 1:
            fig.colorbar(img)

plt.savefig('../figure/wee_2dclearinhibitory.pdf')
