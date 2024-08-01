import numpy as np
import matplotlib.pyplot as plt
import params

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 0.75
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

div_n = 101
combination_n = 2
para = params.Para(div_n)
w_ee_list, w_ee_12_list = para.giveparams_twopopu()

result = np.load('../data/twopopu1.npy')
result2 = np.load('../data/twopopu2.npy')

fig = plt.figure(figsize = (8, 16))

for combination in range(combination_n):
    for popu in range(2):
        for i in range(2):
            ax = fig.add_subplot(4, 2, 4*combination+2*popu+i+1)
            if popu == 0:
                img = ax.imshow(result[combination].T, interpolation='none', origin='lower', cmap='PiYG', vmin=-1, vmax=1)
            else:
                img = ax.imshow(result2[combination].T, interpolation='none', origin='lower', cmap='PiYG', vmin=-1, vmax=1)
            
            if i==0:
                ax.set_xticks([0, div_n-1])
                ax.set_yticks([0, div_n-1])

                ax.set_xticklabels([w_ee_list[0], w_ee_list[div_n-1]]) 
                ax.set_xlabel('$W_{\mathrm{EE}}^1$', labelpad=-10)
                
                if combination == 0:
                    ax.set_yticklabels([w_ee_list[0], w_ee_list[div_n-1]]) 
                    ax.set_ylabel('$W_{\mathrm{EE}}^2$', rotation=0, labelpad=-5)
                elif combination == 1:
                    ax.set_yticklabels([w_ee_12_list[0], w_ee_12_list[div_n-1]]) 
                    ax.set_ylabel('$W_{\mathrm{EE}}^{\mathrm{FF}}$', rotation=0, labelpad=-5)
                
            else:
                fig.colorbar(img)
   
plt.savefig('../figure/twopopuclear.pdf')
