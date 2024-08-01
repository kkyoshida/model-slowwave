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
para = params.Para(div_n)
w_ee_list, w_ee_12_list = para.giveparams_twopopu()
w_ee_list_weak = np.linspace(1.5, 2.5, div_n)

result = np.load('../data/twopopubidirrevise2-1.npy')
result2 = np.load('../data/twopopubidirrevise2-2.npy')

fig = plt.figure(figsize = (8, 16))

for popu in range(2):
    for i in range(2):
        ax = fig.add_subplot(4, 2, 2*popu+i+1)
        if popu == 0:
            img = ax.imshow(result.T, interpolation='none', origin='lower', cmap='PiYG', vmin=-1, vmax=1)
        else:
            img = ax.imshow(result2.T, interpolation='none', origin='lower', cmap='PiYG', vmin=-1, vmax=1)
        
        if i==0:
            ax.set_xticks([0, div_n-1])
            ax.set_yticks([0, div_n-1])

            ax.set_xticklabels([w_ee_list[0], w_ee_list[div_n-1]]) 
            ax.set_xlabel('$W_{\mathrm{EE}}^1$', labelpad=-10)
            
            ax.set_yticklabels([w_ee_list[0], w_ee_list[div_n-1]]) 
            ax.set_ylabel('$W_{\mathrm{EE}}^2$', rotation=0, labelpad=-5)
            
        else:
            fig.colorbar(img)
   
plt.savefig('../figure/twopopuclearbidirrevise2.pdf')
