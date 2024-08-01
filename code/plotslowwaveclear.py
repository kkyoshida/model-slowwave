import numpy as np
import matplotlib.pyplot as plt
import params

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 0.75
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

result = np.load('../data/twodbifurcation.npy')

div_n = 101
combination_n = 10

# parameters
para = params.Para(div_n)
beta_adapt_list, w_ee_list, w_ee_list_wide, w_ei_list, w_ie_list, w_ii_list, I_e_list, I_i_list, noise_e_list, noise_i_list = para.giveparams()

fig = plt.figure(figsize = (16, 40))

for combination in range(combination_n):
    for i in range(2):
        if combination == 0 or (3 <= combination <= 7): 
            ax = fig.add_subplot(10, 4, 4*combination+i+1)
            img = ax.imshow(result[combination].T, interpolation='none', origin='lower', cmap='PiYG', vmin=-1, vmax=1)
            ax.set_xticks([0, div_n-1])
            ax.set_yticks([0, div_n-1])

            if combination ==0:
                ax.set_ylabel(r'$\beta$', rotation=0, labelpad=-5)
                ax.set_yticklabels([beta_adapt_list[0], beta_adapt_list[div_n-1]]) 
            else:
                ax.set_ylabel('$W_{\mathrm{EE}}$', rotation=0, labelpad=-5)
                ax.set_yticklabels([w_ee_list[0], w_ee_list[div_n-1]])

            if i == 1:
                fig.colorbar(img)
        
        labelpad_x = -10
        
        if combination == 0:
            ax.set_xlabel('$W_{\mathrm{EE}}$', labelpad=labelpad_x)
            ax.set_xticklabels([w_ee_list_wide[0], w_ee_list_wide[div_n-1]]) 
        elif combination == 3:
            ax.set_xlabel('$W_{\mathrm{II}}$', labelpad=labelpad_x)
            ax.set_xticklabels([w_ii_list[0], w_ii_list[div_n-1]]) 
        elif combination == 4:
            ax.set_xlabel('$I_{\mathrm{E}}$', labelpad=labelpad_x)
            ax.set_xticklabels([I_e_list[0], I_e_list[div_n-1]]) 
        elif combination == 5:
            ax.set_xlabel('$I_{\mathrm{I}}$', labelpad=labelpad_x)
            ax.set_xticklabels([I_i_list[0], I_i_list[div_n-1]]) 
        elif combination == 6:
            ax.set_xlabel('Excitatory noise strength', labelpad=labelpad_x)
            ax.set_xticklabels([noise_e_list[0], noise_e_list[div_n-1]]) 
        elif combination == 7:
            ax.set_xlabel('Inhibitory noise strength', labelpad=labelpad_x)
            ax.set_xticklabels([noise_i_list[0], noise_i_list[div_n-1]])  

        

plt.savefig('../figure/wee_2dclear.pdf')
