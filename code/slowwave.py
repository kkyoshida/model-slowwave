import time
import numpy as np
import models
import params

t1 = time.time() 

np.random.seed(2)

div_n = 101
combination_n = 10
result = np.zeros((combination_n, div_n, div_n))
total_t_set = 100000

# parameters
para = params.Para(div_n)
beta_adapt_list, w_ee_list, w_ee_list_wide, w_ei_list, w_ie_list, w_ii_list, I_e_list, I_i_list, noise_e_list, noise_i_list = para.giveparams()

for i in range(div_n):
    for j in range(div_n):
        for combination in range(combination_n):
            if combination == 0:
                sw = models.LocalEI(w_ee=w_ee_list_wide[i], beta_adapt=beta_adapt_list[j], total_t=total_t_set)
            elif combination == 1:
                sw = models.LocalEI(w_ei=w_ei_list[i], w_ee=w_ee_list[j], total_t=total_t_set)
            elif combination == 2:
                sw = models.LocalEI(w_ie=w_ie_list[i], w_ee=w_ee_list[j], total_t=total_t_set)
            elif combination == 3:
                sw = models.LocalEI(w_ii=w_ii_list[i], w_ee=w_ee_list[j], total_t=total_t_set)
            elif combination == 4:
                sw = models.LocalEI(I_e=I_e_list[i], w_ee=w_ee_list[j], total_t=total_t_set)
            elif combination == 5:
                sw = models.LocalEI(I_i=I_i_list[i], w_ee=w_ee_list[j], total_t=total_t_set)
            elif combination == 6:
                sw = models.LocalEI(noise_strength_e=noise_e_list[i], w_ee=w_ee_list[j], total_t=total_t_set)
            elif combination == 7:
                sw = models.LocalEI(noise_strength_i=noise_i_list[i], w_ee=w_ee_list[j], total_t=total_t_set)
            elif combination == 8:
                sw = models.LocalEI(w_ei=w_ei_list[i], w_ee=w_ee_list[j], total_t=total_t_set, beta_adapt=1.00)
            elif combination == 9:
                sw = models.LocalEI(w_ie=w_ie_list[i], w_ee=w_ee_list[j], total_t=total_t_set, beta_adapt=1.00)

            sw.simulate()
            result[combination, i, j] = sw.judgeslowwave(sw.r_e)
                
np.save('../data/twodbifurcation', result)

t2 = time.time()
elapsed_time = t2-t1
print(f"Total time：{elapsed_time}")
