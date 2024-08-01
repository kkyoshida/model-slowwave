import time
import numpy as np
import models
import params

t1 = time.time() 

np.random.seed(2)

div_n = 101
combination_n = 4
result = np.zeros((combination_n, div_n, div_n))
total_t_set = 100000

# parameters
para = params.Para(div_n)
beta_adapt_list, w_ee_list, w_ee_list_wide, w_ei_list, w_ie_list, w_ii_list, I_e_list, I_i_list, noise_e_list, noise_i_list = para.giveparams()

w_ei_list = np.linspace(0.5, 2.0, div_n) 
w_ie_list = np.linspace(0.8, 2.0, div_n)

for i in range(div_n):
    for j in range(div_n):
        for combination in range(combination_n):
            if combination == 0:
                sw = models.LocalEI(w_ei=w_ei_list[i], w_ee=w_ee_list[j], total_t=total_t_set)
            elif combination == 1:
                sw = models.LocalEI(w_ie=w_ie_list[i], w_ee=w_ee_list[j], total_t=total_t_set)
            elif combination == 2:
                sw = models.LocalEI(w_ei=w_ei_list[i], w_ee=w_ee_list[j], total_t=total_t_set, beta_adapt=1.10)
            elif combination == 3:
                sw = models.LocalEI(w_ie=w_ie_list[i], w_ee=w_ee_list[j], total_t=total_t_set, beta_adapt=1.10)

            sw.simulate()
            result[combination, i, j] = sw.judgeslowwave(sw.r_e)
                
np.save('../data/twodbifurcationinhibitory', result)

t2 = time.time()
elapsed_time = t2-t1
print(f"Total time：{elapsed_time}")
