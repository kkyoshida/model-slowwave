import time
import numpy as np
import models
import params

t1 = time.time() 

np.random.seed(2)

div_n = 101
combination_n = 1
result = np.zeros((combination_n, div_n, div_n))
total_t_set = 100000

w_ee_list = np.linspace(3.5, 4.0, div_n)
I_e_list = np.linspace(-0.2, 0.2, div_n)
beta_adapt_list = np.linspace(0.50, 1.00, div_n)
alpha_list = np.linspace(0.95, 1.05, div_n)

for i in range(div_n):
    for j in range(div_n):
        for combination in range(combination_n):
            if combination == 0:
                sw = models.LocalEI(w_ee=w_ee_list[j] * alpha_list[div_n - i - 1], I_e=I_e_list[i], beta_adapt=beta_adapt_list[div_n - i - 1], total_t=total_t_set)

            sw.simulate()
            result[combination, i, j] = sw.judgeslowwave(sw.r_e)
                
np.save('../data/twodbifurcationach', result)

t2 = time.time()
elapsed_time = t2-t1
print(f"Total time：{elapsed_time}")
