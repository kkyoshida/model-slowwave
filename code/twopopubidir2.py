import time
import numpy as np
import models
import params

t1 = time.time() 

np.random.seed(2)

div_n = 101
result = np.zeros((div_n, div_n))
result2 = np.zeros((div_n, div_n))

para = params.Para(div_n)
w_ee_list, w_ee_12_list = para.giveparams_twopopu()

total_t_set = 100000

for i in range(div_n):
    for j in range(div_n):
        sw = models.TwopopuEI(w_ee=w_ee_list[i], w_ee_2=w_ee_list[j], w_ee_12=0.1, w_ee_21=0.1, w_ie_12=0.0, w_ie_21=0.0, total_t=total_t_set) #bidirectional
        sw.simulate()
        result[i, j] = sw.judgeslowwave(sw.r_e)
        result2[i, j] = sw.judgeslowwave(sw.r_e_2)
            
np.save('../data/twopopubidirrevise2-1', result)
np.save('../data/twopopubidirrevise2-2', result2)

t2 = time.time()
elapsed_time = t2-t1
print(f"Total time：{elapsed_time}")
