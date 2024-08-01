import numpy as np

class LocalEI:
    """
    The excitatory-inhibitory network model. 
    """

    def __init__(self, w_ee=3.75, w_ei=1.0, w_ie=1.25, w_ii=0.1, beta_adapt=0.75, total_t=100000, noise_strength_e=0.25, noise_strength_i=0.0, I_e=0, I_i=0):
        self.total_t = total_t # msec
        self.r_e = 1.0 * np.ones(self.total_t)
        self.r_i = 1.0 * np.ones(self.total_t)
        self.tau_e = 10.0 # msec
        self.tau_i = 5.0 # msec
        self.w_ee = w_ee
        self.w_ei = w_ei
        self.w_ie = w_ie
        self.w_ii = w_ii
        self.beta_adapt = beta_adapt
        self.noise_strength_e = noise_strength_e
        self.noise_strength_i = noise_strength_i
        self.dt = 0.1 # msec, timestep in solving differential equations
        self.I_e = I_e
        self.I_i = I_i
        self.tau_ou = 1.0 # msec
        self.judge = 1 
        
    def set_params(self):
        self.adapt = np.zeros(self.total_t)
        self.tau_adapt = 200.0 # msec
        
        # Ornstein–Uhlenbeck process
        ans_e = np.zeros(self.total_t+1)
        ans_i = np.zeros(self.total_t+1)
        noise_e = self.noise_strength_e * np.random.normal(loc=0.0, scale=1.0, size=self.total_t) 
        noise_i = self.noise_strength_i * np.random.normal(loc=0.0, scale=1.0, size=self.total_t) 
        for t in range(self.total_t):
            ans_e[t+1] = ans_e[t] + self.dt / self.tau_ou * (-ans_e[t]) + np.sqrt(self.dt) * noise_e[t]    
            ans_i[t+1] = ans_i[t] + self.dt / self.tau_ou * (-ans_i[t]) + np.sqrt(self.dt) * noise_i[t]    
        self.I_ext_e = self.I_e + ans_e[1:(self.total_t+1)]
        self.I_ext_i = self.I_i + ans_i[1:(self.total_t+1)]
        
        self.g_e = 1.0
        self.g_i = 4.0
        self.thre_e = -0.1
        self.thre_i = 5.0 
        
    def g(self, x, g, thre):
        return np.maximum(g * x - thre, 0) 
        
    def simulate(self):
        self.set_params()
        for t in range(self.total_t-1):
            self.r_e[t+1] = self.r_e[t] + self.dt * 1/self.tau_e * (- self.r_e[t] + self.g(self.w_ee * self.r_e[t] - self.w_ei * self.r_i[t] - self.adapt[t] + self.I_ext_e[t], self.g_e, self.thre_e))
            self.r_i[t+1] = self.r_i[t] + self.dt * 1/self.tau_i * (- self.r_i[t] + self.g(self.w_ie * self.r_e[t] - self.w_ii * self.r_i[t] + self.I_ext_i[t], self.g_i, self.thre_i))
            self.adapt[t+1] = self.adapt[t] + self.dt * 1/self.tau_adapt * (- self.adapt[t] + self.beta_adapt * self.r_e[t])
            if self.r_e[t+1] >= 100:
                self.judge = 0
                break
    
    def judgeslowwave(self, r_e_timeseries): #classify up-down, up-only, and the others
        if self.judge == 0:
            return 0 #others
        else:
            if np.any(r_e_timeseries[50000 : self.total_t] > 0.5):
                if np.any(r_e_timeseries[50000 : self.total_t] < 0.1):
                    return 1 #up-down
                else:
                    return -1 #up-only
            else:
                return 0 #others

class TwopopuEI(LocalEI):
    """
    The two-population model. 
    """

    def __init__(self, w_ee, w_ee_2, w_ee_12, w_ee_21, w_ie_12, w_ie_21, w_ii=0.1, w_ii_2=0.1, total_t=100000, I_e=0, I_e_2=0, beta_adapt=0.75, beta_adapt_2=0.75): 
        super().__init__(w_ee=w_ee, w_ii=w_ii, total_t=total_t, I_e=I_e, beta_adapt=beta_adapt)
        self.w_ee_12 = w_ee_12 #synaptic weight from excitatory population 2 to excitatory population 1
        self.w_ee_21 = w_ee_21
        self.w_ie_12 = w_ie_12
        self.w_ie_21 = w_ie_21
        self.w_ee_2 = w_ee_2
        self.w_ei_2 = self.w_ei
        self.w_ie_2 = self.w_ie
        self.w_ii_2 = w_ii_2
        self.r_e_2 = 1.0 * np.ones(self.total_t)
        self.r_i_2 = 1.0 * np.ones(self.total_t)
        self.I_e_2 = I_e_2
        self.I_i_2 = self.I_i
        self.beta_adapt_2 = beta_adapt_2
        
    def set_params(self):
        super().set_params()
        self.adapt_2 = np.zeros(self.total_t)

        # Ornstein–Uhlenbeck process
        ans_e = np.zeros(self.total_t+1)
        ans_i = np.zeros(self.total_t+1)
        noise_e = self.noise_strength_e * np.random.normal(loc=0.0, scale=1.0, size=self.total_t) 
        noise_i = self.noise_strength_i * np.random.normal(loc=0.0, scale=1.0, size=self.total_t) 
        for t in range(self.total_t):
            ans_e[t+1] = ans_e[t] + self.dt / self.tau_ou * (-ans_e[t]) + np.sqrt(self.dt) * noise_e[t]    
            ans_i[t+1] = ans_i[t] + self.dt / self.tau_ou * (-ans_i[t]) + np.sqrt(self.dt) * noise_i[t]    
        self.I_ext_e_2 = self.I_e_2 + ans_e[1:(self.total_t+1)]
        self.I_ext_i_2 = self.I_i_2 + ans_i[1:(self.total_t+1)]
        
    def simulate(self):
        self.set_params()
        for t in range(self.total_t-1):
            self.r_e[t+1] = self.r_e[t] + self.dt * 1/self.tau_e * (- self.r_e[t] + self.g(self.w_ee * self.r_e[t] - self.w_ei * self.r_i[t] + self.w_ee_12 * self.r_e_2[t] - self.adapt[t] + self.I_ext_e[t], self.g_e, self.thre_e))
            self.r_i[t+1] = self.r_i[t] + self.dt * 1/self.tau_i * (- self.r_i[t] + self.g(self.w_ie * self.r_e[t] - self.w_ii * self.r_i[t] + self.w_ie_12 * self.r_e_2[t] + self.I_ext_i[t], self.g_i, self.thre_i))
            self.adapt[t+1] = self.adapt[t] + self.dt * 1/self.tau_adapt * (- self.adapt[t] + self.beta_adapt * self.r_e[t])
            self.r_e_2[t+1] = self.r_e_2[t] + self.dt * 1/self.tau_e * (- self.r_e_2[t] + self.g(self.w_ee_2 * self.r_e_2[t] - self.w_ei_2 * self.r_i_2[t] + self.w_ee_21 * self.r_e[t] - self.adapt_2[t] + self.I_ext_e_2[t], self.g_e, self.thre_e))
            self.r_i_2[t+1] = self.r_i_2[t] + self.dt * 1/self.tau_i * (- self.r_i_2[t] + self.g(self.w_ie_2 * self.r_e_2[t] - self.w_ii_2 * self.r_i_2[t] + self.w_ie_21 * self.r_e[t] + self.I_ext_i_2[t], self.g_i, self.thre_i))
            self.adapt_2[t+1] = self.adapt_2[t] + self.dt * 1/self.tau_adapt * (- self.adapt_2[t] + self.beta_adapt_2 * self.r_e_2[t])
            if self.r_e[t+1]>=100 or self.r_e_2[t+1]>=100:
                self.judge = 0
                break
