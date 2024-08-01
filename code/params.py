import numpy as np

class Para:
    """
    Set parameter ranges for plotting figures.  
    """
    def __init__(self, div_n):
        self.beta_adapt_list = np.linspace(0.25, 1.25, div_n)
        self.w_ee_list_wide = np.linspace(1.5, 4.5, div_n)
        self.w_ee_list = np.linspace(3.5, 4.0, div_n)
        self.w_ei_list = np.linspace(0.5, 1.5, div_n)
        self.w_ie_list = np.linspace(0.8, 1.6, div_n)
        self.w_ii_list = np.linspace(0.05, 0.15, div_n)
        self.I_e_list = np.linspace(-0.2, 0.2, div_n)
        self.I_i_list = np.linspace(-0.4, 0.4, div_n)
        self.noise_e_list = np.linspace(0.0, 0.4, div_n)
        self.noise_i_list = np.linspace(0.0, 0.4, div_n)
        self.w_ee_12_list = np.linspace(0.0, 0.6, div_n)
        
    def giveparams(self):
        return self.beta_adapt_list, self.w_ee_list, self.w_ee_list_wide, self.w_ei_list, self.w_ie_list, self.w_ii_list, self.I_e_list, self.I_i_list, self.noise_e_list, self.noise_i_list
    
    def giveparams_twopopu(self):
        return self.w_ee_list, self.w_ee_12_list