# Model of the MOS Transistor
# loads in reference data
# curve fits
# functions for plotting against the desired data

import numpy as np


# Always true values
Kn = 450e-6
Na = 7e17 * 1e6 # cm^-3 to m^-3
tox = 10.5e-9
ni = 1e10 * 1e6 # cm^-3
q = 1.602e-19 # C
Es = 11.7 * 8.854e-12 # F/m
Eox = 3.9 * 8.854e-12 # F/m
Cox = Eox/tox # F/m^2
k = 1.38e-23 # J/K
T = 300 # K
phiT = k*T/q
phiF = phiT * np.log(Na/ni)



class Model:
    def __init__(self, Width, Length):
        self.Width = Width
        self.Length = Length
        # all other EKV Model parameters here (incomplete)
        self.ueff = None
        self.theta = None
        # ...

    # generic fitting function
    def fit_parameter(self):
        # 
        self.parameter = 0 # obviously edit this #