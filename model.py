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



class EKV_Model:
    def __init__(self, idvg_data : np.array, idvd_data : np.array, Width : float, Length : float):
        """
        EKV Model Class
        Initialize with inputs:
        idvg data - VG sweep data of IV, np array
        idvd data - VD sweep data of IV, np array
        Width - float width in m
        Length - float length in m
        """
        self.idvg_data = idvg_data
        self.idvd_data = idvd_data
        self.Width = Width
        self.Length = Length
        # all other EKV Model parameters here (incomplete)
        self.Is = None
        self.Kappa = None
        self.Vt0 = None
        self.Ut = None

    # generic fitting function
    def fit_parameter(self):
        # 
        self.parameter = 0 # obviously edit this #

    def fit_all(self):
        """
        Method to fit all parameters in order.
        """
        raise NotImplementedError("fit_all is not complete yet")

    def model(self, VGB, VSB, VDB):
        """
        Runs the model. Uses fit values and EKV formula to return a drain current based on input voltages
        """
        if self.Kappa == None:
            raise ValueError("Fit Kappa before running model")
        if self.Is == None:
            raise ValueError("Fit Is before running model")
        if self.Vt0 == None:
            raise ValueError("Fit Vt0 before runnign model")
        if self.Ut == None:
            raise ValueError("Fit Ut before runnign model")
        
        # forward current
        IF = self.Is * np.log(1 + np.exp((self.Kappa*(VGB - self.Vt0) - VSB)/2*self.Ut))**2
        # reverse current
        IR = self.Is * np.log(1 + np.exp((self.Kappa*(VGB - self.Vt0) - VDB)/2*self.Ut))**2
        # sum
        ID = IF - IR
        return ID