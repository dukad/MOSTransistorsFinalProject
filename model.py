# Model of the MOS Transistor
# loads in reference data
# curve fits
# functions for plotting against the desired data

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit, minimize_scalar 


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

VDSID = 0
VGSID = 1
VSBID = 2
IDSID = 3


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
        self.W = Width
        self.L = Length
        self.vds = None
        self.ids = None
        self.vgs = None
        self.vsb = None
        # all other EKV Model parameters here (incomplete)
        self.Is = None
        self.Io = None
        self.Kappa = None
        self.Vt0 = None
        self.mu_0 = None
        self.theta = None
        self.alpha = None
        self.phi0 = None
        self.gamma = None
        self.VFB = None
        self.tox = 10.5e-9
        self.e_ox = 3.45e-11
        self.Ut = phiT
        self.cox = 3.45e-11 / 10.5e-9 # eox/toc
    # generic fitting function
    def fit_parameter(self):
        # 
        self.parameter = 0 # obviously edit this #
    
    def filter_data(self, datafile):
        '''Need to change this but a good starting point'''
        self.vds = datafile["VDS"].values
        self.ids = datafile["IDS"].values
        self.vgs = datafile["VGS"].values
        self.vsb = datafile["VSB"].values
    
    # kappa and Io extraction
    def extract_kappa_I0(self, vsb_val, window_size=7):
        '''This is created for finding all kappa and Io values for vsb value'''
        subset = self.idvg_data[self.idvg_data[:, VSBID] == vsb_val]
        subset = subset[np.argsort(subset[:, VGSID])]

        self.vgs = subset[:, VGSID]
        self.ids = subset[:, IDSID]
        ln_IDS = np.log(self.ids)
        best_r2 = -np.inf
        best_indices = None
        MIN_SLOPE = .5
        for i in range(len(self.vgs) - window_size):
            x_seg = self.vgs[i:i + window_size]
            y_seg = ln_IDS[i:i + window_size]
            slope, intercept, r_value, _, _ = linregress(x_seg, y_seg)
            if abs(slope) < MIN_SLOPE:
                continue

            if r_value**2 > best_r2:
                best_r2 = r_value**2
                best_indices = (i, i + window_size)

        i_start, i_end = best_indices
        x_lin = self.vgs[i_start:i_end]
        y_lin = ln_IDS[i_start:i_end]

        slope, intercept, r_value, _, _ = linregress(x_lin, y_lin)
        Kappa = slope * self.Ut
        Io = np.exp(intercept)
        return Kappa, Io
    
    def extract_all_kappas_IOs(self, plot=True):
        # Kappa should be fit for each curve
        kappas = []
        ios = []
        for vsb in (vsbs := np.unique(self.idvg_data[:, VSBID])):
            kappa, io = self.extract_kappa_I0(vsb)
            kappas.append(kappa)
            ios.append(io)
        if plot:
            plt.figure()
            plt.plot(vsbs, kappas)
            plt.title("K against VSB")
            plt.xlabel("VSB")
            plt.show()
            plt.figure()
            plt.plot(vsbs, ios)
            plt.title("I0 against VSB")
            plt.xlabel("VSB")
            plt.show()

        ## should change later
        self.Kappa = kappas[0]
        self.Is = ios[0]
        

    def fit_Is(self):
        """
        fitting Is and corresponding terms
        """
        self.u = 400
        self.Is = 2*(self.W/self.L)*self.u*Cox*self.Ut**2 / self.Kappa
        
    def fit_Vt(self,vsb=0, vds=0.1,plot=False):
        # load data from VGS sweeps where VSB = -
        mask = (self.idvg_data[:, VSBID] == vsb) & (self.idvg_data[:, VDSID] == 0.1)
        VGS = self.idvg_data[:, VGSID][mask]
        ID = self.idvg_data[:, IDSID][mask]
        # take data close to intercept
        # print(vsb)
        # print(ID)
        maxID = max(ID)
        minID = min(ID)
        diff = maxID - minID

        mask = (ID > 0.05*diff + minID) & (ID < 0.3*diff + minID)
        VGS_fit = VGS[mask]
        ID_fit = ID[mask]
        # linearize this line
        slope, intercept = np.polyfit(VGS_fit, ID_fit, 1)
        VGS_fit = np.linspace(0, 2.5, 100)
        ID_fit = slope * VGS_fit + intercept
        # print(f"slope: {slope}, intercept: {intercept}")
        # find index where ID = 0
        idx = np.where(ID_fit >= 0)[0][0]
        Vt = VGS_fit[idx]
        if plot:
            plt.figure()
            plt.title(f"Vt Extrapolation for VSB = {vsb}")
            plt.plot(VGS, ID, label="data")
            plt.axvline(Vt, label="Vt0", color='red')
            plt.plot()
            plt.plot(VGS_fit, ID_fit, label='fitted data')
            plt.legend()
            plt.grid()
            plt.show()
        return Vt

    def fit_Vts(self, plot=False):
        # for now
        # loop through VSBs, at one given VDS (max VDS, arbitrary)
        vds = 0.1
        mask_vds = (self.idvg_data[:, VDSID] == vds)
        masked_array = self.idvg_data[mask_vds] # now only values at a given vds
        # print(masked_array)
        vsbs = np.unique(masked_array[:, VSBID])
        # print(f"VSBS: {list(vsbs)}")
        Vts = []
        for vsb in vsbs:
            Vts.append(self.fit_Vt(vsb=vsb, vds=vds))
        if plot:
            # show the Vts over Vsb
            plt.figure()
            plt.plot(vsbs, Vts, label="Vts")
            plt.title("Vt over Vsb")
            plt.xlabel("Vsb (V)")
            plt.ylabel("Vt (V)")
            plt.legend()
            plt.show()

        # now fit the variables to the Vt function
        # guess
        phi0 = 2*phiF + 5*phiT
        print(f"Initial phi0 guess: {phi0} V")
        # phi0 =
        X_array = np.sqrt(phi0 + vsbs)
        eps = 1e-12
        phi0_min = -np.min(vsbs) + eps # ensure positive
        # create function to minimize
        def sse_for_phi0(phi0):
            if np.any(phi0 + vsbs <= 0):    
                return np.inf
            x = np.sqrt(phi0 + vsbs)
            X = np.column_stack([np.ones_like(x), x])
            beta, *_ = np.linalg.lstsq(X, Vts, rcond=None)
            resid = Vts - X @ beta
            return np.sum(resid**2)
        # use scipy minimize scalar to minimize the sum of squares error function
        res = minimize_scalar(sse_for_phi0, bounds=(phi0_min, phi0_min + 1e6))
        phi0_opt = res.x
        self.phi0 = phi0_opt
        # print(f"phi0* = {phi0_opt:.6g} V")
        X_array = np.sqrt(phi0_opt + vsbs)
        
            
        # find the slope of the line to get gamma
        slope, intercept = np.polyfit(X_array, Vts, 1)
        self.gamma = slope
        # print(f"gamma: {gamma:.6g} V^0.5")
        if plot:
            plt.figure()
            plt.plot(X_array, Vts, label = "VT Data")
            plt.title("VTs vs sqrt(phi0 + VSBs)")
            plt.xlabel("sqrt(phi0 + VSBs) (V^0.5)")
            plt.ylabel("VTs (V)")
            plt.grid()
            plt.plot(X_array, intercept + slope * X_array, label=f"Fit: gamma = {self.gamma:.6g}")
            plt.legend()
            plt.show()
            
        self.VFB = np.average(Vts - self.phi0 - self.gamma*X_array)
        if plot:
            plt.figure()
            plt.plot(vsbs, Vts, label="Vts")
            plt.plot(vsbs, self.get_Vt(vsbs), label='fit')
            plt.show()
            
        # print(f"VFB: {self.VFB:.6g} V")

    def get_Vt(self, Vsb):
        """
        Get the Vt based off of fit parameters
        """
        return self.VFB + self.phi0 + self.gamma*np.sqrt(self.phi0 + Vsb)


    def fit_all(self):
        """
        Method to fit all parameters in order.
        """
        # generate kappas for each unique VSB
        self.fit_Vts()
        self.extract_all_kappas_IOs() # this creates self.kappas
        # self.fit_Is()
        


    def model(self, VGB, VSB, VDB):
        """
        Runs the model. Uses fit values and EKV formula to return a drain current based on input voltages
        """
        if self.Kappa == None:
            raise ValueError("Fit Kappa before running model")
        if self.Is == None:
            raise ValueError("Fit Is before running model")
        # if self.Vt0 == None:
        #     raise ValueError("Fit Vt0 before runnign model")
        
        # forward current
        vt = self.get_Vt(VSB)
        IF = self.Is * np.log1p(1 + np.exp((self.Kappa*(VGB - vt) - VSB)/(2*self.Ut)))**2
        # reverse current
        IR = self.Is * np.log1p(1 + np.exp((self.Kappa*(VGB - vt) - VDB)/(2*self.Ut)))**2
        # sum
        ID = IF - IR
        # print(f"current {ID}")
        return ID
    
    def plot(self):
        """
        Plots model data against reference data
        """

        ############## PLOTTING ID VDS ###################
        unique_vgss = np.unique(self.idvd_data[:, VGSID])
        unique_vsbs = np.unique(self.idvd_data[:, VSBID])
        vdsmax = np.max(self.idvd_data[:, VDSID])
        vdmin = np.min(self.idvd_data[:, VDSID])
        vds_array = np.linspace(vdmin, vdsmax, 1000)
        
        fig, axs = plt.subplots(2, len(unique_vsbs), figsize=(15, 8))
        
        for i, vsb in enumerate(unique_vsbs):
            vdb_array = vds_array + vsb
            mask_vsb = self.idvd_data[:, VSBID] == vsb
            for vgs in unique_vgss:
                mask_vgs = self.idvd_data[:, VGSID] == vgs
                mask = mask_vgs & mask_vsb
                ##### plot reference data
                axs[0, i].plot(
                    self.idvd_data[mask][:, VDSID],
                    self.idvd_data[mask][:, IDSID],
                    label=f"Ref VGS: {vgs}",
                    linestyle = '--'
                )
                ###### plot model data
                axs[0, i].plot(
                    vds_array,
                    self.model(vgs + vsb, vsb, vdb_array),
                    label=f"VGS: {vgs}"
                )
            axs[0, i].legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0,
            )
            axs[0, i].set_title(f"IDS / VDS Curves for VSB = {vsb}")

        ############# PLOTTING ID VGS ###################
        unique_vdss = np.unique(self.idvg_data[:, VDSID])
        unique_vsbs = np.unique(self.idvg_data[:, VSBID])
        vgsmax = np.max(self.idvg_data[:, VGSID])
        vgsmin = np.min(self.idvg_data[:, VGSID])
        vgs_array = np.linspace(vgsmin, vgsmax, 1000)

        for i, vds in enumerate(unique_vdss):
            mask_vds = self.idvg_data[:, VDSID] == vds
            for vsb in unique_vsbs:
                vgb_array = vgs_array + vsb
                mask_vsb = self.idvg_data[:, VSBID] == vsb
                mask = mask_vds & mask_vsb

                axs[1, i].plot(
                    self.idvg_data[mask][:, VGSID],
                    self.idvg_data[mask][:, IDSID],
                    label=f"Ref VDS: {vds}",
                    linestyle = '--'
                )
                ### model data ######
                axs[1, i].plot(
                    vgs_array,
                    self.model(vgb_array, vsb, vds + vsb),
                    label=f"VSB: {vsb}"
                )
            axs[1, i].legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0,
            )
            axs[1, i].set_title(f"IDS / VGS Curves for VDS = {vds}")

        for ax in axs[0, :]:
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0,
            )
        for ax in axs[1, :]:
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0,
                )

        plt.subplots_adjust(wspace=0.5, right=0.9)
        plt.show()


