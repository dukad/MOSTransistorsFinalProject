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
from scipy.optimize import curve_fit, minimize_scalar, brentq


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
    
    def extract_all_kappas_IOs(self, plot=False):
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
        self.kappa_array = kappas
        self.Kappa = np.average(kappas)
        # self.Is = ios[0]
        

    def fit_Is(self, plot =True):
        """
        fitting Is and corresponding terms
        """
        # Overall process -- use prefit K terms, fit Is terms
        # Using formula for Is, fit u0
        # using other data VSB, VDS != 0 , fit for theta terms
        if self.Kappa == None:
            raise ValueError("Define K first")
        # fit ID-VGS data, only in nonsaturation
        mask = (self.idvg_data[:, VDSID ] == 0.1) & (self.idvg_data[:, VSBID] == 0)
        VGS = self.idvg_data[:, VGSID][mask]
        ID = self.idvg_data[:, IDSID][mask]
        mask2 = (VGS > self.get_Vt(0) + 0.7) # only in nonsat
        VGS = VGS[mask2]
        ID = ID[mask2]


        # reverse calculate an array of IS values 
        def model_inverse(VGB, VSB, VDB, ID):
            vt = self.get_Vt(VSB)

            def L(x):
                return np.log(1 + np.exp((self.Kappa*(VGB - vt) - x) / (2*self.Ut)))

            LF = L(VSB)**2
            LR = L(VDB)**2
            denom = LF - LR

            # element-wise check
            near_zero = np.isclose(denom, 0.0)

            if np.any(near_zero):
                raise ValueError("Some denominator values are zero; IS undefined for those bias points.")

            IS = ID / denom
            return IS
        
        IS_array = model_inverse(VGS, 0, 0.1, ID)

        # directly convert this to an array of ueffs 
        ueffs = IS_array*self.Kappa / (2*(self.W/self.L)*Cox*self.Ut**2)

        # now lets curve fit ueffs to get u0 and theta
        def ueff_from_theta_u0(Vgs, Vsb, u0, theta):
            return u0 / (1 + theta * (Vgs - self.get_Vt(Vsb)))
        
        func = lambda vgs, u0, theta: ueff_from_theta_u0(vgs, 0.0, u0, theta)
        popt, pcov = curve_fit(func, VGS, ueffs, p0=[3e-4, 0.1])
        u0_opt, theta_opt = popt

        init_values = popt
        print(f"u0*: {u0_opt:.6g} m^2/Vs")
        print(f"theta*: {theta_opt:.6g} V^-1")

        if plot:
            plt.figure()
            plt.plot(VGS, ueffs, label="ueff data")
            plt.plot(VGS, ueff_from_theta_u0(VGS, 0.0, u0_opt, theta_opt), label="fit")
            plt.title("Fitting U0 and theta")
            plt.legend()
            plt.show()

        # ################################### determine alpha using ID-VDS data #################################


        def Vds_prime(Vgs, Vt, alpha):
            return (Vgs - Vt) / alpha

        def Idsn_withalpha(ueff, VGS, Vsb, Vds, alpha):
            # using linear region approximation to fit
            return (self.W/self.L) * ueff * Cox * (max((VGS - self.get_Vt(Vsb)), 0)*Vds - (alpha/2)*Vds**2)

        def get_alpha():
            
            # load ID-VDS data
            
            VGS = 3.4
            mask = (self.idvd_data[:, VGSID] == VGS) & (self.idvd_data[:, VSBID] == 0)
            VDS = self.idvd_data[:, VDSID]
            ID = self.idvd_data[:, VDSID]
            # we need to only use data in non-saturation, so lets mask the data
            # assume alpha = 2 so we dont curve fit improperly
            mask = (VDS < Vds_prime(VGS, self.get_Vt(0), 1)) & (VDS > 0.05)
            VDS_fit = VDS[mask]
            ID_fit = ID[mask]
            ueff = ueff_from_theta_u0(VGS, 0.0, u0_opt, theta_opt)

            # curve fit to get alpha
            def fit_func(vds, alpha):
                return Idsn_withalpha(ueff, VGS, 0.0, vds, alpha)

            popt, pcov = curve_fit(fit_func, VDS_fit, ID_fit, p0=[0.1], bounds=([0], [np.inf]))
            alpha_opt = popt[0]
            return alpha_opt

        self.alpha = get_alpha()
        print(f"alpha*: {self.alpha:.6g} V^-1")
        # maybe add something to plot alpha here to visually check

        ############################ Refine u0 and theta now using alpha ############################
        def get_ueff_withalpha(ID_meas, VGS, Vsb, Vds, alpha,
                        ueff_min=1e-20, ueff_max=1e4):
            def f(ueff):
                return Idsn_withalpha(ueff, VGS, Vsb, Vds, alpha) - ID_meas
            # Bracketed root find (robust)
            return brentq(f, ueff_min, ueff_max)

        def refine_u0_theta(u0, theta, Vsb=0):
            VGS = self.idvg_data[:, VGSID]
            ID = self.idvg_data[:, IDSID]
            # we need to only use data in non-saturation, so lets mask the data
            mask2 = (VGS > self.get_Vt(0) + 0.7) # only in nonsat
            VGS_fit = VGS[mask2]
            ID_fit = ID[mask2]
                
            ueffs = []
            for vgs, id in zip(VGS_fit, ID_fit):
                if isinstance(id, np.ndarray):
                    id = id[0]
                vds = min(0.1, Vds_prime(vgs, self.get_Vt(Vsb), self.alpha))
                vds = 0.1
                ueff = get_ueff_withalpha(id, vgs, Vsb, vds, self.alpha)
                ueffs = np.append(ueffs, ueff)

            # now curve fit ueffs to get u0 and theta
            func = lambda vgs, u0, theta: ueff_from_theta_u0(vgs, Vsb, u0, theta)
            # print(np.size(VGS_fit), np.size(ueffs))
            popt, pcov = curve_fit(func, VGS_fit, ueffs, p0=[u0, theta], bounds=([0, 0.005], [np.inf, np.inf]))
            u0_opt, theta_opt = popt
            return u0_opt, theta_opt

        # lets just run this optimization many times to improve values
        # for file in os.listdir(dir := "MOSdata/ID-VGS"):
        for i in range(10):
            u0_opt, theta_opt = refine_u0_theta(u0_opt, theta_opt)
            alpha_opt = get_alpha()
        self.alpha = alpha_opt
        self.u0 = u0_opt
        self.theta = theta_opt

        print(f"Refined u0*: {u0_opt:.6g} m^2/Vs")
        print(f"Refined theta*: {theta_opt:.6g} V^-1")

        print(f"Refined alpha : {alpha_opt}")


        vgsfilter = 3.6
        vdsfilter = 0.1 # lets keep this in linear region
        ############# thetaB calculations #######################
        vsbs = np.unique(self.idvg_data[:, VSBID])
        Ids = []
        # finding VSB depeendances, so iterate through unique VSBs
        for vsb in vsbs:
            mask = self.idvg_data[:, VSBID] == vsb
            # take ID @ VGS == 3.6
            VGS = self.idvg_data[:, VGSID][mask]
            ID = self.idvg_data[:, IDSID][mask]
            VDS = self.idvg_data[:, VDSID][mask]
            # take data when VGS = target 3.6
            mask2 = (VGS == vgsfilter) & (VDS == vdsfilter)
            Ids.append(ID[mask2])

        if plot:
            plt.figure()
            plt.plot(vsbs, Ids, label="Ids vs vsbs")
            plt.title("Drain current at vds and vgs target over vsb")
            plt.show()
            # plt.figure()

        # Get what the ueff should be based on this data
        ueffs = []
        for vsb, id in zip(vsbs, Ids):
            ueffs.append(get_ueff_withalpha(id, vgsfilter, vsb, vdsfilter, alpha_opt))

        # now lets curve fit ueffs to get u0 and theta
        def ueff_from_thetaB(Vgs, Vsb, u0, theta, thetaB):
            return u0 / (1 + theta * (Vgs - self.get_Vt(Vsb)) + thetaB*Vsb)

        func = lambda Vsb, thetaB: ueff_from_thetaB(vgsfilter, Vsb, self.u0, self.theta, thetaB)
        # print(len(vsbs), len(ueffs))
        popt, pcov = curve_fit(func, vsbs, ueffs, bounds=[-100, 100])
        thetaB = popt[0]
        print(thetaB)
        print(f"thetaB*: {thetaB:.6g} V^-1")

        self.thetaB = thetaB
        
        ## PLOT TO CHECK
        if plot:
            plt.figure()
            plt.title("theta B fit")
            plt.plot(vsbs, ueffs, label="scraped ueffs", linestyle='--')
            plt.plot(vsbs, func(vsbs, thetaB), label="ueffs from func")
            plt.legend()
            plt.show()
        
        # self.Is = 2*(self.W/self.L)*get(ue*Cox*self.Ut**2 / self.Kappa
        # self.Is = 5e-4

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
        self.fit_Is()
        
    def get_ueff(self, Vgs, Vsb):
        return self.u0 / (1 + self.theta * (Vgs - self.get_Vt(Vsb)) + self.thetaB*Vsb)

    def model(self, VGB, VSB, VDB):
        """
        Runs the model. Uses fit values and EKV formula to return a drain current based on input voltages
        """
        if self.Kappa == None:
            raise ValueError("Fit Kappa before running model")
        # if self.Vt0 == None:
        #     raise ValueError("Fit Vt0 before runnign model")
        # self.theta = 0
        # self.alpha = 0
        # self.thetaB = 0
        # forward current
        # now adapted to use EKV model
        # forward current
        # assume VGB = Vg - Vb, VSB = Vs - Vb, VDB = Vd - Vb
        self.thetaB = 0
        Vgs = VGB - VSB
        Vgd = VGB - VDB
        Vds = VDB + VSB

        vt = self.get_Vt(VSB)
        ueff = self.get_ueff(Vgs, VSB)   # pass Vgs rather than VGB+VSB

        # IS prefactor: 2 * ueff * Cox * (W/L) * Ut^2 / Kappa
        IS = 2 * ueff * Cox * (self.W / self.L) * (self.Ut ** 2) / self.Kappa
        IS *= (1 + self.alpha * (Vds))

        # softplus = ln(1 + exp(x)) implemented via log1p(exp(x))
        argF = (self.Kappa * (Vgs - vt)) / (2.0 * self.Ut)
        argR = (self.Kappa * (Vgd - vt)) / (2.0 * self.Ut)

        softF = np.log1p(1 + np.exp(argF))
        softR = np.log1p(1 + np.exp(argR))

        IF = IS * (softF ** 2)
        IR = IS * (softR ** 2)

        ID = IF - IR
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

        plt.subplots_adjust(wspace=0.5, right=0.9)
        plt.show()
    
    def plot_kappa(self, plot=True):

        # Extract arrays
        vgs = self.idvg_data[:, VGSID]
        ids = self.idvg_data[:, IDSID]
        vsb_arr = self.idvg_data[:, VSBID]

        kappas = []
        ios = []

        # Unique body-bias values
        vsbs = np.unique(vsb_arr)

        for vsb in vsbs:
            kappa, io = self.extract_kappa_I0(vsb)
            kappas.append(kappa)
            ios.append(io)

        if plot:
            plt.figure()

        for this_vsb in vsbs:
            # Select rows for this VSB
            mask = (vsb_arr == this_vsb)
            vgs_sel = vgs[mask]
            ids_sel = ids[mask]

            # Extract kappa and I0 for this VSB
            kappa, I0 = self.extract_kappa_I0(this_vsb)

            # Compute fitted line: I = I0 * exp(kappa * Vgs / Ut)
            fit_line = I0 * np.exp(kappa * vgs_sel / self.Ut)

            if plot:
                plt.semilogy(vgs_sel, ids_sel, '.', label=f"Measured VSB={this_vsb}")
                plt.semilogy(vgs_sel, fit_line, '-', label=f"Fit VSB={this_vsb}")

        if plot:
            plt.title("IDS vs VGS with Kappa Fit")
            plt.xlabel("VGS (V)")
            plt.ylabel("IDS (A)")
            plt.legend()
            plt.grid(True, which="both")
            plt.show()


