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
from scipy.optimize import curve_fit, minimize_scalar, brentq, least_squares



# Use a built-in colormap with many distinct colors
colors = plt.cm.tab20.colors  # 20 distinct colors

# Set the color cycle for all subsequent plots
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

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
        self.vsat = 9e4 #6×10⁴ → 9×10⁴ m/s
        self.lambda_par = 0
    
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
        

    def fit_Is(self, plot=True):
        """
        fitting Is and corresponding terms
        """
        def Idsn(ueff, VGS, VSB, Vds):
            vt = self.get_Vt(VSB)
            # IS prefactor: 2 * ueff * Cox * (W/L) * Ut^2 / Kappa
            IS = 2 * ueff * Cox * (self.W / self.L) * (self.Ut ** 2) / self.Kappa
            VDB = Vds + VSB
            # softplus = ln(1 + exp(x)) implemented via log1p(exp(x))
            argF = (self.Kappa * (VGS + VSB - vt) - VSB) / (2.0 * self.Ut)
            argR = (self.Kappa * (VGS + VSB - vt) - (VDB)) / (2.0 * self.Ut)

            softF = np.log(1 + np.exp(argF))
            softR = np.log(1 + np.exp(argR))

            IF = IS * (softF ** 2)
            IR = IS * (softR ** 2)

            ID = IF - IR
            return ID
        
        def get_ueff(ID_meas, VGS, Vsb, Vds):
            def f(ueff):
                return Idsn(ueff, VGS, Vsb, Vds) - ID_meas
            # Bracketed root find (robust)
            res = least_squares(f, x0=0.25)
            # print(F"Difference = {f(res.x[0]) - ID_meas}")
            return res.x[0] 

        # Overall process -- use prefit K terms, fit Is terms
        # Using formula for Is, fit u0
        # using other data VSB, VDS != 0 , fit for theta terms
        if self.Kappa == None:
            raise ValueError("Define K first")
        # fit ID-VGS data, only in nonsaturation
        VDS = 0.1 # nonsat
        VSB = 0
        mask = (self.idvg_data[:, VDSID ] == VDS) & (self.idvg_data[:, VSBID] == VSB)
        VGS = self.idvg_data[:, VGSID][mask]
        ID = self.idvg_data[:, IDSID][mask]
        mask2 = (VGS > (filter := self.get_Vt(VSB)*1.5)) # only in nonsat strong inversion
        VGS = VGS[mask2]
        ID = ID[mask2]

        # directly convert this to an array of ueffs. We are assumign no channel length modulation or vsat effects here
        # ueffs = IS_array*self.Kappa / (2*(self.W/self.L)*Cox*(self.Ut**2))
        ueffs = []
        # print(VGS)
        for i in range(len(VGS)):
            ueff = get_ueff(ID[i], VGS[i], VSB, VDS)
            ueffs.append(ueff)
        ueffs = np.array(ueffs)

        if plot:
            plt.figure()
            plt.title("Checking ueffs array")
            plt.plot(VGS, Idsn(ueffs, VGS, VSB, VDS), label="ueff data")
            plt.plot(VGS, ID, label = "data")
            plt.legend()
            plt.show()


        # now lets curve fit ueffs to get u0 and theta
        def ueff_from_theta_u0(Vgs, Vsb, u0, theta):
            
            Vt = self.get_Vt(Vsb)
            # print(Vgs - Vt)
            deltaV = (Vgs - Vt)
            # print(u0 / (1 + theta * deltaV))
            return u0 / (1 + theta * deltaV)
        
        func = lambda vgs, u0, theta: ueff_from_theta_u0(vgs, 0.0, u0, theta)
        popt, pcov = curve_fit(func, VGS, ueffs, p0=[0.05, 0.4])
        u0_opt, theta_opt = popt
        init_values = popt
        print(f"u0*: {u0_opt:.6g} m^2/Vs")
        print(f"theta*: {theta_opt:.6g} V^-1")
        self.u0 = u0_opt
        self.theta = theta_opt
        self.thetaB = 0
        plot_vgs = self.idvg_data[:, VGSID][mask]
        if plot:
            plt.figure()
            plt.plot(plot_vgs, self.idvg_data[:, IDSID][mask])
            plt.plot(plot_vgs, self.model(plot_vgs, 0, 0.1))
            plt.title("Model based on fit u0 and theta")
            plt.xlabel("VGS")
            plt.ylabel("ID")
            plt.axvline(filter)
            plt.show()

        if plot:
            plt.figure()
            plt.plot(VGS, ueffs, label="ueff data")
            plt.plot(VGS, ueff_from_theta_u0(VGS, VSB, u0_opt, theta_opt), label="fit")
            plt.xlabel('VGS')
            plt.ylabel('ueff')
            plt.title("Fitting U0 and theta")
            plt.legend()
            plt.show()

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
            plt.ylabel("Ids")
            plt.xlabel("Vsb")
            plt.show()

        # Get what the ueff should be based on this data
        ueffs = []
        for vsb, id in zip(vsbs, Ids):
            ueffs.append(get_ueff(id, vgsfilter, vsb, vdsfilter))

        # now lets curve fit ueffs to get u0 and theta
        def ueff_from_thetaB(Vgs, Vsb, u0, theta, thetaB):
            return self.u0 / (1 + (self.theta * (Vgs - self.get_Vt(Vsb))) + (thetaB*Vsb))
        
        func = lambda Vsb, thetaB: ueff_from_thetaB(vgsfilter, Vsb, self.u0, self.theta, thetaB)
        # print(len(vsbs), len(ueffs))
        # print(ueffs)
        popt, pcov = curve_fit(func, vsbs, ueffs, bounds=([-100], [100]))
        thetaB = popt[0]
        print(thetaB)
        print(f"thetaB*: {thetaB:.6g} V^-1")

        self.thetaB = thetaB
        # self.thetaB=  0 #######################################
        
        ## PLOT TO CHECK
        if plot:
            plt.figure()
            plt.title("theta B fit")
            # plt.plot(vsbs, self.get_Vt(vsbs), label="VTs")
            plt.plot(vsbs, ueffs, label="scraped ueffs", linestyle='--')
            # self.thetaB = 0
            plt.plot(vsbs, func(vsbs, self.thetaB), label="ueffs from func")
            plt.legend()
            plt.show()
        
        # self.Is = 2*(self.W/self.L)*get(ue*Cox*self.Ut**2 / self.Kappa
        # self.Is = 5e-4

    def fit_Vt(self,vsb=0, vds=0.1,plot=False):
        # load data from VGS sweeps where VSB = -
        mask = (self.idvg_data[:, VSBID] == vsb) & (self.idvg_data[:, VDSID] == vds)
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
        Vt = -intercept / slope

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
        # vds = 0.
        Vdss = [3.3, 0.1]
        vts_mat = []
        if plot:
            plt.figure()
        for vds in Vdss:
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
                plt.plot(vsbs, Vts, label=f"Vts at VDS = {vds}")
                plt.title("Vt over Vsb")
                plt.xlabel("Vsb (V)")
                plt.ylabel("Vt (V)")
                plt.legend()
                
            vts_mat.append(Vts)
        if plot:
            plt.show()
        # check the scaling factor in between VTs
        dibls = []
        voltage_diff = abs(Vdss[0] - Vdss[1])
        # print(vts_mat)
        for vt1, vt2 in zip(vts_mat[0], vts_mat[1]):
            diff = vt1 - vt2
            dibls.append(diff / voltage_diff)

        if plot:
            print(dibls)
            # we have concluded there is no DIBL effect

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

    def VDSsat_EKV(self, Isat, Is, A=0.8):
        """
        EKV formula for VDSsat
        Isat : measured or model IDS at saturation
        Is : EKV scale current (self.Is)
        A : fitting constant
        """
        term1 = 2 * self.Ut * np.log(np.exp(np.sqrt(Isat / Is)) - 1)
        term2 = 2 * self.Ut * np.log(np.exp(np.sqrt(Isat / (A * Is))) - 1)
        return term1 - term2


    def fit_all(self):
        """
        Method to fit all parameters in order.
        """
        # generate kappas for each unique VSB
        self.fit_Vts()
        print(self.gamma)
        self.extract_all_kappas_IOs() # this creates self.kappas
        self.fit_Is()
        
    def get_ueff(self, Vgs, Vsb):
        return self.u0 / (1 + (self.theta * (Vgs - self.get_Vt(Vsb))) + (self.thetaB*Vsb))

    def model(self, VGB, VSB, VDB):
        """
        Runs the model. Uses fit values and EKV formula to return a drain current based on input voltages
        """
        if self.Kappa == None:
            raise ValueError("Fit Kappa before running model")
        # if self.Vt0 == None:
        #     raise ValueError("Fit Vt0 before runnign model")
        # assume VGB = Vg - Vb, VSB = Vs - Vb, VDB = Vd - Vb
        # self.thetaB = 0 # CHANGE THIS LATER
        Vgs = VGB - VSB
        Vds = VDB - VSB

        vt = self.get_Vt(VSB)
        ueff = self.get_ueff(Vgs, VSB)   # pass Vgs rather than VGB+VSB

        # IS prefactor: 2 * ueff * Cox * (W/L) * Ut^2 / Kappa
        IS = 2 * ueff * Cox * (self.W / self.L) * (self.Ut ** 2) / self.Kappa
        IS *= (1 + self.lambda_par*(Vds))

        # softplus = ln(1 + exp(x)) implemented via log1p(exp(x))
        argF = ((self.Kappa * (VGB - vt)) - VSB) / (2.0 * self.Ut)
        argR = ((self.Kappa * (VGB - vt)) - VDB) / (2.0 * self.Ut)

        softF = np.log(1 + np.exp(argF))
        softR = np.log(1 + np.exp(argR))

        IF = IS * (softF ** 2)
        IR = IS * (softR ** 2)

        Isat = np.max(IF)

        VDSsat = self.VDSsat_EKV(Isat, IS, A=0.8)
        IF_sat = IF / (1 + IF * Vds / VDSsat)
        IR_sat = IR / (1 + IR * Vds / VDSsat)
        I_EKV = IF_sat - IR_sat
        return I_EKV
    
    def plot(self, reference=True, model=True):
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
                if len(self.idvd_data[mask][:, IDSID]) > 0:
                    ###### plot model data
                    if model:
                        axs[0, i].plot(
                            vds_array,
                            self.model(vgs + vsb, vsb, vdb_array),
                            label=f"VGS: {vgs}"
                    )
                    if reference:
                        axs[0, i].plot(
                            self.idvd_data[mask][:, VDSID],
                            self.idvd_data[mask][:, IDSID],
                            label=f"Ref VGS: {vgs}",
                            linestyle = '--'
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
                if len(self.idvg_data[mask][:, IDSID]) > 0:
                    ### model data ######
                    if model:
                        axs[1, i].plot(
                            vgs_array,
                            self.model(vgb_array, vsb, vds + vsb),
                            label=f"VSB: {vsb}"
                        )
                    if reference:
                        axs[1, i].plot(
                            self.idvg_data[mask][:, VGSID],
                            self.idvg_data[mask][:, IDSID],
                            label=f"Ref VDS: {vds}",
                            linestyle = '--'
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


    def plot_derivative(self, reference=True, model=True):
        """
        Plots numerical derivatives of model/reference data
        """

        ############## dIDS/dVDS ###################
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

                data = self.idvd_data[mask]

                if len(data) > 1:
                    # model derivative
                    if model:
                        ids_model = self.model(vgs + vsb, vsb, vdb_array)
                        dydx_model = np.gradient(ids_model, vds_array)
                        axs[0, i].plot(
                            vds_array,
                            dydx_model,
                            label=f"VGS: {vgs}"
                        )

                    # reference derivative
                    if reference:
                        x = data[:, VDSID]
                        y = data[:, IDSID]
                        dydx = np.gradient(y, x)
                        axs[0, i].plot(
                            x,
                            dydx,
                            linestyle='--',
                            label=f"Ref VGS: {vgs}"
                        )
            axs[0, i].legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0
            )

            axs[0, i].set_title(f"dIDS/dVDS for VSB = {vsb}")

        ############## dIDS/dVGS ###################
        unique_vdss = np.unique(self.idvg_data[:, VDSID])
        unique_vsbs = np.unique(self.idvg_data[:, VSBID])
        vgsmax = np.max(self.idvg_data[:, VGSID])
        vgsmin = np.min(self.idvg_data[:, VGSID])
        vgs_array = np.linspace(vgsmin, vgsmax, 1000)

        for i, vds in enumerate(unique_vdss):
            mask_vds = self.idvg_data[:, VDSID] == vds

            for vsb in unique_vsbs:
                mask_vsb = self.idvg_data[:, VSBID] == vsb
                mask = mask_vds & mask_vsb

                data = self.idvg_data[mask]

                if len(data) > 1:
                    # model derivative
                    if model:
                        vgb_array = vgs_array + vsb
                        ids_model = self.model(vgb_array, vsb, vds + vsb)
                        dydx_model = np.gradient(ids_model, vgs_array)
                        axs[1, i].plot(
                            vgs_array,
                            dydx_model,
                            label=f"VSB: {vsb}"
                        )

                    # reference derivative
                    if reference:
                        x = data[:, VGSID]
                        y = data[:, IDSID]
                        dydx = np.gradient(y, x)
                        axs[1, i].plot(
                            x,
                            dydx,
                            linestyle='--',
                            label=f"Ref VSB: {vsb}"
                        )

                    

            axs[1, i].legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0
            )
            axs[1, i].set_title(f"dIDS/dVGS for VDS = {vds}")

        plt.subplots_adjust(wspace=0.5, right=0.9)
        plt.show()
