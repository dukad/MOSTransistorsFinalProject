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
        self.tox = tox
        self.e_ox = Eox
        self.Ut = phiT
        self.cox = Cox # eox/toc
        self.vsat = 9e4 #6×10⁴ → 9×10⁴ m/s
        self.lambda_par = 0
        self.vsat = np.inf
        self.scale = 0
        self.l_vgs = 0
        self.l_vsb = 0
        self.l_vsb2 = 0
    
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
        

    def fit_Is(self, plot=False):
        """
        fitting Is and corresponding terms
        """
        def Idsn(ueff, VGS, VSB, Vds):
            vt = self.get_Vt(VSB, Vds)
            # IS prefactor: 2 * ueff * Cox * (W/L) * Ut^2 / Kappa
            IS = 2 * ueff * Cox * (self.W / self.L) * (self.Ut ** 2) / self.Kappa
            VDB = Vds + VSB
            VGB = VGS + VSB
            
            # softplus = ln(1 + exp(x)) implemented via log1p(exp(x))
            argF = ((self.Kappa * (VGB - vt- VSB)) ) / (2.0 * self.Ut)
            argR = ((self.Kappa * (VGB - vt- VDB)) ) / (2.0 * self.Ut)

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
        mask2 = (VGS > (filter := self.get_Vt(VSB, VDS)*1.5)) # only in nonsat strong inversion
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
        def ueff_from_theta_u0(Vgs, Vsb, u0, theta, Vds):
            
            Vt = self.get_Vt(Vsb, Vds)
            # print(Vgs - Vt)
            deltaV = (Vgs - Vt)
            # print(u0 / (1 + theta * deltaV))
            return u0 / (1 + theta * deltaV)
        
        func = lambda vgs, u0, theta: ueff_from_theta_u0(vgs, 0.0, u0, theta, VDS)
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
            plt.plot(VGS, ueff_from_theta_u0(VGS, VSB, u0_opt, theta_opt, VDS), label="fit")
            plt.xlabel('VGS')
            plt.ylabel('ueff')
            plt.title("Fitting U0 and theta")
            plt.legend()
            plt.show()

        vgsfilter = 3.5
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
        def ueff_from_thetaB(Vgs, Vsb, u0, theta, thetaB, Vds):
            return self.u0 / (1 + (self.theta * (Vgs - self.get_Vt(Vsb, Vds))) + (thetaB*Vsb))
        
        func = lambda Vsb, thetaB: ueff_from_thetaB(vgsfilter, Vsb, self.u0, self.theta, thetaB, vdsfilter)
        # print(len(vsbs), len(ueffs))
        # print(ueffs)
        popt, pcov = curve_fit(func, vsbs, ueffs, bounds=([-100], [100]))
        thetaB = popt[0]
        print(thetaB)
        print(f"thetaB*: {thetaB:.6g} V^-1")

        self.thetaB = thetaB 
        
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

        """
        Extract Vt using maximum transconductance method
        """
        mask = (self.idvg_data[:, VSBID] == vsb) & (self.idvg_data[:, VDSID] == vds)
        VGS = self.idvg_data[:, VGSID][mask]
        ID = self.idvg_data[:, IDSID][mask]
        
        # Sort by VGS
        sort_idx = np.argsort(VGS)
        VGS = VGS[sort_idx]
        ID = ID[sort_idx]
        
        # Calculate transconductance gm = dID/dVGS
        gm = np.gradient(ID, VGS)
        
        # Find index of maximum gm
        idx_max_gm = np.argmax(gm)
        
        # Use points around max gm for linear fit (±3 points)
        window = 3
        start_idx = max(0, idx_max_gm - window)
        end_idx = min(len(VGS), idx_max_gm + window + 1)
        
        VGS_fit = VGS[start_idx:end_idx]
        ID_fit = ID[start_idx:end_idx]
        
        # Linear fit
        slope, intercept = np.polyfit(VGS_fit, ID_fit, 1)
        
        # Extrapolate to ID = 0
        Vt = -intercept / slope
        
        if plot:
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(VGS, ID, 'b-', label='Data')
            plt.axvline(Vt, color='r', linestyle='--', label=f'Vt = {Vt:.3f}V')
            plt.axvline(VGS[idx_max_gm], color='g', linestyle='--', label='Max gm point')
            
            # Plot extrapolation line
            VGS_line = np.linspace(Vt, VGS_fit[-1], 100)
            ID_line = slope * VGS_line + intercept
            plt.plot(VGS_line, ID_line, 'r--', alpha=0.5, label='Extrapolation')
            plt.xlabel('VGS (V)')
            plt.ylabel('ID (A)')
            plt.title(f'Vt Extraction (VSB={vsb}V, VDS={vds}V)')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            plt.plot(VGS, gm, 'g-')
            plt.axvline(VGS[idx_max_gm], color='r', linestyle='--')
            plt.xlabel('VGS (V)')
            plt.ylabel('gm (S)')
            plt.title('Transconductance')
            plt.grid(True)
            plt.tight_layout()
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
        Vts = vts_mat[-1]
        if plot:
            plt.show()
        # check the scaling factor in between VTs
        dibls = []
        voltage_diff = Vdss[0] - Vdss[1]  # Should be positive (3.3 - 0.1 = 3.2)
        for vt_high_vds, vt_low_vds in zip(vts_mat[0], vts_mat[1]):
            # Vt typically decreases with increasing VDS
            # DIBL = -(Vt_high - Vt_low) / (VDS_high - VDS_low)
            dibl = -(vt_high_vds - vt_low_vds) / voltage_diff
            dibls.append(dibl)

        # self.dibl = -np.average(dibls)
        self.dibl = dibls[0]
        # self.dibl = 0
        self.vds_ref = Vdss[1]

        if plot:
            print("DIBLS: ", end='')
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
        X_array = np.sqrt(self.phi0 + vsbs)
        
            
        # find the slope of the line to get gamma
        slope, intercept = np.polyfit(X_array, Vts, 1)
        self.gamma = slope
        print(f"gamma: {self.gamma:.6g} V^0.5")
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
        
        VFBs = Vts - self.phi0 - self.gamma*X_array
        # plt.figure()
        # plt.plot(vsbs, VFBs)
        # plt.title("VFBs (checking consistent)")
        # plt.show()
        
        self.VFB = np.average(VFBs)
        if plot:
            plt.figure()
            plt.plot(vsbs, Vts, label="Vts", lw=5.0)
            plt.plot(vsbs, self.get_Vt(vsbs, Vdss[1]), label='fit')
            plt.xlabel("VSB (V)")
            plt.ylabel("VT (V)")
            plt.title("Fit VTs against reference VTs")
            # plt.figure()
            plt.legend()
            plt.show()
            
        # print(f"VFB: {self.VFB:.6g} V")

    def get_Vt(self, Vsb, Vds):
        """
        Get the Vt based off of fit parameters
        """
        # self.gamma = 0
        # Vt_base = self.VFB + self.phi0 + self.gamma*(np.sqrt(self.phi0 + Vsb))
        # Vt = Vt_base - self.dibl*(Vds - self.vds_ref)
        
        vt0 = self.VFB + self.phi0 + self.gamma*np.sqrt(self.phi0)
        Vt = vt0 + self.gamma*(np.sqrt(self.phi0 + Vsb) - np.sqrt(self.phi0))
        
        return Vt

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
        # print(self.gamma)
         # After fitting gamma in fit_Vts():
        self.extract_all_kappas_IOs() # this creates self.kappas
        self.fit_Is()
        for i in range(50):
            self.fit_Vsat()
            self.fit_lambdas()
            

        
        # self.fit_lambda()
        
        
    def get_ueff(self, Vgs, Vsb, Vds):
        return self.u0 / (1 + (self.theta * (Vgs- self.get_Vt(Vsb, Vds))) + (self.thetaB*Vsb))
    
    def lambda_model(self, VGB, VSB, VDB):
        Vgs = VGB - VSB
        Vds = VDB - VSB
        # self.vsat = 1e6
        vt = self.get_Vt(VSB, VDB - VSB)
        ueff = self.get_ueff(Vgs, VSB, VDB - VSB)   # pass Vgs rather than VGB+VSB
        E = Vds / self.L
        ueff = ueff / (1 + ueff*E / self.vsat)
        ueff = ueff * (1 + self.lambda_par*Vds)

        # IS prefactor: 2 * ueff * Cox * (W/L) * Ut^2 / Kappa
        IS = 2 * ueff * Cox * (self.W / self.L) * (self.Ut ** 2) / self.Kappa
        IS *= (1 + self.lambda_par*(Vds))

        # softplus = ln(1 + exp(x)) implemented via log1p(exp(x))
        argF = ((self.Kappa * (VGB - vt- VSB)) ) / (2.0 * self.Ut)
        argR = ((self.Kappa * (VGB - vt- VDB)) ) / (2.0 * self.Ut)

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

    def fit_Vsat(self):
        # fit Vsat
        # this is the last thing we need to fit, so we just need to find the vsat that minimizes total error in a curve
        vgs = 3.4
        vsb = 0
        mask = self.idvd_data[:, VGSID] == vgs
        mask2 = self.idvd_data[:, VSBID] == vsb
        mask = mask & mask2
        ID = self.idvd_data[:, IDSID][mask]
        VDS = self.idvd_data[:, VDSID][mask]
        print(ID)
        print(VDS)
        def fit_func(VDS, vsat):
            self.vsat = vsat
            id = self.model(vgs + vsb, vsb, VDS + vsb)
            return id
        
        popt, pcov = curve_fit(fit_func, VDS, ID, p0=[1e7])

        self.vsat = popt[0]
        print(self.vsat)
        
    def fit_lambda(self, vgs, vsb):
        mask = self.idvd_data[:, VGSID] == vgs
        mask2 = self.idvd_data[:, VSBID] == vsb
        mask = mask & mask2
        ID = self.idvd_data[:, IDSID][mask]
        VDS = self.idvd_data[:, VDSID][mask]

        def fit_func(VDS, lambda_par):
            self.lambda_par = lambda_par
            id = self.lambda_model(vgs + vsb, vsb, VDS + vsb)
            return id

        popt, pcov = curve_fit(fit_func, VDS, ID, p0=[0.1], bounds=[0, np.inf])

        self.lambda_par = popt[0]
        print(self.lambda_par)
        return self.lambda_par

    def fit_lambdas(self, plot=False):
        power = 2
        unique_vgss = np.unique(self.idvd_data[:, VGSID])
        unique_vsbs = np.unique(self.idvd_data[:, VSBID])
        vds_vals = self.idvd_data[:, VDSID]

        ls_results = []
        float_tol=1e-8
        if plot:
            plt.figure()
            plt.xlabel("VGS")
            plt.ylabel("lambda")
            plt.grid(True)

        for vsb in unique_vsbs:
            # use isclose for float comparisons
            mask_vsb = np.isclose(self.idvd_data[:, VSBID], vsb, atol=float_tol)
            vg_list = []
            lambda_list = []

            for vgs in unique_vgss:
                mask_vgs = np.isclose(self.idvd_data[:, VGSID], vgs, atol=float_tol)
                mask = mask_vgs & mask_vsb

                # check we have multiple VDS points for this (vgs, vsb) sweep
                if np.count_nonzero(mask) > 1:
                    try:
                        l_val = self.fit_lambda(vgs, vsb)
                    except Exception:
                        # skip if fit_lambda fails for this point
                        continue
                    if l_val is None:
                        continue
                    vg_list.append(vgs)
                    lambda_list.append(l_val)

            # store results for this VSB
            ls_results.append({"vsb": float(vsb), "vgs": np.array(vg_list), "lambda": np.array(lambda_list)})

            # plot if requested and there is data
            if plot and len(vg_list) > 0:
                plt.plot(vg_list, lambda_list, marker='o', label=f"VSB={vsb}")

        if plot:
            plt.legend()
            plt.show()

        def fit_func_no_vsb(VGS, l_vsb, scale):
            return scale / (1 + (VGS**power)*l_vsb)
        
        # curve fit
        print(vg_list)
        print(ls_results[0]['lambda'])
        popt, pcov = curve_fit(fit_func_no_vsb, vg_list, ls_results[0]['lambda'], p0 = [1000000, 1000000])

        l_vgs = popt[0]
        scale = popt[1]

        self.l_vgs = l_vgs
        self.scale = scale

        fit_vals = []
        for vg in vg_list:
            fit_vals.append(fit_func_no_vsb(vg, l_vgs, scale))

        if plot:
            plt.figure()
            # print(scale)
            # print(l_vgs)
            plt.plot(vg_list, ls_results[0]['lambda'], marker='o', label=f"VSB={0}")
            plt.plot(vg_list, fit_vals, label='fit')
            plt.legend()
            # plt.show()

        
        def fit_func_vsb(VGS, l_vsb, l_vsb2):
            VSB = 3.0
            return (scale* (1+l_vsb*VSB)) / (1 + (VGS**power)*l_vgs + l_vsb2*VSB)
    
        popt, pcov = curve_fit(fit_func_vsb, vg_list, ls_results[1]['lambda'])

        l_vsb = popt[0]
        l_vsb2 = popt[1]
        print(l_vsb)
        self.l_vsb = l_vsb
        self.l_vsb2 = l_vsb2
        fit_vals = []
        for vg in vg_list:
            fit_vals.append(fit_func_vsb(vg, l_vsb, l_vsb2))
        if plot:
            # plt.figure()
            print(scale)
            print(l_vgs)
            plt.plot(vg_list, ls_results[1]['lambda'], marker='o', label=f"VSB={3.0}")
            plt.plot(vg_list, fit_vals, label=f'fit vsb= {3.0}')
            plt.title("Measured lambdas and their fit")
            plt.xlabel("VGS")
            plt.ylabel("lambda")
            plt.legend()
            plt.show()
        

    def get_lambda(self, vgs, vsb):
        return (self.scale* (1+self.l_vsb*vsb)) / (1 + (vgs**2)*self.l_vgs + self.l_vsb2*vsb)


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
        # self.vsat = 1e6
        vt = self.get_Vt(VSB, VDB - VSB)
        ueff = self.get_ueff(Vgs, VSB, VDB - VSB)   # pass Vgs rather than VGB+VSB
        E = Vds / self.L
        ueff = ueff / (1 + ueff*E / self.vsat)
        lam = self.get_lambda(Vgs, VSB)
        ueff = ueff * (1 + lam*Vds)

        # IS prefactor: 2 * ueff * Cox * (W/L) * Ut^2 / Kappa
        IS = 2 * ueff * Cox * (self.W / self.L) * (self.Ut ** 2) / self.Kappa
        IS *= (1 + self.lambda_par*(Vds))

        # softplus = ln(1 + exp(x)) implemented via log1p(exp(x))
        argF = ((self.Kappa * (VGB - vt- VSB)) ) / (2.0 * self.Ut)
        argR = ((self.Kappa * (VGB - vt- VDB)) ) / (2.0 * self.Ut)

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
    
    def compute_error(self, I_ref, I_model, weights=None, eps=1e-12, rel_floor_factor=1e-8):
        I_ref = np.asarray(I_ref, dtype=float)
        I_model = np.asarray(I_model, dtype=float)
        if I_ref.shape != I_model.shape:
            raise ValueError("I_ref and I_model must have same shape")

        K = len(I_ref)
        if K == 0:
            return 0.0, 0.0

        if weights is None:
            weights = np.ones_like(I_ref, dtype=float)
        else:
            weights = np.asarray(weights, dtype=float)
            if weights.shape != I_ref.shape:
                raise ValueError("weights must match I_ref shape")

        # robust denominator: use max(|I_ref|, rel_floor, eps)
        rel_floor = np.maximum(np.max(np.abs(I_ref)) * rel_floor_factor, eps)
        denom = np.maximum(np.abs(I_ref), rel_floor)

        rel = (I_model - I_ref) / denom
        # replace any non-finite rel with large finite number (or drop those points)
        rel = np.where(np.isfinite(rel), rel, 0.0)   # alternative: raise or drop
        E_I = float(np.sum(weights * rel**2))
        E_rms = float(np.sqrt(E_I / float(K)))
        return E_rms
    
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
                            (self.model(vgs + vsb, vsb, vdb_array)),
                            label=f"VGS: {vgs}"
                        )
                        
                    if reference:
                        axs[0, i].plot(
                            self.idvd_data[mask][:, VDSID],
                            self.idvd_data[mask][:, IDSID],
                            label=f"Ref VGS: {vgs}",
                            linestyle = '--'
                        )

                    Vds_points = self.idvd_data[mask][:, VDSID]
                    Iref_points = self.idvd_data[mask][:, IDSID]
                    # pass Vds + vsb (body-referenced Vds) to the model
                    Imodel = self.model(vgs + vsb, vsb, Vds_points + vsb)
                    E_rms = self.compute_error(Iref_points, Imodel)
                    print(f"ID/VDS Error vgs={vgs}, vsb={vsb}: E_rms={E_rms:.6e}")
                    
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
                    # axs[1, i].axvline(self.get_Vt(vsb, 0))
                    if reference:
                        axs[1, i].plot(
                            self.idvg_data[mask][:, VGSID],
                            self.idvg_data[mask][:, IDSID],
                            label=f"Ref VDS: {vds}",
                            linestyle = '--'
                        )
                    
                    Vgs_points = self.idvg_data[mask][:, VGSID]
                    Iref_points = self.idvg_data[mask][:, IDSID]
                    # pass Vds + vsb (body-referenced Vds) to the model
                    Imodel = self.model(Vgs_points + vsb, vsb, vds + vsb)
                    E_rms = self.compute_error(Iref_points, Imodel)
                    print(f"ID/VGS Error vds={vds}, vsb={vsb}: E_rms={E_rms:.6e}")
                    
            axs[1, i].legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0,
            )
            axs[1, i].set_title(f"IDS / VGS Curves for VDS = {vds}")

        plt.subplots_adjust(wspace=0.5, right=0.9)
        plt.show()
    
    def plot_kappa(self, plot=False):

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

                    Vds_points = self.idvd_data[mask][:, VDSID]
                    Iref_points = np.gradient(self.idvd_data[mask][:, IDSID], Vds_points)
                    # pass Vds + vsb (body-referenced Vds) to the model
                    Imodel = self.model(vgs + vsb, vsb, Vds_points + vsb)
                    Imodel = np.gradient(Imodel, Vds_points)
                    E_rms = self.compute_error(Iref_points, Imodel)
                    print(f"dID/dVDS Error vgs={vgs}, vsb={vsb}: E_rms={E_rms:.6e}")

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
                    
                    Vgs_points = self.idvg_data[mask][:, VGSID]
                    Iref_points = np.gradient(self.idvg_data[mask][:, IDSID], Vgs_points)
                    # pass Vds + vsb (body-referenced Vds) to the model
                    Imodel = self.model(Vgs_points + vsb, vsb, vds + vsb)
                    Imodel = np.gradient(Imodel, Vgs_points)
                    E_rms = self.compute_error(Iref_points, Imodel)
                    print(f"dID/dVGS Error vds={vds}, vsb={vsb}: E_rms={E_rms:.6e}")

                    

            axs[1, i].legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0
            )
            axs[1, i].set_title(f"dIDS/dVGS for VDS = {vds}")

        plt.subplots_adjust(wspace=0.5, right=0.9)
        plt.show()

    def Gummel_Slope_Ratio(self):
        """
        test K1 as requested, plots ID/VDS / dID/dVDS
        """

        # for may VGS, find ID/VDS / dId / dVds
        VSB = 0
        VGS = np.linspace(0, 2.5, 1000)
        VDS = np.linspace(0, 1, 300)
        vds_target = 0.5*phiT
        index = np.argmin(np.abs(VDS - vds_target))
        vds_target = VDS[index]

        Srs = []
        for vgs in VGS:
            IDS = self.model(vgs + VSB, VSB, VDS + VSB)
            mask = (VDS== vds_target)
            ratio = IDS[mask] / vds_target
            dIddVds = np.gradient(IDS, VDS)
            derivative = dIddVds[mask]
            Sr = ratio / derivative
            Srs.append(Sr)
            # print(Sr)
        Srs = np.array(Srs)
        
        plt.figure()
        plt.plot(VGS, Srs)
        plt.title("Gummel Slope Ratio Test")
        plt.axhline(1, linestyle="--")
        plt.axhline(2*np.sqrt(np.e) - 2, linestyle='--')
        plt.show()

    def Gummel_symmetry_test(self):
        """
        Gummel symmetry test:
        Computes Ix, dIx/dVx, and d^2Ix/dVx^2 vs Vx to visualize symmetry.
        """
        vgbs = np.linspace(0, 3, 6)
        for vgb in vgbs:
            Vx_vals = np.linspace(-0.1, 0.1, 1000)
            Id_vals = self.model(vgb, -Vx_vals, Vx_vals)
            plt.plot(Vx_vals, Id_vals, label=f"VGB = {vgb}")
        plt.title("Gummel Symmetry Test a")
        plt.ylabel("Ix")
        plt.xlabel("Vx")
        plt.legend()
        plt.show()

        for vgb in vgbs:
            Vx_vals = np.linspace(-0.1, 0.1, 1000)
            Id_vals = self.model(vgb, -Vx_vals, Vx_vals)
            dId_vals = np.gradient(Id_vals, Vx_vals)
            plt.plot(Vx_vals, dId_vals, label=f"VGB = {vgb}")
        plt.title("Gummel Symmetry Test b")
        plt.ylabel("dIx/dVx")
        plt.xlabel("Vx")
        plt.legend()
        plt.show()

        for vgb in vgbs:
            Vx_vals = np.linspace(-0.1, 0.1, 1000)
            Id_vals = self.model(vgb, -Vx_vals, Vx_vals)
            dId_vals = np.gradient(Id_vals, Vx_vals)
            ddId_vals = np.gradient(dId_vals, Vx_vals)
            plt.plot(Vx_vals, ddId_vals, label=f"VGB = {vgb}")
        plt.title("Gummel Symmetry Test c")
        plt.ylabel("ddIx/ddVx")
        plt.xlabel("Vx")
        plt.legend()
        plt.show()



    def gO(self):
        """
        Compute and plot gm/ID vs IDS for a single VSB and VDS.
        """
        VSB= 0
        VDS_val = np.linspace(0, 3, 500)
        VGS_val = np.linspace(1, 4, 5)
        plt.figure(figsize=(8,6))
        for VGS in VGS_val:
            IDS = np.array([self.model(VGS+VSB, VSB, VDS+VSB) for VDS in VDS_val])
            gO = np.gradient(IDS,VDS_val)
            plt.plot(VDS_val, gO)
        plt.xlabel("VDS (V)")
        plt.ylabel("gO (mS)")
        plt.title(f"Output Conductance")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()



    def Conductance_tests2(self):
        VDS = 3
        VSB_val = np.linspace(0, 1.2, 6)
        VGS_val = np.linspace(.5, 6, 1000)

        for VSB in VSB_val:
            IDS = np.array(self.model(VGS_val, VSB, VDS))
            gm = np.gradient(IDS, VGS_val)
            print(gm)
            gm_over_ID = gm / (IDS + 1e-30)  # avoid divide by zero
            plt.semilogx(IDS, gm_over_ID, label=f'VSB={VSB:.2f} V')

        plt.xlabel("IDS (A)")
        plt.ylabel("gm / ID (1/V)")
        plt.title(f"Transconductance gm/ID vs IDS at VDS={VDS} V")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def Conductance_tests(self):

        VDS = 0.05
        VSB_vals = [0.0, 7,15]
        VGS_vals = np.linspace(0.0, 2.0, 1000)  # wider sweep into moderate inversion

        plt.figure()
        for VSB in VSB_vals:
            IDS = np.array(self.model(VGS_vals, VSB, VDS))
            gm = np.diff(IDS) / np.diff(VGS_vals)
            IDS_mid = (IDS[:-1] + IDS[1:]) / 2
            gm_over_ID = gm / (IDS_mid + 1e-30)

            # Plot only for IDS above a tiny floor for clarity
            mask = IDS_mid > 1e-12
            plt.semilogx(IDS_mid[mask], gm_over_ID[mask], label=f"VSB={VSB} V")

        plt.xlabel("IDS (A)")
        plt.ylabel("gm / ID (1/V)")
        plt.title("Weak Inversion Test: gm/ID vs IDS")
        plt.grid(True)
        plt.legend()
        plt.show()
        
