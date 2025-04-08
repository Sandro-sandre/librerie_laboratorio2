import analisi as lib
import numpy as np
import matplotlib.pyplot as plt

def VR_C_charge (t, tau, V_0):
  return V_0*np.exp(-t/tau)

R1 = 67800 # Ohm
sigma_R = 0.1 # kOhm

frequenza = 125.900 # Hz

V_g = 2.019 # Volt


t_RC = np.array([ 2.85, 2.88, 2.91, 2.97, 3.05, 3.10, 3.14,  3.25, 3.30, 3.34, 3.38, 3.41, 3.46, 3.49, 3.54,  3.60, 3.65, 3.71,  3.79,
                 3.83, 3.89, 3.93, 3.96,  4, 4.04, 4.08, 4.12, 4.20, 4.26, 4.32, 4.37,  4.40, 4.50, 4.59, 4.63, 4.82, 4.92, 4.97,  5.02,
                 5.07, 5.13, 5.20, 5.27, 5.32, 5.38, 5.50, 5.68, 5.82, 6.02, 6.10  ]) # ms
ddp_RC = np.array([3.94, 3.76, 3.58, 3.26, 2.90, 2.66, 2.56, 2.16,  2.02, 1.90, 1.80, 1.70, 1.52, 1.46, 1.36, 1.20, 1.14,  1.04,  0.940,
                   0.860, 0.800, 0.720, 0.700, 0.660, 0.620, 0.580, 0.540,  0.500, 0.460, 0.400, 0.380, 0.360, 0.320, 0.280, 0.240, 0.180,
                   0.180, 0.160, 0.140, 0.130, 0.120, 0.100, 0.100, 0.080, 0.080, 0.065,
                   0.060, 0.040, 0.020, 0 ]) # V

sigma_ddp_RC = (2*np.ones(len(ddp_RC))*0.05)/np.sqrt(12)

result = lib.fit(t_RC, ddp_RC, VR_C_charge, yerr=sigma_ddp_RC, p0=[1,1], parameter_names=['tau', 'V_0'])
print(result)
print(result['residuals'])

lib.plot_fit(t_RC, ddp_RC, yerr = sigma_ddp_RC, func = VR_C_charge, p0= [1,1], xlabel = 'tempo(s)',ylabel = 'volt', title= 'Fit carica RC' , residuals = True, parameter_names=['tau', 'V_0'], show_fit_params=True, show_chi_squared=True)