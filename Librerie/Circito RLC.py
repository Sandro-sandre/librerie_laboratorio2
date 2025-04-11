# Circito RLC 
# H_LC
import numpy as np
import analisi as lib 

Vin = 4
R_L = 39 
R = 990 

def mod_HCL_f_RLC(omega, C, L):
    d = (1 - omega**2 * L * C)**2 + omega**2 * C**2 * R_L**2
    numerator = np.sqrt(d)
    denominator = np.sqrt((1 - omega**2 * L * C)**2 + omega**2 * C**2 * (R + R_L)**2)
    return numerator / denominator

def arg_HCL_f_RLC(omega, C, L):
    term1 = np.arctan(omega * C * R_L / (1 - omega**2 * L * C))
    term2 = np.arctan(omega * C * (R + R_L) / (1 - omega**2 * L * C))
    return term1 - term2


frequenze_RLC_RL = np.array([  200, 500, 1000, 1300, 1600, 1800, 1950, 2050, 2150, 2250, 2350, 2400,  2450,  2690, 3000, 3300, 3600,  4600
 , 6100, 9300, 12300, 16500, 20100, 45100  ])

V_RLC_RL = np.array([ 4, 3.90, 3.24, 2.60, 1.92, 1.36, 1.20,             ])

sfase_ang_RLC_RL = np.array([  -4.32, -16.2,  -33.5,  -43.5,  -53, -59.6, -        ])

omega_RLC_RL = 2*np.pi*frequenze_RLC_RL

mod_HCL = V_RLC_RL / Vin

sigma = np.ones(len(frequenze_RLC_RL))

sigma_RLC_HCL =  (2*0.1/np.sqrt(12))*sigma


arg_HCL = sfase_ang_RLC_RL*2*np.pi
sigma_ang_RLC_RL = 2*0.5/np.sqrt(12)*sigma*np.pi/180
