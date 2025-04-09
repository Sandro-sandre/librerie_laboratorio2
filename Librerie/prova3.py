import analisi as lib
import numpy as np
import matplotlib.pyplot as plt

def VR_sovrasmorzato(t,V0,gamma,beta):
    return V0 * np.exp(-gamma*t) * (np.exp(beta*t) - np.exp(-beta*t))

def calcola_errore_R (R) :
  if (R <= 600):
    sigma = 0.009*R + 2*0.1
    b = R + sigma
    a = R - sigma
    sigma_gaus = ((b-a))/np.sqrt(12)
    return sigma_gaus
  elif (R>600 and R <= 60000):
    sigma = 0.009*R + 2
    b = R + sigma
    a = R - sigma
    sigma_gaus = ((b-a))/np.sqrt(12)
    return sigma_gaus
  elif ( R > 60000 and R >= 600000):
    sigma = 0.009*R + 10
    b = R + sigma
    a = R - sigma
    sigma_gaus = ((b-a))/np.sqrt(12)
    return sigma_gaus
  else :
    sigma = 0.009*R + 100
    b = R + sigma
    a = R - sigma
    sigma_gaus = ((b-a))/np.sqrt(12)
    return sigma_gaus


R_sovra_smorz = 11850
sigma_R_sovra_smorz = calcola_errore_R(R_sovra_smorz)
t_sovra_smorz = np.array([2.80, 3.50, 3.70, 4.34, 4.56, 4.76, 4.96, 5.36, 5.72, 6.36, 7.04, 8.08, 9.28, 11.1, 12.9, 14.0, 15.3, 17.1, 20.3, 22.4, 23.7, 22.6, 29.7, 34.4, 40.1, 47.6, 54.3, 62.0, 76.0, 85.0, 91.0, 100.0, 107.0, 118.0, 128.0, 134.0, 158.0, 182.0, 199.0, 221.0, 261.0, 293.0, 334.0, 410.0, 476.0, 561.0 ]) #micros
ddp_sovra_smorz = np.array([1.48, 1.76, 1.84, 2.08, 2.12, 2.22, 2.28, 2.40, 2.70, 2.64, 2.80, 2.96, 3.16, 3.30, 3.42, 3.46, 3.52, 3.56, 3.56, 3.52, 3.50, 3.44, 3.32, 3.20, 3.06, 2.82, 2.66, 2.48, 2.18, 1.98, 1.88, 1.76, 1.62, 1.48, 1.34, 1.26, 1.12, 0.800, 0.680, 0.560, 0.360, 0.300, 0.180, 0.080, 0.040, 0.000])
sigma_ddp_sovra_smorz = (2*np.ones(len(ddp_sovra_smorz))*0.05)/np.sqrt(12) # era 0.25


sigmaX = np.zeros(len(t_sovra_smorz))
t_sovra_smorz = t_sovra_smorz -2.80
t_sovra_smorz = t_sovra_smorz*10**(-6)

result = lib.fit(t_sovra_smorz, ddp_sovra_smorz, VR_sovrasmorzato, yerr=sigma_ddp_sovra_smorz, p0=[1,1,1], parameter_names=['V0', 'gamma', 'beta'])
lib.plot_fit(t_sovra_smorz, ddp_sovra_smorz, yerr=sigma_ddp_sovra_smorz, func = VR_sovrasmorzato,  p0=[1,1,1], parameter_names=['V0', 'gamma', 'beta'], confidence_intervals=True,error_band=1, prediction_band = True)
plt.show()