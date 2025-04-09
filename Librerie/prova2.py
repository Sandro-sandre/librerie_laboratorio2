import analisi as lib
import numpy as np
import matplotlib.pyplot as plt
def calcola_errore_V (V) :
    sigma = 0.005*V + 2*0.1*10**(-3)
    b = V + sigma
    a = V - sigma
    sigma_gaus = ((b-a))/np.sqrt(12)
    return sigma_gaus

def calcola_errore_V_v2 (V) :
    sigma = 0.05*V + 2*0.1*10**(-3)
    b = V + sigma
    a = V - sigma
    sigma_gaus = ((b-a))/np.sqrt(12)
    return sigma_gaus

R1 = 32.8 # Ohm R1 = R2

Vin = 5.357 # V
sigma_Vin = calcola_errore_V(Vin)

Vout = np.array([ 0.555, 0.960, 1.498, 1.755, 2.025, 2.182, 2.322, 2.436, 2.495, 2.554, 2.586, 2.639, 2.651, 2.663, 2.672, 2.677, 2.683, 2.688    ])
RL = np.array([ 4, 10, 20, 30, 50, 70, 100, 150, 200, 300, 400, 800, 1000 , 1500, 2000, 3000,  5000, 10000   ])   #hannop errori
sigma_Vout = []
for i in Vout :
  sigma_Vout.append(calcola_errore_V_v2(i))

# errore Vin misura
print('sigma Vin : ', sigma_Vin)

# threshold RL accettabile:
Vin_mezzi_inf = Vin/2 - sigma_Vin/2

def funzione_partitore_resistivo(RL, R_usata):
  v_in = 5.357
  return RL*v_in/(R_usata+2*RL)
result = lib.fit(RL, Vout, yerr = sigma_Vout, function = funzione_partitore_resistivo, p0= [1], parameter_names=['R_usata'])
print(result['residuals'])
lib.plot_fit(RL, Vout, yerr = sigma_Vout, func = funzione_partitore_resistivo, p0= [1], xlabel = 'RL',ylabel = 'Vout', title= 'Fit partitore resistivo' ,  parameter_names=['R_usata'], show_fit_params=True, show_chi_squared=True, confidence_intervals=True, error_band=1, prediction_band=True)
plt.axhline(y=Vin_mezzi_inf, color='r', linestyle='--', label='Vin/2-sigma/2')
plt.axhline(y=Vin/2, color='g', linestyle='--', label='Vin/2')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def propagate(f, params, cov, x_fit):
    """
    Calcola l'incertezza (sigma) sui valori previsti dalla funzione f,
    propagando le incertezze dei parametri tramite la derivata numerica (Jacobiana).
    
    Parametri:
    - f: funzione che accetta i parametri e restituisce i valori per x_fit.
         Deve essere della forma f(p) = modello(x_fit, *p)
    - params: array dei parametri stimati
    - cov: matrice di covarianza dei parametri
    - x_fit: array di valori di x su cui valutare la funzione f
    
    Restituisce:
    - sigma_y: array degli errori propagati per ciascun x_fit
    """
    # Numero di punti su cui effettuare il fit e numero di parametri
    n = x_fit.size
    m = len(params)
    
    # Calcolo della Jacobiana numerica: J[i, j] = ∂f_i/∂p_j
    eps = np.sqrt(np.finfo(float).eps)
    J = np.zeros((n, m))
    for j in range(m):
        dp = np.zeros_like(params)
        dp[j] = eps * (np.abs(params[j]) + 1e-8)
        f_plus = f(params + dp)
        f_minus = f(params - dp)
        J[:, j] = (f_plus - f_minus) / (2 * dp[j])
    
    # Calcolo della varianza propagata: σ²(x) = J(x) @ cov @ J(x)^T
    # Lo facciamo per ogni punto di x_fit
    sigma2 = np.einsum('ij,jk,ik->i', J, cov, J)
    sigma_y = np.sqrt(sigma2)
    
    return sigma_y



# Fit del modello
popt, pcov = curve_fit(funzione_partitore_resistivo, RL, Vout, sigma= sigma_Vout, p0=[1], absolute_sigma=True)

# Generazione dei punti per il fit, su un intervallo denso
x_fit = np.linspace(np.min(RL), np.max(RL), 1000)
y_fit = funzione_partitore_resistivo(x_fit, *popt)

# Definisco una funzione "wrappata" che varia i parametri (x è fissato a x_fit)
def f_wrapped(p):
    return funzione_partitore_resistivo(x_fit, *p)

# Calcolo dell'errore propagato (sigma) sui valori predetti
sigma_y_fit = propagate(f_wrapped, popt, pcov, x_fit)

# Tracciamento dei risultati
plt.figure(figsize=(10, 6))
plt.errorbar(RL, Vout, yerr=sigma_Vout, fmt='o', capsize=3, label='Dati')
plt.plot(x_fit, y_fit, 'r-', linewidth=2, label='Fit')
plt.fill_between(x_fit, y_fit - sigma_y_fit, y_fit + sigma_y_fit, 
                 color='red', alpha=0.2, label='Intervallo di confidenza (1σ)')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Fit generico e intervallo di confidenza")
plt.legend()
plt.grid(True)
plt.show()
