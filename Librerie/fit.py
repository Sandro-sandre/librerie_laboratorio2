import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import chi2


def eval_function(formula, x, params):
    # Crea un dizionario con i parametri e la variabile x
    namespace = {'x': x}
    namespace.update(params)  # paraemtri nel namespace
    try:
        # Valuta la formula
        return eval(formula, namespace) 
    except Exception as e:
        print(f"Errore nella valutazione della formula: {e}")
        return None

# Funzione residuo per il least_squares
def residuals(params, x, y, sigma_y, formula, param_names):
    # Crea un dizionario con i nomi dei parametri e i loro valori
    params_dict = {name: val for name, val in zip(param_names, params)}
    y_pred = eval_function(formula, x, params_dict)
    return (y - y_pred) / sigma_y

formula_input = input("Inserisci la formula (usa 'x' come variabile e 'a0', 'a1', ... come parametri): ")

x = np.array([float(xi) for xi in input("Inserisci i valori di x separati da spazi: ").split()])
y = np.array([float(yi) for yi in input("Inserisci i valori di y separati da spazi: ").split()])
sigma_y = np.array([float(sigmai) for sigmai in input("Inserisci le incertezze sigma_y separati da spazi: ").split()])

param_names = input("Inserisci i nomi dei parametri separati da spazi (es. 'a0 a1 a2'): ").split()
initial_params = [float(p) for p in input("Inserisci i valori iniziali dei parametri separati da spazi: ").split()]

# Esegui il fit
result = least_squares(residuals, initial_params, args=(x, y, sigma_y, formula_input, param_names))

print("Risultati del fit:")
for name, value in zip(param_names, result.x):
    print(f"{name} = {value}")

# Calcola il chi-quadro ridotto
residuals_final = residuals(result.x, x, y, sigma_y, formula_input, param_names)
chi_squared = np.sum(residuals_final**2)
dof = len(x) - len(initial_params)  # Gradi di libertà
reduced_chi_squared = chi_squared / dof
print(f"Chi-quadro ridotto: {reduced_chi_squared}")

# Calcola il p-value
p_value = 1 - chi2.cdf(chi_squared, dof)
print(f"p-value: {p_value}")

plt.errorbar(x, y, yerr=sigma_y, fmt='o', label='Dati', capsize=5, color='blue')

# Plot della curva fito
x_fit = np.linspace(min(x), max(x), 500)
params_dict = {name: val for name, val in zip(param_names, result.x)}
y_fit = eval_function(formula_input, x_fit, params_dict)
plt.plot(x_fit, y_fit, label='Fit', color='red')

# Aggiunta dei parametri, del chi-quadro ridotto e del p-value al grafico
param_text = "\n".join([f"${name} = {result.x[i]:.3f} \\pm {0.1:.3f}$" for i, name in enumerate(param_names)])
fit_info = (
    f"{param_text}\n"
    f"$\\chi^2/\\mathrm{{dof}} = {reduced_chi_squared:.3f}$\n"
    f"$p\\text{{-value}} = {p_value:.3f}$\n"
    f"$\\text{{Soglia}} = 0.05$"
)
plt.text(0.05, 0.95, fit_info,
         transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))


plt.xlabel("x")
plt.ylabel("y")
plt.title("Fit con Least Squares")
plt.legend()
plt.grid(True)
plt.show()