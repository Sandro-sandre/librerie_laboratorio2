import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2
import math
from IPython.display import display


def model(x, params, formula):
    # Crea un dizionario con i parametri e la variabile x
    namespace = {'x': x, 'np': np, 'math': math}
    namespace.update(params)
    namespace.update(np.__dict__)
    namespace.update(math.__dict__)
    try:
        return eval(formula, namespace)
    except Exception as e:
        print(f"Errore nella valutazione della formula: {e}")
        return None


formula_input = input("Inserisci la formula (usa 'x' come variabile e 'a0', 'a1', ... come parametri): ")


x_data = np.array([float(xi) for xi in input("Inserisci i valori di x separati da spazi: ").split()])
y_data = np.array([float(yi) for yi in input("Inserisci i valori di y separati da spazi: ").split()])
sigma_y = np.array([float(sigmai) for sigmai in input("Inserisci le incertezze sigma_y separati da spazi: ").split()])


param_names = input("Inserisci i nomi dei parametri separati da spazi (es. 'a0 a1 a2'): ").split()
initial_params = {name: float(val) for name, val in zip(param_names, input("Inserisci i valori iniziali dei parametri separati da spazi: ").split())}


least_squares = LeastSquares(x_data, y_data, sigma_y, lambda x, *args: model(x, dict(zip(param_names, args)), formula_input))
m = Minuit(least_squares, *initial_params.values())
m.migrad()
display(m)
print("\nRisultati del fit:")
for i, name in enumerate(param_names):
    print(f"{name} = {m.values[i]} ± {m.errors[i]}")

# Calcolo del chi-quadro ridotto
chi_squared = m.fval
dof = len(x_data) - len(param_names)  # Gradi di libertà
reduced_chi_squared = chi_squared / dof
print(f"Chi-quadro ridotto: {reduced_chi_squared}")

# Calcolo del p-value
p_value = 1 - chi2.cdf(chi_squared, dof)
print(f"p-value: {p_value}")


plt.errorbar(x_data, y_data, yerr=sigma_y, fmt='o', label='Dati', capsize=5, color='blue')
x_fit = np.linspace(min(x_data), max(x_data), 500)
params_fit = {name: m.values[i] for i, name in enumerate(param_names)}
y_fit = model(x_fit, params_fit, formula_input)
plt.plot(x_fit, y_fit, label='Fit', color='red')

# Aggiunta dei parametri, del chi-quadro ridotto e del p-value al grafico
param_text = "\n".join([f"${name} = {m.values[i]:.3f} \\pm {m.errors[i]:.3f}$" for i, name in enumerate(param_names)])
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