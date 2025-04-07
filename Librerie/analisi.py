import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.odr import RealData, Model, ODR
from typing import Union, List, Dict, Callable, Optional
from scipy.integrate import quad
from scipy.stats import chi2, t, norm



#------------propagazione errore----------------
'''
    formula = "log(x)+exp(y)"
    variables = ["x", "y"]
    errors = [1, 2]
    values = [10,4] 
'''

def error_propagation(formula, variables, errors, values):
    sym_variables = [sp.symbols(var) for var in variables]
    expr = sp.sympify(formula)
    partial_derivates = [sp.diff(expr, var) for var in sym_variables]
    partial_derivates_num = [deriv.subs(zip(sym_variables, values)) for deriv in partial_derivates]

    error_squared_formula= sum((deriv*error)**2 for deriv, error in zip(partial_derivates, errors))
    errore_propagato_formula = sp.sqrt(error_squared_formula)

    error_squared_num = sum((deriv*error)**2 for deriv, error in zip(partial_derivates_num, errors))
    errore_propagato_num = sp.sqrt(error_squared_num)
    return errore_propagato_formula, errore_propagato_num.evalf(4)


#------------fit----------------

def fit(x: np.ndarray, 
        y: np.ndarray, 
        function: Callable, 
        xerr: Optional[Union[List[float], np.ndarray]] = None, 
        yerr: Optional[Union[List[float], np.ndarray]] = None, 
        p0: Optional[Union[float, List[float], np.ndarray]] = None, 
        chi_square: bool = True, 
        method: str = 'auto', 
        parameter_names: Optional[List[str]] = None) -> Dict:
    if p0 is not None:
        p0 = np.atleast_1d(p0)

    if method == 'auto':
        if xerr is not None and yerr is not None:
            method = 'odr'
        else:
            method = 'curve_fit'

    if method == 'curve_fit':
        if yerr is not None:
            sigma = np.array(yerr)
        else:
            sigma = None

        popt, pcov = curve_fit(function, x, y, p0=p0, sigma=sigma, absolute_sigma=True)
        residuals = y - function(x, *popt)
        chi2 = np.sum((residuals / sigma) ** 2) if sigma is not None else None

    elif method == 'odr':
        model = Model(function)
        data = RealData(x, y, sx=xerr, sy=yerr)
        odr = ODR(data, model, beta0=p0)
        output = odr.run()
        popt = output.beta
        pcov = output.cov_beta
        residuals = y - function(x, *popt)
        chi2 = output.res_var * len(x) if chi_square else None

    else:
        raise ValueError("Invalid method. Choose 'curve_fit', 'odr', or 'auto'.")

    result = {
        'parameters': popt,
        'covariance': pcov,
        'residuals': residuals,
        'chi2': chi2
    }

    if parameter_names:
        result['parameters_named'] = dict(zip(parameter_names, popt))

    return result

#------------grafico fit----------------
def plot_fit(
    *args: Dict, 
    func: Callable, 
    p0: Optional[Union[List[float], List[List[float]]]] = None, 
    xlabel: str = '', 
    ylabel: str = '', 
    title: str = '', 
    colors: Optional[List[str]] = None, 
    labels: Optional[List[str]] = None, 
    fit_line: bool = True, 
    label_fit: Optional[List[str]] = None, 
    together: bool = True, 
    figsize: tuple = (10, 6), 
    save_path: Optional[str] = None, 
    dpi: int = 300, 
    grid: bool = False, 
    legend_loc: str = 'best', 
    fit_points: int = 1000, 
    fit_range: Optional[tuple] = None, 
    confidence_interval: Optional[float] = None, 
    residuals: bool = False, 
    fmt: str = '+', 
    markersize: int = 6, 
    linewidth: float = 1.5, 
    show_fit_params: bool = False, 
    show_chi_squared: bool = False, 
    xlim: Optional[tuple] = None, 
    ylim: Optional[tuple] = None, 
    capsize: float = 3, 
    axis_fontsize: int = 12, 
    title_fontsize: int = 14, 
    masks: Optional[List[np.ndarray]] = None, 
    parameter_names: Optional[List[List[str]]] = None, 
    method: str = 'auto', 
    show_masked_points: bool = True, 
    masked_fmt: str = 'x', 
    masked_color: Optional[str] = None, 
    masked_alpha: float = 0.5
):
    if together:
        plt.figure(figsize=figsize)

        if residuals:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, ax1 = plt.subplots(figsize=figsize)

        for i, dataset in enumerate(args):
            x = np.array(dataset['x']['value'])
            y = np.array(dataset['y']['value'])
            xerr = np.array(dataset['x']['error']) if 'error' in dataset['x'] else None
            yerr = np.array(dataset['y']['error']) if 'error' in dataset['y'] else None

            mask = masks[i] if masks and i < len(masks) else np.ones_like(x, dtype=bool)
            x_fit, y_fit = x[mask], y[mask]
            xerr_fit = xerr[mask] if xerr is not None else None
            yerr_fit = yerr[mask] if yerr is not None else None

            initial_p0 = p0[i] if isinstance(p0, list) and len(p0) > i else p0
            fit_result = fit(x_fit, y_fit, func, xerr=xerr_fit, yerr=yerr_fit, p0=initial_p0, method=method, parameter_names=parameter_names[i] if parameter_names else None)

            color = colors[i] if colors and i < len(colors) else None
            label = labels[i] if labels and i < len(labels) else None
            ax1.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=fmt, markersize=markersize, label=label, color=color, capsize=capsize)
            if show_masked_points and not np.all(mask):
                masked_color = masked_color or color
                ax1.errorbar(x[~mask], y[~mask], xerr=xerr[~mask] if xerr is not None else None, yerr=yerr[~mask] if yerr is not None else None, fmt=masked_fmt, markersize=markersize, color=masked_color, alpha=masked_alpha, capsize=capsize)

            if fit_line:
                fit_x = np.linspace(fit_range[0], fit_range[1], fit_points) if fit_range else np.linspace(min(x_fit), max(x_fit), fit_points)
                fit_y = func(fit_x, *fit_result['parameters'])
                ax1.plot(fit_x, fit_y, color=color, linewidth=linewidth, label=label_fit[i] if label_fit and i < len(label_fit) else None)

                if confidence_interval:
                    ci = confidence_interval * np.sqrt(np.diag(fit_result['covariance']))
                    ax1.fill_between(fit_x, fit_y - ci, fit_y + ci, color=color, alpha=0.2)

            if show_fit_params:
                param_text = '\n'.join([f'{name} = {val:.3f}' for name, val in zip(parameter_names[i], fit_result['parameters'])]) if parameter_names else '\n'.join([f'p{i} = {val:.3f}' for i, val in enumerate(fit_result['parameters'])])
                ax1.text(0.05, 0.95, param_text, transform=ax1.transAxes, fontsize=axis_fontsize, verticalalignment='top')

            if show_chi_squared and fit_result['chi2'] is not None:
                ax1.text(0.05, 0.85, f'Chi2 = {fit_result["chi2"]:.3f}', transform=ax1.transAxes, fontsize=axis_fontsize, verticalalignment='top')

            if residuals:
                residuals = y - func(x, *fit_result['parameters'])
                ax2.errorbar(x, residuals, xerr=xerr, yerr=yerr, fmt=fmt, markersize=markersize, color=color, capsize=capsize)
                ax2.axhline(0, color='black', linewidth=linewidth, linestyle='--')
                ax2.set_xlabel(xlabel, fontsize=axis_fontsize)
                ax2.set_ylabel('Residuals', fontsize=axis_fontsize)
            ax1.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=fmt, markersize=markersize, label=label, color=color, capsize=capsize)
            if show_masked_points and not np.all(mask):
                masked_color = masked_color or color
                ax1.errorbar(x[~mask], y[~mask], xerr=xerr[~mask] if xerr is not None else None, yerr=yerr[~mask] if yerr is not None else None, fmt=masked_fmt, markersize=markersize, color=masked_color, alpha=masked_alpha, capsize=capsize)

            if fit_line:
                fit_x = np.linspace(fit_range[0], fit_range[1], fit_points) if fit_range else np.linspace(min(x_fit), max(x_fit), fit_points)
                fit_y = func(fit_x, *fit_result['parameters'])
                ax1.plot(fit_x, fit_y, color=color, linewidth=linewidth, label=label_fit[i] if label_fit and i < len(label_fit) else None)

            if confidence_interval:
                ci = confidence_interval * np.sqrt(np.diag(fit_result['covariance']))
                ax1.fill_between(fit_x, fit_y - ci, fit_y + ci, color=color, alpha=0.2)

            if show_fit_params:
                param_text = '\n'.join([f'{name} = {val:.3f}' for name, val in zip(parameter_names[i], fit_result['parameters'])]) if parameter_names else '\n'.join([f'p{i} = {val:.3f}' for i, val in enumerate(fit_result['parameters'])])
                ax1.text(0.05, 0.95, param_text, transform=ax1.transAxes, fontsize=axis_fontsize, verticalalignment='top')

            if show_chi_squared and fit_result['chi2'] is not None:
                ax1.text(0.05, 0.85, f'Chi2 = {fit_result["chi2"]:.3f}', transform=ax1.transAxes, fontsize=axis_fontsize, verticalalignment='top')

            if residuals:
                residuals = y - func(x, *fit_result['parameters'])
                ax2.errorbar(x, residuals, xerr=xerr, yerr=yerr, fmt=fmt, markersize=markersize, color=color, capsize=capsize)
                ax2.axhline(0, color='black', linewidth=linewidth, linestyle='--')
                ax2.set_xlabel(xlabel, fontsize=axis_fontsize)
                ax2.set_ylabel('Residuals', fontsize=axis_fontsize)

        ax1.set_xlabel(xlabel, fontsize=axis_fontsize)
        ax1.set_ylabel(ylabel, fontsize=axis_fontsize)
        ax1.set_title(title, fontsize=title_fontsize)
        ax1.grid(grid)
        ax1.legend(loc=legend_loc)
        if xlim:
            ax1.set_xlim(xlim)
        if ylim:
            ax1.set_ylim(ylim)
        if save_path:
            plt.savefig(save_path, dpi=dpi)
        plt.show()

    for i, dataset in enumerate(args):
        x = np.array(dataset['x']['value'])
        y = np.array(dataset['y']['value'])
        xerr = np.array(dataset['x']['error']) if 'error' in dataset['x'] else None
        yerr = np.array(dataset['y']['error']) if 'error' in dataset['y'] else None

        mask = masks[i] if masks and i < len(masks) else np.ones_like(x, dtype=bool)
        x_fit, y_fit = x[mask], y[mask]
        xerr_fit = xerr[mask] if xerr is not None else None
        yerr_fit = yerr[mask] if yerr is not None else None

        initial_p0 = p0[i] if isinstance(p0, list) and len(p0) > i else p0
        fit_result = fit(x_fit, y_fit, func, xerr=xerr_fit, yerr=yerr_fit, p0=initial_p0, method=method, parameter_names=parameter_names[i] if parameter_names else None)

        if not together:
            plt.figure(figsize=figsize)

        color = colors[i] if colors and i < len(colors) else None
        label = labels[i] if labels and i < len(labels) else None

        plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=fmt, markersize=markersize, label=label, color=color, capsize=capsize)
        if show_masked_points and not np.all(mask):
            masked_color = masked_color or color
            plt.errorbar(x[~mask], y[~mask], xerr=xerr[~mask] if xerr is not None else None, yerr=yerr[~mask] if yerr is not None else None, fmt=masked_fmt, markersize=markersize, color=masked_color, alpha=masked_alpha, capsize=capsize)

        if fit_line:
            fit_x = np.linspace(fit_range[0], fit_range[1], fit_points) if fit_range else np.linspace(min(x_fit), max(x_fit), fit_points)
            fit_y = func(fit_x, *fit_result['parameters'])
            plt.plot(fit_x, fit_y, color=color, linewidth=linewidth, label=label_fit[i] if label_fit and i < len(label_fit) else None)

            if confidence_interval:
                ci = confidence_interval * np.sqrt(np.diag(fit_result['covariance']))
                plt.fill_between(fit_x, fit_y - ci, fit_y + ci, color=color, alpha=0.2)

        if show_fit_params:
            param_text = '\n'.join([f'{name} = {val:.3f}' for name, val in zip(parameter_names[i], fit_result['parameters'])]) if parameter_names else '\n'.join([f'p{i} = {val:.3f}' for i, val in enumerate(fit_result['parameters'])])
            plt.text(0.05, 0.95, param_text, transform=plt.gca().transAxes, fontsize=axis_fontsize, verticalalignment='top')

        if show_chi_squared and fit_result['chi2'] is not None:
            plt.text(0.05, 0.85, f'Chi2 = {fit_result["chi2"]:.3f}', transform=plt.gca().transAxes, fontsize=axis_fontsize, verticalalignment='top')

        if not together:
            plt.xlabel(xlabel, fontsize=axis_fontsize)
            plt.ylabel(ylabel, fontsize=axis_fontsize)
            plt.title(title, fontsize=title_fontsize)
            plt.grid(grid)
            plt.legend(loc=legend_loc)
            if xlim:
                plt.xlim(xlim)
            if ylim:
                plt.ylim(ylim)
            if save_path:
                plt.savefig(f'{save_path}_dataset_{i}.png', dpi=dpi)
            plt.show()

    if together:
        plt.xlabel(xlabel, fontsize=axis_fontsize)
        plt.ylabel(ylabel, fontsize=axis_fontsize)
        plt.title(title, fontsize=title_fontsize)
        plt.grid(grid)
        plt.legend(loc=legend_loc)
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        if save_path:
            plt.savefig(save_path, dpi=dpi)
        plt.show()





#------------integrali----------------
# Calcola integrale [Scipy]
def integral_scipy(f, a, b) : 
  integral = quad(f, a,b)
  return integral[0], integral[1]


#----------media pesata----------------
def media_pesata(x, sigma) :
  m = np.sum(x/sigma**2)/np.sum(1/sigma**2)
  sigma_m = 1/np.sqrt(np.sum(1/sigma**2))
  return m, sigma_m



#------------PDF's & CDF's----------------
def Gaussian(x, mu = 0, sigma = 1) :
	return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x - mu)**2) / (2 * sigma**2))

def Gaussian_standard(z):
  return (1/np.sqrt(2*np.pi))*np.exp((-z**2)/2) 

def Gaussian_cdf_ext(bin_edges, s, mu, sigma) :
  return s*norm.cdf(bin_edges, mu, sigma)

def Gaussian_cdf(bin_edges, mu, sigma) :
  return norm.cdf(bin_edges, mu, sigma)


#----------------HYPOTHESIS TESTING----------------

# Calcolo del p-value
def p_value(chi_square, ndof) :
  s = 1-chi2.cdf(chi_square, ndof)
  r = s*100
  return r

# z_test double sided
def z_test1(x1,x2,s1,s2) : 
  z = np.absolute(x1-x2)/np.sqrt(s1**2+s2**2)  #t di confronto
  R = quad(Gaussian_standard,-z,z) #calcolo del rapporto con l'integrale
  p_value = (1 - R[0])
  return p_value

# z test di ipotesi con un valore calcolato
def z_test2(x1,X,s) :  
  z = np.absolute(x1-X)/s  #t di confronto
  R = quad(Gaussian_standard,-z,z) #calcolo del rapporto con l'integrale
  p_value = (1 - R[0])
  return p_value

# t test con 1 vincolo
def t_test1(x1, X, N, err_media) : # N = N_dati 
	t_test = np.absolute(x1-X)/err_media
	R = t.cdf(-t_test, df=N-1)
	p_value = R*2
	return p_value
	
# t test con 2 vincoli
def t_test2(x1, x2, N, err1, err2) : 
	t_test = np.absolute(x1-x2)/np.sqrt(err1**2+err2**2)
	R = t.cdf(-t_test, df = N-1)
	p_value = R*2
	return p_value


