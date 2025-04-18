import numpy as np
from scipy.integrate import quad
from scipy.stats import norm, chi2, t
import matplotlib.pyplot as plt
from iminuit import Minuit


# ZEROS & EXTREMA 
# Trova minimo [metodo Bisezione]
def bisezione(xmin, xmax, f, prec = 0.0001 , max_attempts = 10000) : 
	if f(xmin)*f(xmax) >= 0 :
		raise ValueError("Non c'è nessuno zero nell'intervallo compatto o la funzione agli estremi dell'intervallo non ha segni opposti ")
		
	i = 0
	while (i< max_attempts and (xmax - xmin) > prec) :
		xave = (xmax + xmin)/2
		if f(xmin)*f(xave) > 0 :
			xmin = xave
		else :
			xmax = xave
			
		i += 1
		
	return (xmax+xmin)/2

# Trova max [Golden Ratio Method]
def max1(f, xmin, xmax, prec=0.0001, max_attempts=10000): #(attenzione segni cambiano tra minimo e massimo)
    phi = (np.sqrt(5) - 1) / 2
    x1 = xmin + phi * (xmax - xmin)
    x2 = xmin + (1 - phi) * (xmax - xmin)
    i = 0
    while abs(xmax - xmin) > prec and i < max_attempts:
        if f(x2) < f(x1):
            xmin = x2
            x2 = x1
            x1 = phi * (xmax - xmin) + xmin
        else:
            xmax = x1
            x1 = x2
            x2 = xmin + (1 - phi) * (xmax - xmin)
        i += 1
    x_max = (x1 + x2) / 2
    return x_max, f(x_max)

# INTEGRALS CALCULATION
# Calcola integrale [Hit or Miss] 
def integral_HoM(f, xmin, xmax, ymin, ymax , N_evt) :  
	x_coord = np.random.uniform(xmin, xmax, N_evt)
	y_coord = np.random.uniform(ymin, ymax, N_evt)
	f_coord = f(x_coord)
	nhits = np.sum((y_coord>=0) & (y_coord<=f_coord))-np.sum((y_coord<0) & (y_coord>f_coord))
	'''arrayerr = [integral(func, 0, 2*np.pi, -1,1, x) for x in range(100,N_max,20)]
	y_coord = list(arrayerr[i][0] for i in range(len(arrayerr)))
	y_err = list(arrayerr[i][1] for i in range(len(arrayerr)))'''
	if nhits < 0 :
		nhits = abs(nhits)
	area = (xmax -xmin)*(ymax -ymin)
	integer = area*(nhits/N_evt)
	p = abs(nhits) / N_evt  # Proporzione assoluta di successi
	uncertainty = np.sqrt(area**2 * p * (1 - p) / N_evt)
	return integer, uncertainty	

# Calcola integrale [Monte Carlo]
def integral_MC(f, a, b, N_evt) :  
	x_random = np.random.uniform(a, b, N_evt)
	mean_f = np.mean(f(x_random))
	integer = (b-a)*mean_f
	uncertainty = np.std(x_random)/np.sqrt(N_evt)
	return integer, uncertainty

# Calcola integrale [Scipy]
def integral_scipy(f, a, b) : 
  integral = quad(f, a,b)
  return integral[0], integral[1]

# EXTRA 

def media_pesata(x, sigma) :
  m = np.sum(x/sigma**2)/np.sum(1/sigma**2)
  sigma_m = 1/np.sqrt(np.sum(1/sigma**2))
  return m, sigma_m


def loglikelihood (theta, pdf, lista) :
	r = 0 
	for x in lista :
		if pdf(x, theta) >0 :
			r = r + np.log(pdf(x, theta))
	return r

# PDF's & CDF's
def Gaussian(x, mu = 0, sigma = 1) :
	return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x - mu)**2) / (2 * sigma**2))

# Gaussiana standardizzata
def Gaussian_standard(z):
  return (1/np.sqrt(2*np.pi))*np.exp((-z**2)/2) 

def Gaussian_cdf_ext(bin_edges, s, mu, sigma) :
  return s*norm.cdf(bin_edges, mu, sigma)

def Gaussian_cdf(bin_edges, mu, sigma) :
  return norm.cdf(bin_edges, mu, sigma)


# HYPOTHESIS TESTING 
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


def grafico_fit(fit, x_coord, y_coord, sigma_x, sigma_y, f) :
    fit.migrad()
    fit.hesse()
    Q_squared = fit.fval
    N_dof = fit.ndof
    p_val = p_value(fit.fval, fit.ndof)
    print('Success of the fit:', fit.valid)
    plt.errorbar(x_coord, y_coord, xerr= sigma_x, yerr = sigma_y, fmt="ok", label="data")
    x = np.linspace(min(x_coord), max(x_coord), 10000)
    plt.plot(x, f(x, *fit.values), label="fit")
    fit_info = [f"$\\chi^2$/$n_\\mathrm{{dof}}$ = {fit.fval:.1f} / {fit.ndof:.0f} = {fit.fval/fit.ndof:.1f}",
    f"$p$-value = {p_val:.3f}"]
    for p, v, e in zip(fit.parameters, fit.values, fit.errors):
        fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")
    plt.legend(title="\n".join(fit_info), frameon=False)
    #plt.xlabel("x")
    #plt.ylabel("y")
    #plt.show()
	
	


	



	


