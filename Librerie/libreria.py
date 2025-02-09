import numpy as np
from scipy.integrate import quad 

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


def loglikelihood (theta, pdf, lista) :
	r = 0 
	for x in lista :
		if pdf(x, theta) >0 :
			r = r + np.log(pdf(x, theta))
	return r

# Gaussiana standardizzata 
def Gaussian(x, mu = 0, sigma = 1) :
	return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x - mu)**2) / (2 * sigma**2))

def Gaussian_cdf_ext(bin_edges, s, mu, sigma) :
  return s*norm.cdf(bin_edges, mu, sigma)

def Gaussian_cdf(bin_edges, mu, sigma) :
  return norm.cdf(bin_edges, mu, sigma)


	


