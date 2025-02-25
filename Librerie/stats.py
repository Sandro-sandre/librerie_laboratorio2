import matplotlib.pyplot as plt
import numpy as np
import sys 
from scipy.stats import norm, describe, expon
import random

# HISTOGRAM PLOT
# Sturges per binnaggio
def sturges(x): 
    return int(np.ceil(1 + 3.322 * np.log(x)))

# Classe che analizza la statistica di un sample
class lavoro :
	sample = []
	N_events = 0
	Sample_sum = 0
	Sample_sumsq = 0 #La somma dei quadrati
	
	def __init__(self, array) :
	
		self.sample = array
		self.N_events = len(self.sample)
		self.Sample_sum = np.sum(self.sample)
		self.Sample_sumsq = np.sum([x*x for x in self.sample]) #come se facessi un nuovo array
	
	def media(self):
		return self.Sample_sum /self.N_events
	
	def varianza_np(self) :
		return np.var(self.sample)

	def varianza(self):
		var = self.Sample_sumsq/self.N_events - (self.media())**2
		return var
	
	def devst(self) :
		return np.sqrt(self.varianza())
			
	def errore_standard(self) : # della media
		return self.devst()/np.sqrt(self.N_events)
	
	def skewness(self) :
		moments = list(describe(self.sample))
		return moments[4]
		
	def kurtosis(self) :
		moments = list(describe(self.sample))
		return moments[5]
		
	def crea_hist(self, output_file) :
		xMin = np.floor(np.min(self.sample))
		xMax = np.ceil(np.max(self.sample))
		N_bins = sturges(len(self.sample))
		bin_edges = np.linspace(xMin, xMax, N_bins)
		print('lenghth of bin_edges container:', len(bin_edges))
		fig, ax = plt.subplots( nrows= 1, ncols=1)
		ax.hist(self.sample, bins = bin_edges, label = 'Gaussian distribution',
			color = 'deepskyblue', density = False)
		ax.set_xlabel('variable')
		ax.set_ylabel('event coounts')
		ax.legend()
		plt.savefig(output_file)


	def dati(self) :
		print("Dati distribuzione: \nMedia:", self.media(),
		      "\nVarianza:", self.varianza(),"\nDeviazione standard:",self.devst(),
		      "\nErrore standard della media:",self.errore_standard(),"\nSkewness:",
		      self.skewness(),"\nKurtosis:", self.kurtosis())
		return "L'istogramma è stato aggiunto correttamente nella directory sottoforma di png"

# PSEUDO-RANDOM NUMBERS
# Funzione Try and Catch per generazione di un sample con una certa pdf
# (se esiste pdf analitica meglio usare scipy)
def rand_TAC(xmin, xmax, ymax, f, max_attempts=100000):
    attempts = 0
    while attempts < max_attempts:
        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(0, ymax)
        if y <= f(x):
            return x  # Restituisce un valore, non un sample!
        attempts += 1
    print('Maximum attempts reached')
    return None


	
# Generazione di un numero pseudo-casuale distribuito fra xMin ed xMax
def rand_range (xMin, xMax) : 
	return xMin + np.random.uniform () * (xMax - xMin)
	
# Generazione di un numero pseudo-casuale con il metodo del teorema centrale del limite su un intervallo fissato
def rand_TCL (xMin, xMax, N_sum = 10) :
	y = 0.
	''' 
	N_sum: Indica quante variabili casuali vengono sommate (o mediate) 
	per ogni singolo numero generato. Più alto è N_sum, più il risultato
	si avvicina a una distribuzione normale. Questo parametro controlla
	quanto fortemente la distribuzione risultante si approssima a una Gaussiana. 
 	'''
	for i in range (N_sum) : 
		y = y + rand_range (xMin, xMax)
	y /= N_sum 
	return y 

# Generazione di N numeri pseudo-casuali con il metodo del TCL, note media e sigma della gaussiana, a partire da un determinato seed
def generate_TCL_ms (mean, sigma, N, N_sum = 10, seed = 0.) :	
	#  N: Indica quanti numeri pseudo-casuali totali vengono generati,
    #  ossia la dimensione del campione prodotto.
	if seed != 0. : random.seed (float (seed))
	randlist = []
	delta = np.sqrt (3 * N_sum) * sigma
	xMin = mean - delta
	xMax = mean + delta
	for i in range (N):
        	# Return the next random floating point number in the range 0.0 <= X < 1.0
		randlist.append (rand_TCL (xMin, xMax, N_sum))
	return randlist





