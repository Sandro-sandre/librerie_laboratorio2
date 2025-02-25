from sympy import *
import sys 
import numpy as np

def error_propagation(formula, variables, errors):
    '''
    calcola propagazione degli errori con formula data 
    formula deve essere simbolica 
    errors: lista degli errori associate alle variabili
    variabili lista delle variabili 
    TUTTE LE LISTE DEVONO essere in stringhe tranne quella degli errori
    '''
    sym_variables = [symbols(var) for var in variables]
    expr = sympify(formula)
    partial_derivates = [diff(expr, var) for var in sym_variables]
    
    error_squared = sum((deriv*error)**2 for deriv, error in zip(partial_derivates, errors))
    errore_propagato = sqrt(error_squared)
    return errore_propagato.evalf(4)

def main():
    
    formula = "x+y"
    variables = ["x", "y"]
    errors = [1, 2]

    result = error_propagation(formula, variables, errors)
    print(f"errore propagato sulla formula'{formula}': {result}")

if __name__ == "__main__" :
    main()