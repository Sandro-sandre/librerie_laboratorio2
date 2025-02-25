from sympy import *
import sys 
import numpy as np

def error_propagation(formula, variables, errors, values):
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
    partial_derivates_num = [deriv.subs(zip(sym_variables, values)) for deriv in partial_derivates]

    error_squared_formula= sum((deriv*error)**2 for deriv, error in zip(partial_derivates, errors))
    errore_propagato_formula = sqrt(error_squared_formula)

    error_squared_num = sum((deriv*error)**2 for deriv, error in zip(partial_derivates_num, errors))
    errore_propagato_num = sqrt(error_squared_num)
    return errore_propagato_formula, errore_propagato_num.evalf(4)

def main():
    '''
    formula = "log(x)+exp(y)"
    variables = ["x", "y"]
    errors = [1, 2]
    values = [10,4]'''
    if len(sys.argv)<5 :
        print("usa: python3 propagation_error.py <formula> <variables> <errors> <values>")
        print("example: python3 propagation_error.py 'log(x)+exp(y)' 'x,y' '1,2' '10,4'")
        sys.exit(1)

    
    formula = sys.argv[1] #mettere come stringa
    variables = sys.argv[2].split(',')
    errors = list(map(float, sys.argv[3].split(',')))
    values = list(map(float, sys.argv[4].split(',')))


    formula, result = error_propagation(formula, variables, errors, values)
    print(f"errore propagato sulla formula'{formula}': {formula}")
    print(f"valore numerico della propagazione:{result}")
    print(f"Formula propagazione in Latex : {latex(formula)}")

if __name__ == "__main__" :
    main()