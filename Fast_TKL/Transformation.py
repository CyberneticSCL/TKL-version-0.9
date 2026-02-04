import numpy as np
import itertools
import math
from matplotlib import pyplot as plt
from libsvm import svmutil

def x_to_basis(x, monomial_index):
    '''
    x should be shape (len samples, prob_dim)
    '''
    monom_to_val = lambda x, monom: np.prod([x[i]**monom[i]/math.factorial(monom[i]) for i in range(len(x))])
    
    arr_x = np.repeat(np.array(x)[np.newaxis, :], len(monomial_index), axis = 0)
    
    answ = list(map(monom_to_val, arr_x, monomial_index))
    
    return answ


def x_vec_to_basis(x, monomial_index):
    '''
    x should be shape (len samples, prob_dim)
    '''
    monom_to_val = lambda x, monom: np.prod([x[:, i]**monom[i]/math.factorial(monom[i]) for i in range(x.shape[-1])], axis = 0)
    
    arr_x = np.zeros((len(x), len(monomial_index)))
    
    
    for mon_ind in range(len(monomial_index)):
#         print(mon_ind, monomial_index[mon_ind])
        values_ind = monom_to_val(x, monomial_index[mon_ind])
        arr_x[:, mon_ind] = values_ind
#         print(values_ind)
#     arr_x = np.repeat(np.array(x)[np.newaxis, :], len(monomial_index), axis = 0)
    
#     answ = list(map(monom_to_val, arr_x, monomial_index))
    
    return arr_x

def monomials(X, d):
    '''
    z = monomials(x,d) function takes a matrix of inputs (x) and computes
    the monomial basis of degree d (z).
    INPUT
    X: Input data. (N_samples, N_features)
    d: Maximum monomial degree.
        
    OUTPUT
    z: Monomial basis of the input data.  
    '''
    if d == 1:
        Z = np.c_[ np.ones(len(X)), X ]  
        return Z
    
    prob_dim = X.shape[-1]
    monomial_index = np.array(list(itertools.product(list(range(d+1)), repeat = prob_dim)))
    monomial_index = monomial_index[monomial_index.sum(axis = -1) <= d][:,::-1]

    Z = x_vec_to_basis(X, monomial_index)
#     Z = np.zeros((len(X), len(monomial_index )))
#     for i in range(len(X)):
#         Z[i] = np.array(x_to_basis(X[i, :], 
#                                    monomial_index))
    return Z
 
