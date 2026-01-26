import numpy as np
import numba as nb
from numba import jit, njit, prange
import time


def solve_QP_IPM(D, V, c, b, l, u, alpha0, y0, mu10, mu20, 
                 max_iter = 30, tol = 1.e-8, div_tol = 1.e-20, print_losses = 0):
    '''
    Solving QP IPM 
    min  alpha^T Q alpha - c^T alpha
    s.t. b^T alpha = 0
         l < alpha < u
    RETURN alpha_var, y_var, mu1_var, mu2_var, primal_loss, dual_loss
    '''
    
    N = len(D)
    convergence_marker = 0
    # using initial guess
    alpha_var = alpha0
    mu1_var   = mu10
    mu2_var   = mu20
    y_var     = y0

    WBI_inv_Q= woodbury_inverse(D, V)

    for iteration in range(max_iter):
        
#         start1 = time.time()
        AL_inv = 1/(alpha_var - l + div_tol)
        UA_inv = 1/(u - alpha_var + div_tol)
        D_mat = D + (UA_inv*mu2_var)[:, 0] + (AL_inv*mu1_var)[:, 0]  
        
        
#         start2 = time.time()
        WBI_inv  = woodbury_inverse(D_mat, V)
    
        V_div_D  = (V.T/D_mat).T
        V_div_D_Q= (V.T/D).T         
        ##### PREDICTOR STEP #####
#         start3 = time.time()
        R = np.block([[matvec_Q(alpha_var, D, V) - c - y_var*b], [-b.T@alpha_var]]) 
        sol1 = matvec_Gbb0_inv(R, D_mat, WBI_inv, V_div_D, b, -1)     
        
        
        Delta_alpha = sol1[0]
        Delta_y     = sol1[1]
        
        # Find Delta mu_1, mu_2
        Delta_mu1   = -mu1_var - AL_inv*(mu1_var*Delta_alpha)
        Delta_mu2   = -mu2_var + UA_inv*(mu2_var*Delta_alpha)
     
        
        
#         start4 = time.time()
        # Largest Step Size computation
        step_size = compute_largest_possible_step_v2(alpha_var, mu1_var, mu2_var, Delta_alpha, Delta_mu1, Delta_mu2, l, u)
        step_size = step_size

#         start5 = time.time()
        # Predictor Grad Step 
        alpha_upd = alpha_var + step_size*Delta_alpha
        mu1_upd   = mu1_var   + step_size*Delta_mu1
        mu2_upd   = mu2_var   + step_size*Delta_mu2

        # compute optimization parameters
        mu     =  ((u - alpha_var).T@mu2_var + (alpha_var - l).T@mu1_var)/(2*N)
        mu_aff =  ((u - alpha_upd).T@mu2_upd + (alpha_upd - l).T@mu1_upd)/(2*N)
        sigma  =  (mu_aff/mu)**3 


#         start6 = time.time()
        ##### Corrector Step ##### 
        # corrector for quadratic terms
        update_AM1 = (step_size**2)*(Delta_alpha*Delta_mu1)
        update_AM2 = (step_size**2)*(Delta_alpha*Delta_mu2)
        
        r1 = AL_inv*(sigma*mu - update_AM1)
        r2 = UA_inv*(sigma*mu + update_AM2) 
        
        # RHS
        R[:N,:] = R[:N, :] - r1 + r2# = np.block([[matvec_Q(alpha_var, D, V) - c - y_var*b - r1 + r2 ], [-b.T@alpha_var]]) 

#         start7 = time.time()
        # solution for corrector step
        sol2 =  matvec_Gbb0_inv(R, D_mat, WBI_inv, V_div_D, b, -1)
        Delta_alpha_corr = sol2[0]
        Delta_y_corr     = sol2[1]

        # compute Delta mu_1, mu_2 for corrector step
        Delta_mu1_corr   = -mu1_var - AL_inv*(mu1_var * Delta_alpha_corr) + r1
        Delta_mu2_corr   = -mu2_var + UA_inv*(mu2_var * Delta_alpha_corr) + r2

#         start8 = time.time()
        # step size for corrector
        step_size2 = compute_largest_possible_step_v2(alpha_var, mu1_var, mu2_var, 
                                                   Delta_alpha_corr, Delta_mu1_corr, Delta_mu2_corr, l, u)
        step_size2 = step_size2*0.99

#         start9 = time.time()
        # update decision variables
        alpha_var = alpha_var + step_size2*Delta_alpha_corr
        mu1_var   = mu1_var   + step_size2*Delta_mu1_corr
        mu2_var   = mu2_var   + step_size2*Delta_mu2_corr
        y_var     = y_var     + step_size2*Delta_y_corr
#         print(step_size2, step_size)
        #loss function
        primal_loss = 0.5*alpha_var.T@matvec_Q(alpha_var, D, V) - c.T@alpha_var

#         start10 = time.time()
        
#         subvector = np.block([[mu1_ - mu2_ + c + y_*b], [0]]) # matvec_Gbb0_inv(subvector, D, WBI, V_sub_D, b, -1)[0]#
    #         alpha_star =np.linalg.inv(Q)@(mu1_ - mu2_ + c + y_*b)
        alpha_star = matvec_Qinv(mu1_var - mu2_var + c + y_var*b, D, WBI_inv_Q, V_div_D_Q)
        dual_loss = -0.5*alpha_star.T@matvec_Q(alpha_star, D, V) + l*np.sum(mu1_var) - u*np.sum(mu2_var)
        
#         start11 = time.time()    
        
        if print_losses:
            print('Primal Loss %.2e' % primal_loss, ' Dual Loss %.2e' % dual_loss,
                  ' Dual Gap %.2e' % (primal_loss - dual_loss), ' Step Size %.2e' % step_size2)
#             print('|| Inter 1 %.2e ||'  % (start2 - start1),
#                   '|| Inter 2 %.2e ||'  % (start3 - start2),
#                   '|| Inter 3 %.2e ||'  % (start4 - start3),
#                   '|| Inter 4 %.2e ||'  % (start5 - start4),
#                   '|| Inter 5 %.2e || \n'  % (start6 - start5),
#                   '|| Inter 6 %.2e ||'  % (start7 - start6),
#                   '|| Inter 7 %.2e ||'  % (start8 - start7),
#                   '|| Inter 8 %.2e ||'  % (start9 - start8),
#                   '|| Inter 9 %.2e ||'  % (start10 - start9),
#                   '|| Inter 10 %.2e || \n'  % (start11 - start10),
#              )
        if np.abs(primal_loss-dual_loss)/np.abs(primal_loss) < tol:
            
            convergence_marker = 1
            break
        
    if convergence_marker == 0:
        print('The problem does not converge: \n Increase maximum number of iterations or change initial guess')
        print('Duality Gap %.2e' % ((primal_loss - dual_loss)/np.abs(primal_loss)))
    else:
        
        if print_losses:
            print('convergence in #iter', iteration, 
                  ', Duality Gap %.2e'  % ((primal_loss - dual_loss)/np.abs(primal_loss)))

    return alpha_var, y_var, mu1_var, mu2_var, primal_loss, dual_loss


# @njit(nb.float64[:, :](nb.float64[:], nb.float64[:,:], nb.float64))
def woodbury_inverse(D, V, tol = 0):
    """
    Computes the inverse of (D + V V^T) using the Woodbury matrix identity:
    (D + V V^T)^(-1) = D^(-1) - D^(-1) V (I + V^T D^(-1) V)^(-1) V^T D^(-1)
    
    Parameters:
    - D: (n, ) diagonal matrix part
    - V: (n x r) matrix
     
    OUTPUT:
    (I + V^T D^{-1} V)^{-1}
    """
    D_inv = 1/(D + tol)
    DV = D_inv*V.T
    DV = DV.T
    VDV= V.T@DV + np.eye(V.shape[1])
   
    result = np.linalg.inv(VDV)
    
    return result

# @njit(nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:,:]))
def matvec_Q(vector, D, V):
    '''
    Q = D + V V^T
    '''
    vector1 = V.T @vector
    vector2 = V@vector1
    vector3 = (D*vector.T).T
    vector4 = vector3 + vector2
#     print(vector3.shape, vector35.shape)
    return vector4


# @njit(nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:, :], nb.float64[:,:], nb.float64))
def matvec_Qinv(vector, D, WBI, VdivD, tol = 0):
    """
    Computes the matvec of (D + V V^T)^{-1} using the Woodbury matrix identity:
    (D + V V^T)^(-1) vector = D^(-1) - D^(-1) V (I + V^T D^(-1) V)^(-1) V^T D^(-1)
    
    Parameters:
    - D: (n, ) diagonal matrix part
    - WBI: (I + V^T D^(-1) V)^(-1)
    - VdivD: D^{-1} V
     
    OUTPUT:
    (D + V V^T)^(-1) vector
    """
    vector1 = VdivD.T @vector
    vector2 = WBI@vector1
    vector3 = VdivD@vector2
    
    vector35= (vector.T/(D + tol)).T
    vector4 = vector35 - vector3
#     print(vector3.shape, vector35.shape)
    return vector4


# @njit(nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:, :], nb.float64[:,:], nb.float64[:], nb.float64, nb.float64))
def matvec_Gbb0_inv(vector, D, WBI, VdivD, b, sign=1, tol = 0):
    """
    Mat Vec Multiplication for matrix
    [sign*G   b] <- N
    [b.T      0] <- 1
    
    b -- vector (N, 1)
    G = D + V V^T
    WBI = (I + V^T D^(-1) V)^(-1), (r, r)
    VdivD = D^{-1} V    (N, r)
    D -- diagonal (N, )
    """
    
    N = len(D)
    vector_part = vector[:N]
    scalar_part = vector[N]
    
    G_inv_b = matvec_Qinv(b, D, WBI, VdivD) #vector
#     print(G_inv_b.shape)
    b_G_inv_b_inv = 1/(b.T@G_inv_b + tol) # scalar
#     print(b_G_inv_b_inv.shape)
    
    
    G_inv_vector = matvec_Qinv(vector_part, D, WBI, VdivD) #vector
#     print(G_inv_vector.shape)
    b_G_inv_vector= b.T@G_inv_vector#scalar
#     print(b_G_inv_vector.shape)
    result11 = G_inv_vector - b_G_inv_b_inv*b_G_inv_vector*G_inv_b
    
    result12 = scalar_part*b_G_inv_b_inv*G_inv_b
    
    result21 = b_G_inv_b_inv*(G_inv_b.T@vector_part)
    
    result22 = -scalar_part*b_G_inv_b_inv
#     print(result11.shape, result12.shape, result21.shape, result22.shape)
    
    result1 = sign*result11 + result12
    result2 = result21 + sign*result22
    return result1, result2


def compute_largest_possible_step(var1, var2, var3, gvar1, gvar2, gvar3, low, upp, etol = 1.e-20):
    # Find Step Size for predictor step 
    # l <= var1 + step*gvar1 <= u
    # var2 + step*gvar2 >= 0
    # var3 + step*gvar3 >= 0
    def update_step_max(var1, gvar, indeces, etol):
        if len(indeces) == 0:
            step = -1
        else:
            step = np.max(-var1[indeces, :]/(gvar[indeces, :] + etol))
        return step
    
    def update_step_min(var1, gvar, indeces, etol):
        if len(indeces) == 0:
            step = 1
        else:
            step = np.min(-var1[indeces, :]/(gvar[indeces, :] + etol))
        return step
    
    tol = 1.e-6
    neg_var1 = np.argwhere(gvar1 < -etol)[:, 0]
    pos_var1 = np.argwhere(gvar1 >  etol)[:, 0] 
    
    neg_var2 = np.argwhere(gvar2 < -etol)[:, 0]
    pos_var2 = np.argwhere(gvar2 >  etol)[:, 0]
    
    neg_var3 = np.argwhere(gvar3 < -etol)[:, 0]
    pos_var3 = np.argwhere(gvar3 >  etol)[:, 0]

    step1_var2 = update_step_max(var2-etol, gvar2, pos_var2, etol)  
    step1_var3 = update_step_max(var3-etol, gvar3, pos_var3, etol)
    
    # variable 2 deacreasing: Find step_size such that
    #  var2 + step_size * gvar2 > tol
    step2_var2 = update_step_min(var2-etol, gvar2, neg_var2, etol) 
    # variable 3 deacreasing:
    step2_var3 = update_step_min(var3-etol, gvar3, neg_var3, etol) 
    
    
    
    step1_var1 = max(update_step_max(var1 - low, gvar1, pos_var1, etol),
                     update_step_max(var1 - upp, gvar1, neg_var1, etol))
    
#         np.max((low-var1[pos_var1, :])/gvar1[pos_var1, :]), 
#                      np.max((upp-var1[neg_var1, :])/gvar1[neg_var1, :])) 
    step2_var1 = min(update_step_min(var1 - low - etol, gvar1, neg_var1, etol),
                     update_step_min(var1 - upp + etol, gvar1, pos_var1, etol))
    
    #min(np.min((low-var1[neg_var1, :])/gvar1[neg_var1, :]),
                 #    np.min((upp-var1[pos_var1, :])/gvar1[pos_var1, :]))

#     print(step1_var1, step1_var2, step1_var3)
#     print(step2_var1, step2_var2, step2_var3)
     
    step1 = max(step1_var1, step1_var2, step1_var3)
    step2 = min(step2_var1, step2_var2, step2_var3)
    
    #CHECK 
    check1 = np.min(var1 - low + step2*gvar1) < -etol
    check2 = np.min(upp - var1 - step2*gvar1) < -etol
    check3 = np.min(var2 + step2*gvar2) < -etol
    check4 = np.min(var3 + step2*gvar3) < -etol
#     print(np.min(var1 + step2*gvar1 - l ), 
#           np.min(u - (var1 + step2*gvar1)),
#           np.min(var2 + step2*gvar2), 
#           np.min(var3 + step2*gvar3),
#          etol)
    
    
#     print(check1, check2, check3, check4)
    
    if (check1 or check2 or check3 or check4):
        print('ERROR: Wrong STEP SIZE!')
        print('UPDATE Function: compute_largest_possible_step ')
        
    
    if step1 > step2:
        print('ERROR: There is no step size!')
        
    return step2

def compute_largest_possible_step_v2(var1, var2, var3, gvar1, gvar2, gvar3, low, upp, etol = 1.e-60, tol = 1.e-5):
    # Find Step Size for predictor step using binary search
    # l <= var1 + step*gvar1 <= u
    # var2 + step*gvar2 >= 0
    # var3 + step*gvar3 >= 0
    
    step_min = 0
    step_max = 1 
    
    while (step_max-step_min)>tol:
        step = (step_max+step_min)/2
        var1_plus = var1 + step*gvar1
        var2_plus = var2 + step*gvar2
        var3_plus = var3 + step*gvar3
     
        #CHECK 
        check1 = np.min(var1_plus - low) < etol
        check2 = np.min(upp - var1_plus) < etol
        check3 = np.min(var2_plus)       < etol
        check4 = np.min(var3_plus)       < etol
        if (check1 or check2 or check3 or check4):
            step_max = step
        else:
            step_min = step
#     print(np.min(var1 + step2*gvar1 - l ), 
#           np.min(u - (var1 + step2*gvar1)),
#           np.min(var2 + step2*gvar2), 
#           np.min(var3 + step2*gvar3),
#          etol)
    step = step_min
    
    var1_plus = var1 + step*gvar1
    var2_plus = var2 + step*gvar2
    var3_plus = var3 + step*gvar3

    #CHECK 
    check1 = np.min(var1_plus - low) < etol
    check2 = np.min(upp - var1_plus) < etol
    check3 = np.min(var2_plus)     < etol
    check4 = np.min(var3_plus)     < etol
#     print(check1, check2, check3, check4)
    
    if (check1 or check2 or check3 or check4):
        print('ERROR: Wrong STEP SIZE!')
        print('UPDATE Function: compute_largest_possible_step ')
        
     
    return step