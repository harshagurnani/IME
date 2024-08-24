import numpy as np
import sys
sys.path.insert(0, '/Users/harshagurnani/Documents/GitHub/VAEPytorch/IME')
from helpers.utils import *

def fast_ime_mstep_MParams(E_X_posterior, COV_X_posterior, const):
    """
    Update the parameters M1, M2, m0, and V for the internal model estimation.
    
    Args:
        E_X_posterior (numpy.ndarray): Posterior estimates of the internal states.
        COV_X_posterior (numpy.ndarray): Covariance matrices of the internal states.
        const (dict): Dictionary containing constant parameters.
        
    Returns:
        dict: Dictionary containing the updated parameters M1, M2, m0, and V.
    """
    
    EXDim, T = E_X_posterior.shape
    xDim = const['xDim']
    uDim = const['uDim']
    
    # Determine number of relevant timesteps in each trial for internal state estimates
    num_internal_states_per_t = EXDim // xDim - 1
    N_x = num_internal_states_per_t * T
    
    # Sums across state chains
    sum_COV_X = np.sum(COV_X_posterior, axis=2)
    
    sumk_cov_xkxk = np.zeros((xDim, xDim))
    sumk_cov_xkxkm1 = np.zeros((xDim, xDim))
    sumk_cov_xkm1xkm1 = np.zeros((xDim, xDim))
    
    for i in range(0, EXDim - xDim, xDim):
        idx_xkm1 = slice(i, i + xDim)
        idx_xk = slice(i + xDim, i + 2 * xDim)
        sumk_cov_xkxkm1 += sum_COV_X[idx_xk, idx_xkm1]
        sumk_cov_xkm1xkm1 += sum_COV_X[idx_xkm1, idx_xkm1]
        sumk_cov_xkxk += sum_COV_X[idx_xk, idx_xk]
    
    sumk_cov_akak = np.block([
        [sumk_cov_xkm1xkm1, np.zeros((xDim, uDim + 1))],
        [np.zeros((uDim, xDim + uDim + 1))],
        [np.zeros((1, uDim + xDim + 1))]
    ])
    
    # Gather sufficient statistics from E-step
    E_xk = np.zeros((xDim, T * num_internal_states_per_t))
    E_xkm1 = np.zeros((xDim, T * num_internal_states_per_t))
    
    j = 0
    for i in range(0, EXDim - xDim, xDim):
        idx_xkm1 = slice(i, i + xDim)
        idx_xk = slice(i + xDim, i + 2 * xDim)
        t_idx = slice(T * j, T * (j + 1))
        E_xk[:, t_idx] = E_X_posterior[idx_xk, :]
        E_xkm1[:, t_idx] = E_X_posterior[idx_xkm1, :]
        j += 1
    
    E_a = np.vstack((E_xkm1, const['Urep'], np.ones((1, T * num_internal_states_per_t))))
    sum_Ex_Ea = E_xk @ E_a.T
    temp = np.hstack((E_xkm1 @ const['Urep'].T, np.sum(E_xkm1, axis=1, keepdims=True)))
    sum_Ea_Ea = np.block([
        [E_xkm1 @ E_xkm1.T, temp],
        [temp.T, const['sum_U1_U1']]
    ])
    
    # M variables update (AND V)
    sum_E_aa = sumk_cov_akak + sum_Ea_Ea
    sum_E_xa = np.hstack((sumk_cov_xkxkm1, np.zeros((xDim, uDim + 1)))) + sum_Ex_Ea
    
    # Update for M1, M2 and m0
    # x_t = M*a_t, a_t = [x_{t-1}; u{t}; 1]
    M = np.linalg.solve(sum_E_aa, sum_E_xa.T).T  #np.linalg.solve(sum_E_aa, sum_E_xa.T).T  #sum_E_aa is symmetric
    estParams = {}
    estParams['M1'] = M[:, :xDim]
    estParams['M2'] = M[:, xDim:xDim + uDim]
    estParams['m0'] = M[:, -1]
    
    # Update for V
    estParams['V'] = (1 / N_x) * (np.diag(sumk_cov_xkxk) + diagProduct(E_xk, E_xk.T) - diagProduct(M, sum_E_xa.T))
    
    return estParams


def velime_mstep_alphaParams(G, E_X_posterior, COV_X_posterior, const):
    """
    Estimate the alpha parameters and R vector for VE-LIME algorithm.

    Args:
        G (numpy.ndarray)
        E_X_posterior (numpy.ndarray): The posterior mean of the latent variables.
        COV_X_posterior (numpy.ndarray): The posterior covariance of the latent variables.
        const (dict): A dictionary containing constant values.

    Returns:
        dict: A dictionary containing the estimated alpha parameter and R vector.

    """
    
    pt_idx = const['x_pt_idx']
    vt_idx = const['x_vt_idx']
    xt_idx = const['x_idx'][:, -1]
    gDim = const['gDim']
    
    #E_xt = E_X_posterior[xt_idx, :]
    E_pt = E_X_posterior[pt_idx, :]
    E_vt = E_X_posterior[vt_idx, :]
    
    #COV_xt = COV_X_posterior[xt_idx, xt_idx, :]
    COV_ptpt = COV_X_posterior[pt_idx][:, pt_idx, :]
    COV_ptvt = COV_X_posterior[pt_idx][:, vt_idx, :]
    COV_vtvt = COV_X_posterior[vt_idx][:, vt_idx, :]
    
    T = const['T']
    
    # Indices of the diagonal elements of a xDim x xDim subblock of COV_X_posterior
    SLICE_DIAG_IDX = np.arange(0, gDim * gDim, gDim + 1)
    
    # Diagonal indices of each slice of a 3-D vector for trace covariance computations
    DIAG_IDX = np.add.outer(SLICE_DIAG_IDX, np.arange(0, T * gDim * gDim - 1, gDim * gDim))#.flatten()
    #if max(DIAG_IDX) >= COV_ptpt.size:
    #    DIAG_IDX = DIAG_IDX[:-1]
    
    cpt = COV_ptpt.flatten('F')
    cvt = COV_vtvt.flatten('F')
    cptvt = COV_ptvt.flatten('F')
    trace_E_pp = np.sum(cpt[DIAG_IDX] + E_pt*E_pt, axis=0)
    trace_E_pv = np.sum(cptvt[DIAG_IDX] + E_pt * E_vt, axis=0)
    trace_E_vv = np.sum(cvt[DIAG_IDX] + E_vt * E_vt, axis=0)
    trace_E_Gv = np.sum(E_vt * G, axis=0)
    
    alpha = (trace_E_Gv - trace_E_pv) / trace_E_vv
    
    # Constrain alphas to be non-negative
    alpha = np.maximum(alpha, 0) # for each timepoint
    
    term1 = np.sum(G * G, axis=0)
    term2 = -2 * np.sum(G * (E_pt + E_vt * alpha), axis=0)
    term3 = trace_E_pp + 2 * alpha * trace_E_pv + (alpha ** 2) * trace_E_vv
    r = 1 / (T * gDim) * np.sum(term1 + term2 + term3, axis=0)
    R = r * np.ones(gDim)
    
    estParams = {'alpha': alpha, 'R': R}
    
    return estParams


def velime_mstep_MParams(E_X_posterior, COV_X_posterior, C, const):
    """
    Estimate the model parameters for the VE-LIME algorithm.

    Args:
        E_X_posterior (numpy.ndarray): Posterior mean of the latent positions.
        COV_X_posterior (numpy.ndarray): Posterior covariance of the latent positions.
        C (numpy.ndarray): Observation matrix.
        const (dict): Dictionary containing constant parameters.

    Returns:
        dict: Dictionary containing the estimated model parameters.

    Notes:
        This function estimates the model parameters for VELIME. 
        It concatenates the velocity feedback into the latent positions and
        performs the estimation using the fast_ime_mstep_MParams function.

    """
    
    # Velocity feedback is not maintained in E_X, so first we concatenate it
    # into E_X for this portion of code only.  It might be cleaner to add it in
    # permanently, but then we'd carry around 2 timesteps of position feedback.
    
    # MAGIC NUMBERS
    C_v_idx = [2, 3]
    vDim = 2
    V_idx = const['x_idx'][C_v_idx, :]
    
    EVDim = V_idx.size
    
    E_V_posterior = np.vstack((C[C_v_idx, :], E_X_posterior[V_idx.flatten('F'), :]))
    
    shift_idx = list(range(vDim, EVDim + vDim))
    COV_V_posterior = np.zeros((EVDim + vDim, EVDim + vDim, COV_X_posterior.shape[2]))
    COV_V_posterior[np.ix_(shift_idx, shift_idx)] = COV_X_posterior[np.ix_(V_idx.flatten('F'), V_idx.flatten('F'))]
    
    # const['xDim'] is overwritten since we are leveraging fastfmc code
    const['xDim'] = const['vDim']
    estParams = fast_ime_mstep_MParams(E_V_posterior, COV_V_posterior, const)
    estParams['V'] = np.mean(estParams['V']) * np.ones((const['vDim'], 1))
    
    return estParams