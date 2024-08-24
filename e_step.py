import numpy as np
from numpy.linalg import inv, slogdet
from helpers.assignopts import assignopts
from helpers.utils import velime_assemble_data, velime_x_index, multiprod, multitransp

'''

Based on Matlab code by Matt Golub, 2014.

@Harsha Gurnani, 2024
'''


def velime_buildM(estParams):
    """
    Constructs concatenated linear system so the unrolling of all prior
    expectations can be computed with the single evaluation: X = M*C + M0
    C has columns c_t = [p_{t-tau}; v_{t-tau-1}; u_{t-tau+1}; ... ; u_{t+1}]

    Parameters:
    estParams (dict): Dictionary containing the following parameters:
        - A (numpy.ndarray): cursor dynamics matrix
        - B (numpy.ndarray): BCI mapping matrix
        - b0 (numpy.ndarray): bias term
        - TAU (int): integral delay
        - dt (float): timestep (== 1)

    Returns:
    M (numpy.ndarray): Concatenated linear system matrix
    M0 (numpy.ndarray): Concatenated linear system offset vector
    """
    A = estParams['A']
    B = estParams['B']
    b0 = estParams['b0']
    TAU = estParams['TAU']
    dt = estParams['dt']

    vDim = A.shape[0]
    pDim = vDim
    xDim = pDim + vDim
    uDim = B.shape[1]

    # Build M0 = [0; b0; b0*dt; A*b0+b0; (A*b0+2*b0)*dt; ...]
    M0 = np.zeros(((TAU+1)*xDim, 1))
    M0[pDim:xDim, 0] = b0

    M = np.zeros(((TAU+1)*xDim, xDim + (TAU+1)*uDim))
    M[:pDim, :pDim] = np.eye(pDim)
    M[pDim:pDim+vDim, pDim:pDim+vDim+uDim] = np.hstack((A, B))
    col = xDim + uDim 
    for row in range(xDim, TAU*xDim+1, xDim):
        prev_p_idx = np.arange(row-xDim, row-vDim)
        prev_v_idx = np.arange(row-vDim, row)

        cur_p_idx = np.arange(row, row+pDim)
        cur_v_idx = np.arange(row+pDim, row+xDim)
        cur_x_idx = np.arange(row, row+xDim)

        # M
        prev_p_term = M[prev_p_idx, :]
        prev_v_term = M[prev_v_idx, :]
        M[cur_p_idx, :] = prev_p_term + prev_v_term*dt
        M[cur_v_idx, :] = A@prev_v_term
        M[cur_v_idx, col:col+uDim] = B
        col += uDim

        # M0
        prev_p_term = M0[prev_p_idx, 0]
        prev_v_term = M0[prev_v_idx, 0]
        M0[cur_x_idx, 0] = np.hstack((prev_p_term + prev_v_term*dt, A@prev_v_term + b0))

    return M, M0




def velime_estep(C, G, estParams, const):
    '''
    Computes the posterior distributions of [x_{t-tau}^t,...,x_t^t] given
    p_{t-tau-1}, p_{t-tau}, u_{t-tau+1}, ..., u_{t+1}, G_t.
    '''
    R = np.diag(estParams['R'])
    alpha = estParams['alpha']
    
    gDim = const['gDim']
    llconst = gDim * np.log(2 * np.pi)
    half = 1 / 2
    
    x_pt_idx = const['x_pt_idx']
    x_vt_idx = const['x_vt_idx']
    xt_idx = x_pt_idx + x_vt_idx
    
    E_X, SIGMA11, E_TARG = velime_prior_expectation(C, estParams)
    EXDim = E_X.shape[0]
    
    x2_minus_mu2 = G - E_TARG
    
    T = E_X.shape[1]
    I = np.eye(const['gDim'])
    
    if const['gDim'] == 2:
        SIGMA_pp = SIGMA11[np.ix_(x_pt_idx, x_pt_idx)] # cov(p_t^t)
        SIGMA_pv = SIGMA11[np.ix_(x_pt_idx, x_vt_idx)] # cov(p_t^t,v_t^t)
        SIGMA_vp = SIGMA11[np.ix_(x_vt_idx, x_pt_idx)] # cov(v_t^t,p_t^t)
        SIGMA_vv = SIGMA11[np.ix_(x_vt_idx, x_vt_idx)] # cov(v_t^t)
        SIGMA_Xp = SIGMA11[:, x_pt_idx] # cov(X, p_t^t)
        SIGMA_Xv = SIGMA11[:, x_vt_idx] # cov(X, v_t^t)
        
        term1 = np.tile(SIGMA_pp.flatten('F').reshape(-1,1), (1,T))  #np.tile(SIGMA_pp.flatten(), (T, 1)).T
        term2 = (np.tile((SIGMA_pv + SIGMA_vp).flatten('F').reshape(-1,1), (1,T))) * alpha  #(np.tile((SIGMA_pv + SIGMA_vp).flatten(), (T, 1)).T) * alpha
        term3 = (np.tile(SIGMA_vv.flatten('F').reshape(-1,1), (1,T))) * alpha**2 #(np.tile(SIGMA_vv.flatten(), (T, 1)).T) * alpha**2
        
        SIGMA22s = term1 + term2 + term3 + R.flatten('F')[:, None]
        detSIGMA22s = SIGMA22s[0, :] * SIGMA22s[3, :] - SIGMA22s[1, :] * SIGMA22s[2, :]
        logdetSIGMA22s = np.log(detSIGMA22s)
        
        inv_term = np.array([SIGMA22s[3, :], -SIGMA22s[1, :], -SIGMA22s[2, :], SIGMA22s[0, :]])
        invSIGMA22s = inv_term / detSIGMA22s
        
        XcXc = np.array([x2_minus_mu2[0, :]**2,
                         x2_minus_mu2[0, :] * x2_minus_mu2[1, :],
                         x2_minus_mu2[0, :] * x2_minus_mu2[1, :],
                         x2_minus_mu2[1, :]**2])
        
        LLi = -half * (T * llconst + np.sum(logdetSIGMA22s + np.sum(XcXc * invSIGMA22s, axis=0), axis=0))
        
        invSIGMA22_page = invSIGMA22s.reshape(gDim, gDim, T, order='F')
        SIGMA12s = np.tile(SIGMA_Xp.flatten('F').reshape(-1,1), (1, T)) + (np.tile(SIGMA_Xv.flatten('F').reshape(-1,1), (1, T)) * alpha)# np.tile(SIGMA_Xp.flatten(), (T, 1)).T + (np.tile(SIGMA_Xv.flatten(), (T, 1)).T * alpha)
        
        SIGMA12_paged = SIGMA12s.reshape(EXDim, gDim, T, order='F')
        SIGMA12_iSIGMA22_paged = np.zeros((SIGMA12_paged.shape[0], invSIGMA22_page.shape[1], invSIGMA22_page.shape[-1]))
        for jj in range(T):
            SIGMA12_iSIGMA22_paged[:, :, jj] = SIGMA12_paged[:, :, jj] @ invSIGMA22_page[:, :, jj]
        #SIGMA12_iSIGMA22_paged = multiprod(SIGMA12_paged, invSIGMA22_page)#np.einsum('ijk,ikl->ijl', SIGMA12_paged, invSIGMA22_page)
        
        mt2 = multitransp(SIGMA12_paged)
        SIGMA12_iSIGMA22_SIGMA21_paged = np.zeros((SIGMA12_iSIGMA22_paged.shape[0], mt2.shape[1], mt2.shape[-1]))
        for jj in range(T):
            SIGMA12_iSIGMA22_SIGMA21_paged[:, :, jj] = SIGMA12_iSIGMA22_paged[:, :, jj] @ mt2[:, :, jj]
        #SIGMA12_iSIGMA22_SIGMA21_paged = multiprod(SIGMA12_paged, multitransp(invSIGMA22_page))#p.einsum('ijk,ikl->ijl', SIGMA12_iSIGMA22_paged, np.transpose(SIGMA12_paged, (0, 2, 1)))
        
        COV_X_posterior = SIGMA11[:, :, None] - SIGMA12_iSIGMA22_SIGMA21_paged
        
        mt2 =  x2_minus_mu2.reshape(gDim,1,T, order='F')
        SIGMA12_iSIGMA22_X2_minus_mu2_paged = np.zeros((SIGMA12_iSIGMA22_paged.shape[0],mt2.shape[1],mt2.shape[-1]))
        for jj in range(T):
            SIGMA12_iSIGMA22_X2_minus_mu2_paged[:, :, jj] = SIGMA12_iSIGMA22_paged[:, :, jj] @ mt2[:, :, jj]
        #SIGMA12_iSIGMA22_X2_minus_mu2_paged = multiprod(SIGMA12_iSIGMA22_paged,  x2_minus_mu2.reshape(gDim,1,T) )#np.einsum('ijk,ikl->ijl', SIGMA12_iSIGMA22_paged, x2_minus_mu2[:, :, None])
        E_X_posterior = E_X + np.squeeze(SIGMA12_iSIGMA22_X2_minus_mu2_paged)
    else:
        LLi = 0
        E_X_posterior = np.full((EXDim, T), np.nan)
        COV_X_posterior = np.full((EXDim, EXDim, T), np.nan)
        
        for t in range(T):
            C_t = np.hstack((I, alpha[t] * I))
            SIGMA22 = C_t @ SIGMA11[np.ix_(xt_idx, xt_idx)] @ C_t.T + R
            invSIGMA22 = inv(SIGMA22)
            _, logdetSIGMA22 = slogdet(SIGMA22)
            
            LLi -= half * (llconst + logdetSIGMA22 + np.sum((x2_minus_mu2[:, t][:, None] @ x2_minus_mu2[:, t][None, :]) * invSIGMA22))
            
            SIGMA12 = SIGMA11[:, xt_idx] @ C_t.T
            SIGMA12_iSIGMA22 = SIGMA12 @ invSIGMA22
            COV_X_posterior[:, :, t] = SIGMA11 - SIGMA12_iSIGMA22 @ SIGMA12.T
            
            if 'E_X_posterior' in locals():
                E_X_posterior[:, t] = E_X[:, t] + SIGMA12_iSIGMA22 @ x2_minus_mu2[:, t]

    return LLi, E_X_posterior, COV_X_posterior




def velime_LL(data, estParams, **kwargs):
    """
    Compute the log-likelihood pf the data given a set of IME parameters.

    Args:
        data (numpy.ndarray): The input data.
        estParams (dict): The estimated parameters of the VELIME model.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        float: The log-likelihood of the VELIME model.

    Raises:
        ValueError: If T_START is less than TAU + 2.

    """
    
    T_START = kwargs.get( 'T_START', estParams['TAU'] + 2)

    TAU = estParams['TAU']
    dt = estParams['dt']
    R = np.diag(estParams['R'])
    alpha = estParams['alpha']

    C, G, const = velime_assemble_data(data, TAU, dt)
    gDim = const['gDim']
    llconst = gDim * np.log(2 * np.pi)

    x_pt_idx = const['x_pt_idx']
    x_vt_idx = const['x_vt_idx']

    # Step 1: Put together E(x_-{tau}^t,...,x_0^t, u_t | x_-{tau}^t, u_{t-tau+1},...u_{t-1})
    # and cov(x_-{tau}^t,...,x_0^t, u_t | x_-{tau}^t, u_{t-tau+1},...u_{t-1})
    E_X, SIGMA11, E_TARG = velime_prior_expectation(C, estParams)

    x2_minus_mu2 = G - E_TARG

    # Sigma_22 = cov(x^{target}_t | x_-{tau}^t, u_{t-tau+1},...u_t)
    # SIGMA12 = cov([x_-{tau+1}^t;...;x_0^t], x^{target}_t | x_-{tau}^t, u_{t-tau+1},...u_{t-1})

    T = E_X.shape[1]

    # Fast code
    SIGMA_pp = SIGMA11[np.ix_(x_pt_idx, x_pt_idx)] # cov(p_t^t)
    SIGMA_pv = SIGMA11[np.ix_(x_pt_idx, x_vt_idx)] # cov(p_t^t,v_t^t)
    SIGMA_vp = SIGMA11[np.ix_(x_vt_idx, x_pt_idx)] # cov(v_t^t,p_t^t)
    SIGMA_vv = SIGMA11[np.ix_(x_vt_idx, x_vt_idx)] # cov(v_t^t)
    
    # Build up terms to SIGMA22
    term1 = np.tile(SIGMA_pp.flatten('F').reshape(-1,1), (1, T))
    term2 = np.multiply(np.tile(SIGMA_pv.flatten('F').reshape(-1,1) + SIGMA_vp.flatten('F').reshape(-1,1), (1, T)), alpha)
    term3 = np.multiply(np.tile(SIGMA_vv.flatten('F').reshape(-1,1), (1, T)), np.power(alpha, 2))

    SIGMA22s = term1 + term2 + term3 + np.tile(R.flatten('F').reshape(-1,1), (1, T))
    # To recover SIGMA22: np.reshape(SIGMA22s[:, 0], (gDim, gDim))

    detSIGMA22s = SIGMA22s[0, :] * SIGMA22s[3, :] - SIGMA22s[1, :] * SIGMA22s[2, :]
    logdetSIGMA22s = np.log(detSIGMA22s)

    inv_term = np.array([SIGMA22s[3, :], -SIGMA22s[1, :], -SIGMA22s[2, :], SIGMA22s[0, :]])
    invSIGMA22s = inv_term / detSIGMA22s

    XcXc = np.array([x2_minus_mu2[0, :] ** 2,
                     x2_minus_mu2[0, :] * x2_minus_mu2[1, :],
                     x2_minus_mu2[0, :] * x2_minus_mu2[1, :],
                     x2_minus_mu2[1, :] ** 2])

    if 'T_START' in locals():
        T_trial = [len(t) for t in const['trial_map']]
        original_T_START = estParams['TAU'] + 2

        if T_START < original_T_START:
            raise ValueError('T_START must be >= TAU + 2')

        # T_START is nominally (TAU+2).  If user specifies larger T_START for
        # LL computation, then decrease T accordingly.
        T_adjusted = sum(max(0, t - (T_START - original_T_START)) for t in T_trial)
        trial_map_keep = [t[(T_START - original_T_START):] for t in const['trial_map']]
        idx_keep = np.concatenate(trial_map_keep)

        LLi = -(1 / 2) * (T_adjusted * llconst + np.sum(logdetSIGMA22s[idx_keep] + np.sum(XcXc[:, idx_keep] * invSIGMA22s[:, idx_keep], axis=0)))
    else:
        LLi = -(1 / 2) * (T * llconst + np.sum(logdetSIGMA22s + np.sum(XcXc * invSIGMA22s, axis=0)))

    return LLi



def velime_prior_expectation(C, estParams, **kwargs):
    """
    Compute the prior expectation of the VELIME model.

    Args:
        C (numpy.ndarray): The input data.
        estParams (dict): The estimated parameters of the VELIME model.

    Returns:
        tuple: A tuple containing the prior expectation of the VELIME model.
            - E_X (numpy.ndarray): The prior expectation of the state.
            - COV_X (numpy.ndarray): The prior covariance of the state.
            - E_TARG (numpy.ndarray): The prior expectation of the target.

    """
    
    COMPUTE_COV_X = kwargs.get('COMPUTE_COV_X', True)  # Assuming you always want to compute COV_X
    COMPUTE_E_TARG = kwargs.get('COMPUTE_E_TARG', True)  # Assuming you always want to compute E_TARG
    
    TAU = estParams['TAU']
    
    M, M0 = velime_buildM(estParams)
    E_X = M @ C + M0
    
    COV_X = None
    E_TARG = None
    
    if COMPUTE_COV_X:
        # COV_X = cov([x_{t-tau}^t;...;x_{t+1}^t] | y_{t-tau}, u_{t-tau},...u_{t})
        A = estParams['A']
        W_p = estParams['W_p']
        W_v = estParams['W_v']
        dt = estParams['dt']
        
        vDim = A.shape[0]
        pDim = vDim
        xDim = vDim + pDim
        
        M1 = np.block([[np.eye(pDim), np.eye(vDim) * dt],
                       [np.zeros((pDim, pDim)), A]])
        W = np.block([
            [np.diag(np.squeeze(W_p)), np.zeros((pDim, vDim))],
            [np.zeros((vDim, pDim)), np.diag(np.squeeze(W_v))]
        ])
        
        COV_X = np.zeros(((TAU + 1) * xDim, (TAU + 1) * xDim))
        
        K, x_idx, _ = velime_x_index(TAU, xDim)
        
        # First compute block diagonal components
        COV_X[:pDim, :pDim] = 0  # position feedback is observed
        COV_X[pDim:xDim, pDim:xDim] = np.diag(W_v)  # first velocity in state is not observed
        for k in range(-TAU+1, 1):
            xkm1_idx = x_idx[:, K == k - 1].flatten('F')
            xk_idx = x_idx[:, K == k].flatten('F')
            COV_X[np.ix_(xk_idx, xk_idx)] = M1 @ COV_X[np.ix_(xkm1_idx, xkm1_idx)] @ M1.T + W
        
        # Now compute off diagonals
        for k1 in range(-TAU, 1): # first velocity in state is NOT observed
            xk1_idx = x_idx[:, K == k1].flatten('F')
            for k2 in range(k1 + 1, 1):
                xk2_idx = x_idx[:, K == k2].flatten('F')
                xk2m1_idx = x_idx[:, K == k2 - 1].flatten('F')
                COV_X[np.ix_(xk1_idx, xk2_idx)] = COV_X[np.ix_(xk1_idx, xk2m1_idx)] @ M1.T
                COV_X[np.ix_(xk2_idx, xk1_idx)] = COV_X[np.ix_(xk1_idx, xk2_idx)].T
    
    if COMPUTE_E_TARG and 'alpha' in estParams.keys():
        # This computes E(G_t | y_{t-tau}^t, u_{t-tau},...u_{t})
        xPosIdx = np.arange(pDim)  
        xVelIdx = np.arange(pDim, xDim)  
        E_TARG = E_X[np.ix_(x_idx[xPosIdx, K == 0].flatten('F'),)] + E_X[np.ix_(x_idx[xVelIdx, K == 0].flatten('F'),)] * estParams['alpha']
    else:
        E_TARG = None
        
    return E_X, COV_X, E_TARG