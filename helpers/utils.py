import numpy as np
from helpers.assignopts import assignopts
import sys
sys.path.insert(0, '../')
#from scipy.stats import mode
from statistics import mode

'''
Helper functions for IME.
Based on Matlab code by Matt Golub, 2014.

@Harsha Gurnani, 2024
'''

def diagProduct(A, B):
    """
    diagonal = diagProduct(A, B)
    Efficient computation of diag(A@B) when A@B is square, 
    ie A is DxM, B is MxD.
    """
    return np.sum(A * B.T, axis=1)


def angular_error(v1, v2):
    """
    Vectors are rows of v1 and v2. Both inputs must be the same size.
    """
    
    if v1.shape != v2.shape:
        raise ValueError("Inputs are not the same size")
    
    error_is_counterclockwise = np.sign(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    v1norm = np.linalg.norm(v1, axis=1)
    v2norm = np.linalg.norm(v2, axis=1)
    
    dot_prods = np.sum(v1 * v2, axis=1)
    degs = np.rad2deg(np.arccos(dot_prods / (v1norm * v2norm)))
    degs[error_is_counterclockwise == 1] = -degs[error_is_counterclockwise == 1]
    
    return np.real(degs)


def angular_error_from_perimeter(x_t_t, x_tp1_t, target_center, radius):
    """
    [angles, theta, phi] = angular_error_from_perimeter(x_t_t, x_tp1_t, target_center, radius)
    
    INPUTS:
    x_t_t and x_tp1_t are DxN, N = number of time points, D = internal state
    dimensionality. Only position elements (0:1,:) are looked at.
    
    target_center is 2x1, radius is 1x1
    
    OUTPUTS: 
    - angles is Nx1 signed angular errors from the target perimeter, i.e., the
    minimal angular rotation of the aiming direction such that rotating
    (x_tp1_t - x_t_t) about x_t_t would intersect with a point, p, on the
    target perimeter
    - theta is the angular error between the aiming direction and (target_center - x_t_t)
    - phi is the max angle between (target_center - x_t_t) and a perimeter point
    
    """
    
    D, N = x_t_t.shape
    D2, N2 = x_tp1_t.shape
    
    if N != N2 or D != D2 or target_center.size != 2 or np.isscalar(radius) is False:
        raise ValueError("Inputs are not formatted correctly")
    if target_center.shape[0] != 2 :
        target_center = target_center.reshape(-1, 1)

    pos_idx = [0, 1]

    v1 = target_center - x_t_t[pos_idx, :]
    v2 = x_tp1_t[pos_idx, :] - x_t_t[pos_idx, :]
    v1_dot_v2 = np.sum(v1 * v2, axis=0)
    
    error_is_counterclockwise = np.sign(v1[0, :] * v2[1, :] - v1[1, :] * v2[0, :])
    length_v1 = np.sqrt(np.sum(v1 ** 2, axis=0))
    length_v2 = np.sqrt(np.sum(v2 ** 2, axis=0))
    
    # Find the angle between v1 and v2
    cos_theta = v1_dot_v2 / (length_v1 * length_v2)
    theta = np.rad2deg(np.arccos(cos_theta))
    theta[error_is_counterclockwise == -1] = -theta[error_is_counterclockwise == -1]
    
    # Find the largest angle between v1 and v3,
    # where v1 is x_t_t to target_center, and
    # v3 is x_t_t to a point on the circle perimeter
    phi = np.rad2deg(np.arcsin(radius / length_v1)) * np.sign(theta) #closest point on perimeter is at the tangent line
    phi[length_v1 < radius] = 180
    
    radius_sq = radius ** 2
    
    # theta(n) and phi(n) must be the same sign.
    idx_zero = np.abs(theta) < np.abs(phi)  # aiming direction would bring cursor to target
    
    angles = np.full(N, np.nan)
    angles[idx_zero] = 0
    angles[~idx_zero] = theta[~idx_zero] - phi[~idx_zero]
    
    return angles, theta, phi



def target_regression(data):
    """
    Perform target regression analysis on spike count and cursor position data.

    Args:
        data (dict): A dictionary containing the following keys:
            - 'spike_counts' (list of numpy arrays): Spike count data for each trial.
            - 'cursor_position' (list of numpy arrays): Cursor position data for each trial.
            - 'target_position' (list of numpy arrays): Target position data for each trial.

    Returns:
        tuple: A tuple containing the following elements:
            - M2 (numpy array): The regression coefficients.
            - m0 (numpy array): The intercept term.
            - V (numpy array): The variance of the regression coefficients.

    Initialize M2, m0 by fitting
    u_t = M2*d_t + m0 + v_t, v_t ~ N(0,V),
    where d_t is a unit vector pointing from the center of the workspace to 
    the target, scaled by the mean distance traveled per timestep.
    """
    
    U = data['spike_counts']
    X = data['cursor_position']
    Xtarget = data['target_position']

    xDim = X[0].shape[0]
    num_trials = len(U)
    T = [u.shape[1] for u in U]

    #distances = [np.sqrt(np.sum(np.diff(x, axis=1)**2, axis=0)) for x in X]  #delta x
    #mean_distance_per_timestep = np.mean(np.concatenate(distances))
    mdpt = mean_distance_per_timestep(X)

    target_direction = [None] * num_trials
    intended_velocity = [None] * num_trials
    intended_kinematics = [None] * num_trials

    for trialNo in range(num_trials):
        # Unit vectors pointing from center to target
        target_direction[trialNo] = np.tile(Xtarget[trialNo], (1, T[trialNo]))
        target_norm = np.sqrt(np.sum(target_direction[trialNo]**2, axis=0))
        target_direction[trialNo] = target_direction[trialNo] / target_norm

        # Scale unit vectors so lengths are the mean distance traveled per timestep
        intended_velocity[trialNo] = mdpt * target_direction[trialNo]
        intended_kinematics[trialNo] = np.vstack([intended_velocity[trialNo], np.zeros((xDim-2, intended_velocity[trialNo].shape[1]))])

    M2, m0, V = regress_with_delay(intended_kinematics, U, d=0)   # based on current position
    V = np.diag(V)

    return M2, m0, V


def regress_with_delay(u, x, d):
    """
    Fits {L, l0, W} according to the following regression equation:
    u(t) = L*x(t-d) + l0 + w_t, w_t ~ N(0,W)

    INPUTS:
    u and x are lists of arrays, where u[i] and x[i] have the same number of columns (timepoints).
    d is an integer timestep delay
    """
    if isinstance(u, np.ndarray):
        u = [u]
        x = [x]

    num_trials = len(u)

    sizeX1 = [xi.shape[0] for xi in x]
    sizeX2 = [xi.shape[1] for xi in x]
    sizeU1 = [ui.shape[0] for ui in u]
    sizeU2 = [ui.shape[1] for ui in u]

    # Check for appropriate input dimensionalities
    if (all(np.array(sizeX1) == np.array(sizeU1)) and
        len(set(sizeX2)) == 1 and
        len(set(sizeU2)) == 1 and
        not all(np.array(sizeX2) == np.array(sizeU2))):
        # x and u have the same number of rows (timepoints)
        # all xi  have the same dimensionality (columns=pos,vel,etc)
        # all ui  have the same dimensionality (columns=spike counts)
        u = [ui.T for ui in u]  #make it so that u and x have the same number of columns
        x = [xi.T for xi in x]
    elif (all(np.array(sizeX2) == np.array(sizeU2)) and
          len(set(sizeX1)) == 1 and
          len(set(sizeU1)) == 1):
        # xi and ui have the same number of columns
        # all xi have the same dimensionality
        # all ui have the same dimensionality
        pass  # do nothing
    elif (all(np.array(sizeX1) == np.array(sizeU1)) and
          all(np.array(sizeX2) == np.array(sizeU2))):
        # Ambiguous which dimension is what in list of arrays
        # Proceed as if observations are rows of each matrices
        pass
    else:
        raise ValueError("Invalid input cell array(s)")

    U, X = list2Array_forLaggedRegression(u, x, d)

    # Append ones for bias term (i.e. to account for mean spike counts)
    if X.size == 0:
        # If no predictors/regressors/features are given, then this regression
        # will only return l0, i.e. the data mean: l0 = mean(U,2)
        X = np.ones((1, U.shape[1]))
        # --> X*X' = N
        # --> U*X' = sum(U,2)
        # --> l0 = sum(U,2)/N = mean(U,2)
    else:
        X = np.vstack((X, np.ones((1, X.shape[1]))))

    Ll0 = (U @ X.T) @ np.linalg.inv(X @ X.T)

    l0 = Ll0[:, -1] # bias term
    L = Ll0[:, :-1] # regression coefficients

    residual = U - Ll0 @ X

    N = residual.shape[1]
    W = (residual @ residual.T) / N

    return L, l0, W

def list2Array_forLaggedRegression(u, x, d):
    """
    Converts lists u and x into matrices suitable for lagged regression
    considering a delay d.
    """
    if d>0:    # x(t-d) -> u(t), x lags behind u
        U = np.hstack([ui[:, d:] for ui in u])
        X = np.hstack([xi[:, :-d] for xi in x])
    elif d==0:
        U = np.hstack([ui for ui in u])
        X = np.hstack([xi for xi in x])
    else:       # x(t) -> u(t+d), x leads u
        d = np.abs(d)
        U = np.hstack([ui[:, :-d] for ui in u])
        X = np.hstack([xi[:, d:] for xi in x])

    return U, X

def current_regression(data):
    """
    For position only state: initialize M2, m0 by assuming M1 = I and assume
    the subject aims from the current cursor position straight to the target
    with a constant speed corresponding to the mean distance per timestep.
    
    Parameters:
    - data: Dictionary with keys 'spike_counts', 'cursor_position', 'target_position'.
    
    Returns:
    - M1: Identity matrix.
    - M2: Regression matrix.
    - m0: Offset.
    - V: Diagonal matrix of noise variances.
    - mdpt: Mean distance per timestep.
    """
    U = data['spike_counts']
    X = data['cursor_position']
    Xtarget = data['target_position']

    POS_IDX = slice(0, 2)  # Adjust this if needed

    num_trials = len(U)

    mdpt = mean_distance_per_timestep(X)

    straight2target = [None] * num_trials
    straight2target_direction = [None] * num_trials
    intended_velocity = [None] * num_trials

    for trialNo in range(num_trials):
        # Unit vectors pointing from Xt to target
        straight2target[trialNo] = Xtarget[trialNo] - X[trialNo][POS_IDX, :]
        straight2target_norm = np.linalg.norm(straight2target[trialNo], axis=0)
        straight2target_direction[trialNo] = straight2target[trialNo] / straight2target_norm
        
        # Scale unit vectors so lengths are the mean distance traveled per timestep
        intended_velocity[trialNo] = mdpt * straight2target_direction[trialNo]
    
    M2, m0, V = regress_with_delay(intended_velocity, U, d=-1)  # x_t not yet displayed when u_t recorded
    M1 = np.eye(len(m0))
    V = np.diag(V)
    
    return M1, M2, m0, V, mdpt


def mean_distance_per_timestep(X):
    """
    Calculate the mean distance per timestep from cursor positions.
    
    Parameters:
    - X: List of arrays where each array is [2 x num_timesteps].
    
    Returns:
    - mean_distance: Mean distance traveled per timestep.
    """
    distances = [np.sqrt(np.sum(np.diff(x[:2, :], axis=1)**2, axis=0)) for x in X]
    distances = np.concatenate(distances)
    return np.mean(distances)








def velime_assemble_data(data, TAU, dt):
    """
    Produce structured data matrices required for the EM algorithm.
    
    INPUTS:
    data: Dictionary containing 'spike_counts', 'cursor_position', and 'target_position'.
    TAU: Integer specifying the delay.
    dt: Scalar specifying the time step.

    OUTPUTS:
    C: Matrix with columns c_t = [p_{t-tau}; v_{t-tau-1}; u_{t-tau+1}; ... ; u_{t+1}]
                                ( where v_{t-tau-1} = (p_{t-tau} - p_{t-tau-1})/dt )
    G: Matrix with columns g_t = target position from appropriate trial
    const: Dictionary with various constants and indices
    """
    U = data['spike_counts']
    P = data['cursor_position']
    TARGETS = data['target_position']
    
    t_start = TAU + 1  # "whiskers" are only defined for x_k^t with t >= TAU + 1
    num_trials = len(U)
    T = [u.shape[1] for u in U]

    xDim = 4
    gDim = 2
    uDim = mode([u.shape[0] for u in U])  # Mode of the number of rows in U
    EXDim = (TAU + 1) * xDim  #expected x for each of the whiskers = Tau + 1 timesteps

    # Write this, give xMap: one string for each set of indices into x
    K, x_idx, _ = velime_x_index(TAU, xDim)

    T_valid = np.sum(np.maximum(0, np.array(T) - (TAU + 2)))
    C = np.zeros((xDim + (TAU + 1) * uDim, T_valid))
    G = np.zeros((gDim, T_valid))

    const = {'trial_map': [None] * num_trials}

    idx = 0
    for trialNo in range(num_trials):
        trial_idx_range = np.arange(idx, idx + T[trialNo] - t_start - 1)
        const['trial_map'][trialNo] = trial_idx_range
        
        V_trial = np.diff(P[trialNo], axis=1) / dt  # velocities computed from differences in positions
        for t in range(t_start, T[trialNo]-1):
            U_seq = U[trialNo][:, t - TAU+1:t + 2].flatten('F')
            # Available from feedback: p(t-tau) and v(t-tau-1) = p(t-tau)-p(t-tau-1)
            # (using convention: p(t) = p(t-1) + v(t-1)dt)
            # Sanity check, these should be identical:
            # V_trial(:,t-TAU-1)
            # (P{trialNo}(:,t-TAU)-P{trialNo}(:,t-TAU-1))/dt
            C[:, idx] = np.hstack([
                P[trialNo][:, t - TAU],
                V_trial[:, t - TAU - 1],
                U_seq
            ])
            G[:, idx] = TARGETS[trialNo][:, 0]
            idx += 1

    const.update({
        'T': T_valid,
        'dt': dt,
        'uDim': uDim,
        'gDim': gDim,
        'xDim': xDim,
        'pDim': 2,
        'vDim': 2,
        'EXDim': EXDim,
        'K': K,
        'x_idx': x_idx,
        'x_pt_idx': np.arange(EXDim - 4, EXDim - 2),
        'x_vt_idx': np.arange(EXDim - 2, EXDim)
    })

    # Here TAU + 1 gives the number of timesteps worth of u's that contribute to each latent state chain, x.
    Urep = np.zeros((uDim, T_valid * (TAU + 1)))
    for j in range(TAU + 1):
        idx_u = np.arange(xDim + uDim * j, xDim + uDim * (j + 1))  # rows
        idx_t = slice(T_valid * j, T_valid * (j + 1))  # columns
        Urep[:, idx_t] = C[idx_u, :]
    const['Urep'] = Urep

    sum_Urep = np.sum(Urep, axis=1)
    const['sum_U1_U1'] = np.vstack([
        np.hstack([Urep @ Urep.T, sum_Urep[:, np.newaxis]]),
        np.hstack([sum_Urep[np.newaxis,:], np.array(T_valid * (TAU + 1)).reshape(-1,1)])
    ])      # uDim+1 x uDim+1

    return C, G, const

def velime_x_index(TAU, xDim):
    """
    Generate indices for the X matrix based on TAU and xDim.
    
    INPUTS:
    TAU: Integer specifying the delay.
    xDim: Dimensionality of the x vector.

    OUTPUTS:
    K: Indices for X matrix.
    x_idx: Index map for the X matrix.
    var_names:  variable names for each index 
    """
    K = np.arange(-TAU, 1)
    x_idx = np.arange(0, (TAU + 1) * xDim ).reshape(xDim, TAU + 1, order='F')

    var_names = None
    var_names = np.empty((2, TAU + 1), dtype=object)
    k = min(K)
    for i in range(TAU):
        var_names[0, i] = f'p{{t{k}}}'
        var_names[1, i] = f'v{{t{k}}}'
        k += 1
    var_names[0, i + 1] = 'p_t'
    var_names[1, i + 1] = 'v_t'

    return K, x_idx, var_names





def subsample_trials(data, trial_idx):
    """
    Returns a subsampled dataset containing only the trials specified in trial_idx.
    
    Parameters:
    - data: Dictionary containing keys 'spike_counts', 'cursor_position', 'target_position',
            'cursor_radius', and 'target_radius'.
    - trial_idx: List or array of indices specifying which trials to include.
    
    Returns:
    - sub_data: Dictionary with subsampled data, containing 'cursor_position', 'spike_counts',
                'target_position', 'cursor_radius', and 'target_radius'.
    """
    sub_data = {
        'cursor_position': [data['cursor_position'][i] for i in trial_idx],
        'spike_counts': [data['spike_counts'][i] for i in trial_idx],
        'target_position': [data['target_position'][i] for i in trial_idx],
        'cursor_radius': data['cursor_radius'],
        'target_radius': data['target_radius']
    }
    return sub_data



def generate_shuffled_blocks(trials_targets):
    """
    Generates sets of non-overlapping trial indices with matched target distributions for use in cross-validation.

    Parameters:
    - trials_targets: List of arrays, where each array is a column vector indicating a target for that trial.

    Returns:
    - block_trial_indices: List of lists containing random, non-overlapping sets of trial indices.
    - block_targets: List of lists containing the target IDs of the trials indexed by block_trial_indices.
    """
    # Convert list of arrays to a single array and find unique targets and their indices
    all_targets = np.concatenate(trials_targets, axis=1).T
    unique_targets, trials_target_idx = np.unique(all_targets, return_inverse=True, axis=0)

    # Find all trials with each target
    num_targets = unique_targets.shape[0]
    trials_idx_per_target = [np.where(trials_target_idx == t)[0] for t in range(num_targets)]
    num_target_repetitions = np.array([len(trials) for trials in trials_idx_per_target])

    # Shuffle trial indices for each target
    shuffled_trials_idx_per_target = [
        np.random.permutation(trials_idx) for trials_idx in trials_idx_per_target
    ]

    max_num_trials_per_target = max(num_target_repetitions)

    # Pad shuffled trial indices with NaNs to make arrays of the same length
    shuffled_trials_idx_per_target_padded = [
        np.concatenate([trials, np.full(max_num_trials_per_target - len(trials), np.nan)])
        for trials in shuffled_trials_idx_per_target
    ]
    trials_idx_mat = np.array(shuffled_trials_idx_per_target_padded).T  #ntrials x ntargets

    # Create blocks of trial indices
    block_trial_indices = []
    block_targets = []

    for block_idx in range(trials_idx_mat.shape[0]):
        temp = trials_idx_mat[block_idx,:]
        valid_indices = ~np.isnan(temp)
        block_indices = temp[valid_indices].astype(int)
        block_trial_indices.append(block_indices.tolist())
        block_targets.append(trials_target_idx[block_indices].tolist())

    return block_trial_indices, block_targets



def compute_average_absolute_angle(P_tt, P_ttp1, G, radius):
    """
    Compute the average absolute angular error.
    
    Parameters:
    - P_tt: Array of estimated positions.
    - P_ttp1: Array of estimated positions at the next timestep.
    - G: Array of target positions.
    - radius: Radius for the acceptance zone.
    
    Returns:
    - error_angle: Average absolute angular error.
    """
    error_angles_from_perimeter, _, _ = angular_error_from_perimeter(P_tt, P_ttp1, G, radius)
    error_angle = np.mean(np.abs(error_angles_from_perimeter))
    
    return error_angle

def compute_angular_errors(data, E_P, E_V, estParams, T_START):
    """
    Compute the angular errors for cursor and model predictions.
    
    Parameters:
    - data: Dictionary with keys 'spike_counts', 'cursor_position', 'target_position'.
    - E_P: List of arrays containing internal estimates of cursor position.
    - E_V: List of arrays containing internal estimates of velocity.
    - estParams: Dictionary containing model parameters.
    - T_START: Integer timestep to begin evaluating the data.
    
    Returns:
    - errors: Dictionary with 'model' and 'cursor' keys containing vectors of angular errors.
    """
    N_DIMS = estParams['A'].shape[0]
    TAU = estParams['TAU']
    DT = estParams['dt']

    acceptance_zone_radius = data['cursor_radius'] + data['target_radius']
    p_t_t_idx = (TAU + 1) * N_DIMS + np.array([-2, -1])
    v_t_t_idx = p_t_t_idx
    num_trials = len(data['cursor_position'])

    errors = {
        'model': np.full(num_trials, np.nan),
        'cursor': np.full(num_trials, np.nan)
    }

    for trial_idx in range(num_trials):
        G = data['target_position'][trial_idx]
        
        P_t = data['cursor_position'][trial_idx]
        P_tp1 = np.hstack([data['cursor_position'][trial_idx][:, 1:], np.nan * np.zeros((N_DIMS, 1))])
        
        V_tilde_tt = E_V[trial_idx][v_t_t_idx, :]
        P_tilde_tt = E_P[trial_idx][p_t_t_idx, :]
        P_tilde_ttp1 = P_tilde_tt + V_tilde_tt * DT
        
        valid_t = ~(np.isnan(P_t).any(axis=0) | np.isnan(P_tp1).any(axis=0) |
                    np.isnan(P_tilde_tt).any(axis=0) | np.isnan(P_tilde_ttp1).any(axis=0))
        valid_t[:T_START-1] = False
        
        errors['model'][trial_idx] = compute_average_absolute_angle(P_tilde_tt[:, valid_t], P_tilde_ttp1[:, valid_t], G, acceptance_zone_radius)
        errors['cursor'][trial_idx] = compute_average_absolute_angle(P_t[:, valid_t], P_tp1[:, valid_t], G, acceptance_zone_radius)
    
    return errors


def multiprod(a, b, idA=None, idB=None):
    # Default idA and idB values
    if idA is None and idB is None:
        idA = [1, 2]
        idB = [1, 2]
    elif idB is None:
        idB = idA

    # ESC 1 - Special simple case (both A and B are 2D), solved using C = A * B
    if a.ndim == 2 and b.ndim == 2 and idA == [1, 2] and idB == [1, 2]:
        return np.dot(a, b)

    # MAIN 0 - Checking and evaluating array size, block size, and IDs
    if isinstance(a, (int, float)): #is scalar
        a = np.array([[a]])
        b =  np.array([[b]])

    sizeA0 = a.shape
    sizeB0 = b.shape
    
    sizeA, sizeB, shiftC, delC, sizeisnew, idA, idB, squashOK, sxtimesOK, timesOK, mtimesOK, sumOK = sizeval(idA, idB, sizeA0, sizeB0)

    # MAIN 1 - Applying dimension shift (first step of AX) and turning both A and B into arrays of either 1-D or 2-D blocks
    if sizeisnew[0]:
        a = a.reshape(sizeA)
    if sizeisnew[1]:
        b = b.reshape(sizeB)

    # MAIN 2 - Performing products with or without SX (second step of AX)
    if squashOK:
        c = squash2D_mtimes(a, b, idA, idB, sizeA, sizeB, squashOK)
    elif timesOK:
        if sumOK:
            c = np.sum(a * b, axis=sumOK)
        else:
            c = a * b
    elif sxtimesOK:
        if sumOK:
            c = np.sum(np.multiply(a, b), axis=sumOK)
        else:
            c = np.multiply(a, b)
    elif mtimesOK:
        c = np.dot(a, b)

    # MAIN 3 - Reshaping C (by inserting or removing singleton dimensions)
    sizeC, sizeCisnew = adjustsize(c.shape, shiftC, False, delC, False)
    if sizeCisnew:
        c = c.reshape(sizeC)

    return c
def addsing(size, index, num_sing):
    size = list(size)
    for _ in range(num_sing):
        size.insert(index, 1)
    return tuple(size), True

def delsing(size, index, num_sing):
    size = list(size)
    for _ in range(num_sing):
        if size[index] == 1:
            del size[index]
        else:
            break
    return tuple(size), True

def swapdim(size, index):
    size = list(size)
    if len(size) > index + 1:
        size[index], size[index + 1] = size[index + 1], size[index]
    return tuple(size), True

def adjustsize(sizeA0, shiftA, addA, delA, swapA):
    # Dimension shifting (by adding or deleting trailing singleton dim.)
    if shiftA > 0:
        sizeA, newA1 = addsing(sizeA0, 1, shiftA)
    elif shiftA < 0:
        sizeA, newA1 = delsing(sizeA0, 1, -shiftA)
    else:
        sizeA = sizeA0
        newA1 = False
    
    # Modifying block size (by adding, deleting, or moving singleton dim.)
    if addA:
        sizeA, newA2 = addsing(sizeA, addA + shiftA, 1)
    elif delA:
        sizeA, newA2 = delsing(sizeA, delA + shiftA, 1)
    elif swapA:
        sizeA, newA2 = swapdim(sizeA, swapA + shiftA)
    else:
        newA2 = False
    
    sizeisnew = newA1 or newA2
    return sizeA, sizeisnew


def squash2D_mtimes(a, b, idA, idB, sizeA, sizeB, squashOK):
    if squashOK == 1:  # A is multi-block, B is single-block (squashing A)
        nd = len(sizeA)
        d2 = idA[1]
        order = list(range(d2 - 1)) + list(range(d2, nd)) + [d2 - 1]
        a = np.transpose(a, order)

        q = sizeB[0]
        s = sizeB[1]
        collapsedsize = np.prod(sizeA[order[:len(order) - 1]])
        a = a.reshape(collapsedsize, q)
        fullsize = sizeA[order[:len(order) - 1]] + (s,)
    else:  # B is multi-block, A is single-block (squashing B)
        nd = len(sizeB)
        d1 = idB[0]
        order = [d1 - 1] + list(range(d1 - 1)) + list(range(d1, nd))
        b = np.transpose(b, order)

        p = sizeA[0]
        q = sizeA[1]
        collapsedsize = np.prod(sizeB[order[1:len(order)]])
        b = b.reshape(q, collapsedsize)
        fullsize = (p,) + sizeB[order[1:len(order)]]

    invorder = np.argsort(order)
    c = np.transpose(np.dot(a, b).reshape(fullsize), invorder)
    return c

def sizeval(idA0, idB0, sizeA0, sizeB0):
    idA = idA0.copy()
    idB = idB0.copy()
    squashOK = 0
    sxtimesOK = False
    timesOK = False
    mtimesOK = False
    sumOK = 0
    shiftC = 0
    delC = 0

    NidA = len(idA)
    NidB = len(idB)
    idA1 = idA[0]
    idB1 = idB[0]
    
    if NidA > 2 or NidB > 2 or NidA == 0 or NidB == 0 or not np.isreal(idA1) or not np.isreal(idB1) or not isinstance(idA1, int) or not isinstance(idB1, int) or idA1 < 0 or idB1 < 0 or not np.isfinite(idA1) or not np.isfinite(idB1):
        raise ValueError("Internal-dimension arguments (e.g., [IDA1 IDA2]) must contain only one or two non-negative finite integers")

    declared_outer = False
    idA2 = idA[-1] if NidA > 1 else idA[0]
    idB2 = idB[-1] if NidB > 1 else idB[0]

    if 0 in idA or 0 in idB:
        #  "Inner products": C = MULTIPROD(A, B, [0 idA2], [idB1 0])
        if idA1 == 0 and idA2 > 0 and idB1 > 0 and idB2 == 0:
            idA1 = idA2
            idB2 = idB1
        # "Outer products": C = MULTIPROD(A, B, [idA1 0], [0 idB2])
        elif idA1 > 0 and idA2 == 0 and idB1 == 0 and idB2 > 0:
            declared_outer = True
            idA2 = idA1
            idB1 = idB2
        else:
            raise ValueError("Misused zeros in the internal-dimension arguments")
        NidA = 1
        NidB = 1
        idA = [idA1]
        idB = [idB1]
    elif (NidA == 2 and idA2 != idA1 + 1) or (NidB == 2 and idB2 != idB1 + 1):
        raise ValueError("If an array contains 2-D blocks, its two internal dimensions must be adjacent (e.g. IDA2 == IDA1+1)")

    # Case for which no reshaping is needed (both A and B are scalars)
    scalarA = np.array_equal(sizeA0, [1, 1])
    scalarB = np.array_equal(sizeB0, [1, 1])
    if scalarA and scalarB:
        sizeA = sizeA0
        sizeB = sizeB0
        sizeisnew = [False, False]
        timesOK = True
        return sizeA, sizeB, shiftC, delC, sizeisnew, idA, idB, squashOK, sxtimesOK, timesOK, mtimesOK, sumOK

    NsA = len(sizeA0) - idA2
    NsB = len(sizeB0) - idB2 
    adjsizeA = np.concatenate([sizeA0, np.ones(NsA, dtype=int)])
    adjsizeB = np.concatenate([sizeB0, np.ones(NsB, dtype=int)])
    
    extsizeA = np.concatenate([adjsizeA[:idA1-1], adjsizeA[idA2:]])
    extsizeB = np.concatenate([adjsizeB[:idB1-1], adjsizeB[idB2:]])
    p = adjsizeA[idA1 - 1]
    q = adjsizeA[idA2 - 1]
    r = adjsizeB[idB1 - 1]
    s = adjsizeB[idB2 - 1]

    if np.array_equal(extsizeA, extsizeB) or len(extsizeA) == 0 or len(extsizeB) == 0 or np.all(np.equal(extsizeA, 1)) or np.all(np.equal(extsizeB, 1)):
        extsize = np.maximum(extsizeA, extsizeB)
        if p == 1 or r == 1:
            squashOK = 1 if p == 1 else 2
        else:
            squashOK = 0
        sizeA = [p] + list(extsize) + [q]
        sizeB = [r] + list(extsize) + [s]
    else:
        raise ValueError("Size of the external dimensions of A and B must either be identical or compatible with AX rules")

    shiftC = [1] * (len(sizeA) - len(extsize) - 2)
    delC = [len(extsize) + 1, len(extsize) + 2]
    sizeisnew = [True, True]
    if declared_outer:
        sumOK = [len(sizeA) - 1]
        squashOK = 0

    if len(sizeA) == len(sizeB):
        if squashOK != 0:
            timesOK = False
            sxtimesOK = True
            mtimesOK = False
        elif sizeA[0] == sizeB[0] and sizeA[1] == sizeB[1]:
            timesOK = True
            sxtimesOK = False
            mtimesOK = False

    return sizeA, sizeB, shiftC, delC, sizeisnew, idA, idB, squashOK, sxtimesOK, timesOK, mtimesOK, sumOK
'''

def multiprod(a, b, idA=None, idB=None):
    if idA is None and idB is None:
        idA, idB = [0, 1], [0, 1]
    elif idA is not None and idB is None:
        idB = idA
    elif idA is None and idB is not None:
        idA = idB

    ## case 0 - matrix multiplication
    if a.ndim == 2 and b.ndim == 2 and idA == [0, 1] and idB == [0, 1]:
        return a@b
    
    ## case 1 - matrix multiplication with 2-D blocks
    sizeA0 = np.array(a.shape)
    sizeB0 = np.array(b.shape)

    def sizeval(idA, idB, sizeA0, sizeB0):
        idA1, idA2 = idA[0], idA[-1]
        idB1, idB2 = idB[0], idB[-1]
        
        squashOK = 0
        sxtimesOK = False
        timesOK = False
        mtimesOK = False
        sumOK = 0
        
        NidA = len(idA)
        NidB = len(idB)
        if NidA == 2 and idA2 != idA1 + 1 or NidB == 2 and idB2 != idB1 + 1:
            raise ValueError('If an array contains 2-D blocks, its two internal dimensions must be adjacent.')
        
        NsA = idA2 - sizeA0.size
        NsB = idB2 - sizeB0.size
        adjsizeA = np.append(sizeA0, np.ones(NsA))
        adjsizeB = np.append(sizeB0, np.ones(NsB))
        extsizeA = adjsizeA[np.arange(len(adjsizeA)) != idA1]
        extsizeB = adjsizeB[np.arange(len(adjsizeB)) != idB1]
        
        extsize = np.maximum(extsizeA, extsizeB)
        
        if np.array_equal(adjsizeA, extsizeA) and np.array_equal(adjsizeB, extsizeB):
            squashOK = 0
        elif np.array_equal(adjsizeA, extsizeA) and not np.array_equal(adjsizeB, extsizeB):
            squashOK = 1
        elif not np.array_equal(adjsizeA, extsizeA) and np.array_equal(adjsizeB, extsizeB):
            squashOK = 2
        
        sizeA = np.insert(extsize, idA1, adjsizeA[idA1:idA2+1])
        sizeB = np.insert(extsize, idB1, adjsizeB[idB1:idB2+1])
        
        return sizeA, sizeB, squashOK, sxtimesOK, timesOK, mtimesOK, sumOK

    sizeA, sizeB, squashOK, sxtimesOK, timesOK, mtimesOK, sumOK = sizeval(idA, idB, sizeA0, sizeB0)

    a = a.reshape(sizeA)
    b = b.reshape(sizeB)

    if squashOK == 1:
        a = np.moveaxis(a, idA[1], -1)
        n = np.prod(sizeA[:-1])
        a = a.reshape(n, sizeB[0])
        c = np.dot(a, b)
        c = c.reshape(sizeA[:-1] + sizeB[1:])
        return np.moveaxis(c, -1, idA[1])
    elif squashOK == 2:
        b = np.moveaxis(b, idB[0], 0)
        n = np.prod(sizeB[1:])
        b = b.reshape(sizeA[1], n)
        c = np.dot(a, b)
        c = c.reshape(sizeA[0] + sizeB[1:])
        return np.moveaxis(c, 0, idB[0])
    else:
        raise ValueError('Unsupported operation.')
'''    

def multitransp(a, dim=0):
    """
    Transposes arrays of matrices along specified dimensions.

    Parameters:
    a : numpy.ndarray
        Input array.
    dim : int, optional
        Dimension along which to transpose. Default is 0.

    Returns:
    numpy.ndarray
        Transposed array.
    """
    # Setting order for permutation
    order = list(range(a.ndim))
    order[dim], order[dim+1] = order[dim+1], order[dim]
    return np.transpose(a, order)