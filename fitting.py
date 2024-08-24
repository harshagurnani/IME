import time
import numpy as np
import sys, pathlib
from helpers.utils import compute_angular_errors
from e_step import *
from m_step import *



def initialize_velime(data, TAU, INIT_METHOD, INIT_PARAMS, *args):
    """
    Initializes the parameters for the VE-LIME algorithm.

    Args:
        data (numpy.ndarray): The input data.
        TAU (float): The time constant.
        *args: Additional positional argument - to set new variables.

    Returns:
        tuple: A tuple containing the initialized parameters, data matrices, and constants.

    Raises:
        ValueError: If the initialization method is unsupported.
    """
    
    
    '''
    TOL = 1e-8
    ZERO_TOL = 1e-10
    MAX_ITERS = 5e3
    MIN_ITERS = 10
    VERBOSE = False
    INIT_METHOD = 'current_regression'
    INIT_PARAMS = np.nan
    DO_LEARN_M_PARAMS = True
    DO_LEARN_ALPHA_PARAMS = True
    globals().update(locals())

    assignopts(globals(), *args )
    '''

    DT = 1

    if INIT_METHOD.lower() == 'target_regression':
        B, b0, W_v = target_regression(data)
        estParams = {
            'A': np.eye(2),
            'B': B,
            'b0': b0,
            'W_v': W_v,
            'W_p': np.zeros_like(W_v),
            'TAU': TAU,
            'dt': DT
        }
    elif INIT_METHOD.lower() == 'current_regression':
        A, B, b0, W_v, _ = current_regression(data)
        estParams = {
            'A': np.zeros_like(A),
            'B': B,
            'b0': b0,
            'W_v': W_v,
            'W_p': np.zeros_like(W_v),
            'TAU': TAU,
            'dt': DT
        }
    elif INIT_METHOD.lower() == 'init_params':
        estParams = INIT_PARAMS
        estParams.update({
            'TAU': TAU,
            'dt': DT
        })
    else:
        raise ValueError('Unsupported initialization method')

    C, G, const = velime_assemble_data(data, TAU, DT)

    if 'alpha' not in estParams.keys():
        E_X, COV_X, _ = velime_prior_expectation(C, estParams)
        init_alphaParams = velime_mstep_alphaParams(G, E_X, np.tile(COV_X[:,:,np.newaxis], (1, 1, const['T'])), const)
        estParams['alpha'] = init_alphaParams['alpha']
        estParams['R'] = init_alphaParams['R']

    return estParams, C, G, const

def velime_fit(data, TAU, **kwargs):
    """
    Fits the internal model estimation (IME) framework via expectation
    maximization (EM). The IME framework is described in Golub, Yu & Chase,
    eLife, 2015 (https://elifesciences.org/articles/10015). In the
    descriptions below, we reference specific equations from this paper.

    INPUTS:
    - data: Dictionary with keys 'spike_counts', 'cursor_position', 'target_position'.
    - TAU: Integer, the sensory feedback delay.

    OPTIONAL INPUTS (with default values):
    - TOL: Convergence tolerance for training log-likelihood (default: 1e-8).
    - ZERO_TOL: Tolerance for detecting violations of EM-guaranteed increases in log-likelihood (default: 1e-10).
    - MAX_ITERS: Maximum number of EM iterations (default: 5000).
    - MIN_ITERS: Minimum number of EM iterations (default: 10).
    - VERBOSE: Whether to print status updates (default: False).
    - INIT_METHOD: Initialization method (default: 'current_regression').
    - INIT_PARAMS: Initial set of parameters (default: NaN).
    - DO_LEARN_M_PARAMS: Whether to learn dynamics parameters (default: True).
    - DO_LEARN_ALPHA_PARAMS: Whether to learn scale parameters (default: True).

    OUTPUTS:
    - estParams: Dictionary with estimated parameters.
    - LL: List of training log-likelihoods.
    """
    start_time = time.time()
    
    # Default values
    TOL = kwargs.get('TOL', 1e-8)
    ZERO_TOL = kwargs.get('ZERO_TOL', 1e-10)
    MAX_ITERS = kwargs.get('MAX_ITERS', 5000)
    MIN_ITERS = kwargs.get('MIN_ITERS', 10)
    VERBOSE = kwargs.get('VERBOSE', False)
    INIT_METHOD = kwargs.get('INIT_METHOD', 'current_regression')
    INIT_PARAMS = kwargs.get('INIT_PARAMS', np.nan)
    DO_LEARN_M_PARAMS = kwargs.get('DO_LEARN_M_PARAMS', True)
    DO_LEARN_ALPHA_PARAMS = kwargs.get('DO_LEARN_ALPHA_PARAMS', True)

    # Initialization
    estParams, C, G, const = initialize_velime(data, TAU, INIT_METHOD, INIT_PARAMS)
    
    LLi = np.nan
    LL = np.full(MAX_ITERS+1, np.nan)
    iter_times = np.full(MAX_ITERS, np.nan)
    iters_completed = 0

    while True:
        iters_completed_start_time = time.time()
        LLold = np.copy(LLi)
        
        # E-step
        LLi, E_X_posterior, COV_X_posterior = velime_estep(C, G, estParams, const)
        LL[iters_completed] = LLi
        
        if iters_completed % 50 == 0 and iters_completed > 0:
            if VERBOSE:
                print(f'\t\tvelime(TAU={TAU}) iters: {iters_completed}, Mean iter time: {np.mean(iter_times[:iters_completed]):.1e}s, LL improvement: {LLi - LLold:.3e}')
        
        # Check for convergence
        if iters_completed > 1 and LLi < LLold and abs(LLold - LLi) > ZERO_TOL:
            print(f'\t\tVIOLATION velime(TAU={TAU}) iters: {iters_completed}: {LLi - LLold}')
            try:
                import matplotlib.pyplot as plt
                plt.plot(LL)
                #plt.show()
            except Exception as e:
                pass
        if (iters_completed > MIN_ITERS) and (LLi > LLold) and (LLi - LLold < TOL):
            if VERBOSE:
                print(f'\t\tvelime(TAU={TAU}) converged after {time.time() - start_time:.2f} seconds: {iters_completed} iters')
            break  # Convergence
        elif iters_completed >= MAX_ITERS:
            if VERBOSE:
                print(f'\t\tvelime(TAU={TAU}) iter limit reached, aborting after {time.time() - start_time:.2f} seconds, LL improvement: {LLi - LLold:.3e}')
            break

        # M-step
        if DO_LEARN_M_PARAMS:
            M_Params_fastfmc = velime_mstep_MParams(E_X_posterior, COV_X_posterior, C, const)
            estParams['A'] = M_Params_fastfmc['M1']
            estParams['B'] = M_Params_fastfmc['M2']
            estParams['b0'] = M_Params_fastfmc['m0']
            estParams['W_v'] = M_Params_fastfmc['V']
        
        if DO_LEARN_ALPHA_PARAMS:
            ALPHA_Params = velime_mstep_alphaParams(G, E_X_posterior, COV_X_posterior, const)
            estParams['alpha'] = ALPHA_Params['alpha']
            estParams['R'] = ALPHA_Params['R']
        
        iters_completed += 1
        iter_times[iters_completed - 1] = time.time() - iters_completed_start_time
    
    return estParams, LL[:iters_completed]


def velime_predict(data, estParams):
    """
    Extract the subject's internal state predictions given available visual 
    feedback of cursor position and previously issued neural activity. 
    Importantly, these predictions do not take into account target positions.
    
    Parameters:
    - data: Dictionary with keys 'spike_counts', 'cursor_position', 'target_position'.
    - estParams: Dictionary containing IME parameters as identified by velime_fit.
    
    Returns:
    - E_P: List of arrays. Each array is [2*(TAU+1) x num_timesteps].
    - E_V: List of arrays. Each array is [2*(TAU+1) x num_timesteps].
    """
    TAU = estParams['TAU']
    dt = estParams['dt']
    
    # Get predictions from the model
    C, G, const = velime_assemble_data(data, TAU, dt)
    E_X, _, _ = velime_prior_expectation(C, estParams, COMPUTE_E_TARG=False)
    
    # Initialize E_P and E_V
    N_trials = len(const['trial_map'])
    pIdx = const['x_idx'][:const['pDim']].flatten('F')
    vIdx = const['x_idx'][const['pDim']:const['xDim']].flatten('F')
    
    E_P = [None] * N_trials
    E_V = [None] * N_trials
    
    for trialNo in range(N_trials):
        trial_indices = const['trial_map'][trialNo]
        T_data = data['cursor_position'][trialNo].shape[1]
        
        # Pad with TAU + 1 columns of NaNs at the start and one column at the end
        if T_data >= TAU + 2:
            E_P[trialNo] = np.hstack([np.full((pIdx.size, TAU + 1), np.nan), E_X[np.ix_(pIdx, trial_indices)], np.full((pIdx.size, 1), np.nan)])
            E_V[trialNo] = np.hstack([np.full((vIdx.size, TAU + 1), np.nan), E_X[np.ix_(vIdx, trial_indices)], np.full((vIdx.size, 1), np.nan)])
        else:
            # Make sure NaN padding does not change the length of the trial
            E_P[trialNo] = np.full((pIdx.size, T_data), np.nan)
            E_V[trialNo] = np.full((vIdx.size, T_data), np.nan)
    
    return E_P, E_V


import numpy as np

def velime_evaluate(data, E_P, E_V, estParams, **kwargs):
    """
    Compute timestep-by-timestep angular errors and optionally the log-likelihood.
    
    Parameters:
    - data: Dictionary with keys 'spike_counts', 'cursor_position', 'target_position'.
    - E_P: List of arrays. Each array is [2*(TAU+1) x num_timesteps].
    - E_V: List of arrays. Each array is [2*(TAU+1) x num_timesteps].
    - estParams: Dictionary containing fitted IME parameters.
    - **kwargs: Additional optional arguments.
    
    Returns:
    - angular_error: Dictionary with 'cursor' and 'model' keys containing vectors of angular errors.
    - LL (optional): Scalar log-likelihood value.
    """
    VERBOSE = kwargs.get('VERBOSE', False)
    T_START = kwargs.get('T_START', estParams['TAU'] + 2)
    DO_COMPUTE_LL = kwargs.get('DO_COMPUTE_LL', False)
    DATA_ARE_TRAINING_DATA = kwargs.get('DATA_ARE_TRAINING_DATA', False)
    MAX_ITERS = kwargs.get('MAX_ITERS', 50)

    if estParams['TAU'] < 0:
        raise ValueError('TAU must be >= 0')
    if T_START < (estParams['TAU'] + 2):
        raise ValueError('T_START must be >= TAU + 2')

    angular_error = compute_angular_errors(data, E_P, E_V, estParams, T_START)

    LL = np.nan
    if DO_COMPUTE_LL:
        LL = compute_LL(data, estParams, T_START, DATA_ARE_TRAINING_DATA, MAX_ITERS, VERBOSE)
    
    return angular_error, LL


def velime_cross_validate(data, TAU, **kwargs):
    """
    Perform cross-validation to evaluate the internal model's predictions.
    
    Parameters:
    - data: Dictionary with keys 'spike_counts', 'cursor_position', 'target_position'.
    - TAU: Integer representing the sensory feedback delay.
    - **kwargs: Additional optional arguments.

    Returns:
    - estParams: List of internal model parameters for each training set.
    - predictions: Dictionary containing cross-validated internal state estimates.
    - evaluations: Dictionary containing cross-validated angular errors and log-likelihood.
    - cv_folds: List of trial indices used in the cross-validation procedure.
    """
    VERBOSE = kwargs.get('VERBOSE', False)
    T_START = kwargs.get('T_START', TAU + 2)
    DO_COMPUTE_LL = kwargs.get('DO_COMPUTE_LL', True)
    velime_fit_args = kwargs
    MAX_FOLDS = kwargs.get('MAX_FOLDS', 10)

    # Generate cross-validation folds
    cv_folds, _ = generate_shuffled_blocks(data['target_position'])
    num_cv_folds = min( len(cv_folds), MAX_FOLDS )
    num_trials = len(data['cursor_position'])

    # Initialize storage
    E_P = [None] * num_trials
    E_V = [None] * num_trials
    angular_error = {
        'model': np.full(num_trials, np.nan),
        'cursor': np.full(num_trials, np.nan)
    }
    LL_fold = np.full(num_cv_folds, np.nan)
    estParams = [None] * num_cv_folds

    for fold_idx in range(num_cv_folds):
        if VERBOSE:
            print(f'Beginning cross-validation fold {fold_idx + 1} of {num_cv_folds}.')

        test_idx = cv_folds[fold_idx]
        test_data = subsample_trials(data, test_idx)
        
        all_folds_but_one = [i for i in range(len(cv_folds)) if i != fold_idx]
        train_idx = [idx for sublist in [cv_folds[i] for i in all_folds_but_one] for idx in sublist]
        train_data = subsample_trials(data, train_idx)

        # Fit velocity-IME model
        
        estParams[fold_idx], LLtemp = velime_fit(train_data, TAU, **velime_fit_args)
        
        if VERBOSE:
            print('\tExtracting cross-validated predictions.')

        # Extract prior latent variable distributions ("whiskers")
        E_P_temp, E_V_temp = velime_predict(test_data, estParams[fold_idx])
        for i, idx in enumerate(test_idx):
            E_P[idx] = E_P_temp[i]
            E_V[idx] = E_V_temp[i]
        
        if VERBOSE:
            print('\tEvaluating cross-validated predictions.')

        # Evaluate predictions
        fold_angular_errors, LL_fold[fold_idx] = velime_evaluate(
            test_data,
            [E_P[i] for i in test_idx],
            [E_V[i] for i in test_idx],
            estParams[fold_idx],
            T_START=T_START,
            DO_COMPUTE_LL=DO_COMPUTE_LL,
            VERBOSE=VERBOSE
        )

        for i, idx in enumerate(test_idx):
            angular_error['model'][idx] = fold_angular_errors['model'][i]
            angular_error['cursor'][idx] = fold_angular_errors['cursor'][i]

        if VERBOSE:
            print('Done.\n')
    
    predictions = {
        'E_P': E_P,
        'E_V': E_V
    }
    evaluations = {
        'angular_error': angular_error,
        'LL': np.sum(LL_fold)
    }

    return estParams, predictions, evaluations, cv_folds



def compute_LL(data, estParams, T_START, DATA_ARE_TRAINING_DATA, MAX_ITERS=5000, VERBOSE=False):
    """
    Compute the log-likelihood of the data.
    
    Parameters:
    - data: Dictionary with keys 'spike_counts', 'cursor_position', 'target_position'.
    - estParams: Dictionary containing fitted IME parameters.
    - T_START: Integer timestep to begin evaluating the data.
    - DATA_ARE_TRAINING_DATA: Boolean indicating if params were fit to the data.
    - MAX_ITERS: Integer for the number of EM iterations.
    - VERBOSE: Boolean flag to print status updates.
    
    Returns:
    - LL: Scalar log-likelihood value.
    """
    if VERBOSE:
        print('\tEvaluating log-likelihood.')

    if DATA_ARE_TRAINING_DATA:
        LL = velime_LL(data, estParams, T_START=T_START)
    else:
        # Initialize test parameters by removing alpha and R
        init_testParams = {k: v for k, v in estParams.items() if k not in ['alpha', 'R']}
        
        # Fit the model to the data to get alpha and R
        TAU = estParams['TAU']
        testParams, _ = velime_fit(data, TAU,
            INIT_METHOD='init_params',
            INIT_PARAMS=init_testParams,
            DO_LEARN_M_PARAMS=False,
            MAX_ITERS=MAX_ITERS,
            VERBOSE=VERBOSE
        )
        
        LL = velime_LL(data, testParams, T_START=T_START)
    
    return LL


def velime_sweep_TAU(data, TAUs, **kwargs):
    """
    Perform a sweep over a set of TAU values to fit and evaluate IME models.
    
    Parameters:
    - data: Dictionary with keys 'spike_counts', 'cursor_position', 'target_position'.
    - TAUs: List of candidate TAU values to test.
    - **kwargs: Additional optional arguments for model fitting and evaluation.
    
    Returns:
    - sweep: List of dictionaries, each containing the following fields:
        - estParams: Dictionary containing fitted IME parameters.
        - angular_error: Dictionary with 'model' and 'cursor' vectors of angular errors.
        - LL: Scalar log-likelihood value.
        - TAU: The TAU value used for the model.
    """
    DO_CROSS_VALIDATE = kwargs.get('DO_CROSS_VALIDATE', False)
    DO_COMPUTE_LL = kwargs.get('DO_COMPUTE_LL', True)
    velime_args = {key: kwargs[key] for key in kwargs if key not in {'DO_CROSS_VALIDATE', 'DO_COMPUTE_LL'}}

    T_START = 2 + max(TAUs)
    sweep = []

    for TAU in TAUs:
        print(f'TAU = {TAU}')

        if DO_CROSS_VALIDATE:
            estParams, _, evaluations, _ = velime_cross_validate(data, TAU,
                T_START=T_START,
                DO_COMPUTE_LL=True,
                **velime_args)
            
            angular_error = evaluations['angular_error']
            LL = evaluations['LL']
        else:
            estParams = velime_fit(data, TAU, **velime_args)
            E_P, E_V = velime_predict(data, estParams)
            angular_error, LL = velime_evaluate(data, E_P, E_V, estParams, T_START=T_START, DO_COMPUTE_LL=DO_COMPUTE_LL)
        
        sweep.append({
            'estParams': estParams,
            'angular_error': angular_error,
            'LL': LL,
            'TAU': TAU
        })
    
    return sweep