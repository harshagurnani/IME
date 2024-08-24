import numpy as np
import matplotlib.pyplot as plt
import os
def plot_sweep_TAU(sweep, savfolder='../results/', savname='sweep_TAU.png'):
    """
    Plot various assessments for IME models fit across a range of settings of TAU.
    
    Parameters:
    - sweep: List of dicts, each containing the results for different settings of TAU.
    
    Each dict in sweep should have the following keys:
    - 'TAU': The value of TAU.
    - 'LL': The log-likelihood of the model.
    - 'angular_error': A dict with the key 'model' containing the angular errors.
    - 'estParams': A list of parameter matrices A (or a single matrix) for cross-validation.
    """

    if not os.path.exists(savfolder):
        os.makedirs(savfolder)
    
    TAUs = np.array([s['TAU'] for s in sweep])
    
    # Extract log-likelihood
    LL = np.array([s['LL'] for s in sweep])
    idx_max_LL = np.argmax(LL)
    TAU_star_LL = TAUs[idx_max_LL]
    
    # Extract angular error
    angular_errors = np.array([np.nanmean(s['angular_error']['model']) for s in sweep])
    idx_min_angular_error = np.argmin(angular_errors)
    TAU_star_angular_error = TAUs[idx_min_angular_error]
    
    # Extract autoregressive coefficients
    if isinstance(sweep[0]['estParams'], list):
        ar_coeffs = np.full(len(sweep), np.nan)
        for i, s in enumerate(sweep):
            estParams_array = s['estParams']
            ar_coeffs_per_fold = np.array([np.mean(np.diag(estParam['A'])) for estParam in estParams_array])
            ar_coeffs[i] = np.mean(ar_coeffs_per_fold)
    else:
        ar_coeffs = np.array([np.mean(np.diag(s['estParams']['A'])) for s in sweep])
    
    idx_max_ar_coeff = np.argmax(ar_coeffs)
    TAU_star_ar_coeff = TAUs[idx_max_ar_coeff]
    
    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    
    # Plot log-likelihood
    axs[0].plot(TAUs, LL, label='Log-likelihood')
    axs[0].plot(TAU_star_LL, LL[idx_max_LL], 'ro', label='Best LL')
    axs[0].set_ylabel('Log-likelihood')
    axs[0].box(False)
    axs[0].tick_params(axis='x', direction='out')
    axs[0].legend()
    
    # Plot angular error
    axs[1].plot(TAUs, angular_errors, label='Angular error')
    axs[1].plot(TAU_star_angular_error, angular_errors[idx_min_angular_error], 'ro', label='Best Angular Error')
    axs[1].set_ylabel('Absolute angular error (degrees)')
    axs[1].set_xlabel('TAU')
    axs[1].box(False)
    axs[1].tick_params(axis='x', direction='out')
    axs[1].legend()
    
    # Plot autoregressive coefficient
    axs[2].plot(TAUs, ar_coeffs, label='Autoregressive Coefficient')
    axs[2].plot(TAU_star_ar_coeff, ar_coeffs[idx_max_ar_coeff], 'ro', label='Max AR Coefficient')
    axs[2].set_ylabel('Approx autoregressive coefficient [mean(diag(A_v))]')
    axs[2].set_xlabel('TAU')
    axs[2].box(False)
    axs[2].tick_params(axis='x', direction='out')
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig(savfolder + savname)


def plot_trials_with_whiskers(P, TARGETS, E_P, E_V, estParams, CURSOR_RADIUS, TARGET_RADIUS, TAU, savfolder='results/', savname='whiskers.png', plot_max=None):
    """
    Plot trials with whiskers representing internal estimates of cursor position.

    Parameters:
    - P: List of actual cursor trajectories, where each element is a 2D array of shape (2, num_timesteps).
    - TARGETS: List of target positions, where each element is a 2D array of shape (2,).
    - E_P: List of estimated cursor positions, where each element is a 2D array of shape (2 * num_timesteps,).
    - E_V: List of estimated velocities, where each element is a 2D array of shape (2 * num_timesteps,).
    - estParams: Object containing parameters including dt.
    - CURSOR_RADIUS: Radius of the cursor.
    - TARGET_RADIUS: Radius of the target.
    - TAU: Sensory feedback delay.
    """
    if not os.path.exists(savfolder):
        os.makedirs(savfolder)
    
    plt.figure()
    plt.gca().set_aspect('equal')

    num_test_trials = len(P)
    if plot_max is not None:
        plot_trials = np.random.choice(num_test_trials, plot_max, replace=False)
    else:
        plot_trials = range(num_test_trials)
    for n in plot_trials:
        # Plot target with enlarged radius to match the "acceptance zone" radius
        fill_circle(TARGETS[n], TARGET_RADIUS, 'g')
        plot_circle(TARGETS[n], CURSOR_RADIUS + TARGET_RADIUS, 'g')

        # Plot the actual cursor trajectory in black
        plt.plot(P[n][0, :], P[n][1, :], 'k-o')

        # At each timestep, plot a "whisker" representing the evolution of the subject's internal estimates
        T = P[n].shape[1]
        for t in range(TAU, T):
            # Plot estimated cursor positions
            plt.plot(E_P[n][0:2*T:2, t], E_P[n][1:2*T:2, t], 'r.-')

            # Plot the subject's up-to-date estimate of the current cursor position
            p_t_t = E_P[n][-2:, t]
            plt.plot(p_t_t[0], p_t_t[1], 'ro')

            # Compute predicted cursor position after issuing the velocity command
            v_t_t = E_V[n][-2:, t]
            p_tp1_t = p_t_t + v_t_t * estParams['dt']
            plt.plot([p_t_t[0], p_tp1_t[0]], [p_t_t[1], p_tp1_t[1]], 'r-')

    plt.axis('image')
    plt.savefig(savfolder + savname)



def plot_angular_aiming_errors(angular_errors, savefolder='results/', savname='angular_errors.png'):
    """
    Plot the angular aiming errors for cursor and internal model.

    Parameters:
    - angular_errors: A dictionary containing:
        - 'cursor': Array of within-trial averages of absolute angular errors in the actual cursor trajectory.
        - 'model': Array of within-trial averages of absolute angular errors according to an internal model.
    """
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    

    # Helper functions
    

    xx = [1, 2]
    cursor_avg_error, cursor_sem_error = mean_and_sem(angular_errors['cursor'])
    model_avg_error, model_sem_error = mean_and_sem(angular_errors['model'])
    
    # Plotting
    plt.figure()
    bars = plot_bar_and_sem(xx, [cursor_avg_error, model_avg_error], [cursor_sem_error, model_sem_error])
    
    plt.xlim(xx[0] - 0.5, xx[-1] + 0.5)
    plt.ylabel('Absolute Angular Error (degrees)')
    plt.xticks(xx, ['Cursor', 'Internal Model'])
    plt.gca().tick_params(axis='x', direction='out')
    plt.box(False)
    
    plt.savefig(savefolder + savname)
    
    return bars


def mean_and_sem(x):
    """Compute mean and standard error of the mean (SEM)."""
    x = np.array(x).flatten()
    m = np.nanmean(x)
    n = np.sum(~np.isnan(x))
    sem = np.nanstd(x, ddof=1) / np.sqrt(n)  # Use ddof=1 for sample std deviation
    return m, sem

def plot_bar_and_sem(x, y, sem):
    """Plot bars with SEM error bars."""
    bar_width = 0.8
    bars = plt.bar(x, y, bar_width, color='none', edgecolor='k')

    for i in range(len(x)):
        plt.plot([x[i], x[i]], [y[i] - sem[i], y[i] + sem[i]], color='k')

    return bars


def fill_circle(center, radius, color):
    """Helper function to plot filled circles."""
    circle = plt.Circle(center, radius, color=color, alpha=0.3)
    plt.gca().add_artist(circle)

def plot_circle(center, radius, color):
    """Helper function to plot circle edges."""
    circle = plt.Circle(center, radius, color=color, fill=False)
    plt.gca().add_artist(circle)

