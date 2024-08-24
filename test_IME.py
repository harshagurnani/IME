import numpy as np
import fitting
import matplotlib.pyplot as pp

import scipy.io
mat = scipy.io.loadmat('example_data.mat')


sc = mat['data']['spike_counts'][0][0][0]
pos = mat['data']['cursor_position'][0][0][0]
tgt = mat['data']['target_position'][0][0][0]

data = {'spike_counts':[sc[jj] for jj in range(sc.shape[0])],
        'cursor_position':[pos[jj] for jj in range(sc.shape[0])],
        'target_position':[tgt[jj] for jj in range(sc.shape[0])],
        'cursor_radius':mat['data']['cursor_radius'][0][0][0][0], 
        'target_radius':mat['data']['target_radius'][0][0][0][0],
        }

MAX_ITERS=100
TAU=5
MAX_FOLDS=11

# simple
estParams, LL = fitting.velime_fit(data,TAU, MAX_ITERS=MAX_ITERS, INIT_METHOD='current_regression', VERBOSE=True)
pp.close('all')
pp.plot(LL)
pp.show()

# cross-validated fits
estParams, predictions, evaluations, cv_folds = fitting.velime_cross_validate(data, TAU, MAX_ITERS=MAX_ITERS, MAX_FOLDS=MAX_FOLDS, VERBOSE=True)
plot_max=20
import helpers.plotting as ph
ph.plot_trials_with_whiskers(data['cursor_position'], data['target_position'], predictions['E_P'], predictions['E_V'], estParams[-1], data['cursor_radius'], data['target_radius'], TAU, plot_max=plot_max, savname='whiskers_.png')
ph.plot_angular_aiming_errors(evaluations['angular_error'], savname='angular_errors_.png')
