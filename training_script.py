from model_functions import *
from config_script import *
import plotting_functions as pf
import pickle as pkl

all_inputs, all_outputs, all_masks, _ = self_timed_movement_task(config['T_start'], config['T_cue'], config['T_wait'], config['T_movement'], config['T'], null_trial=cs.config['train_null_trials'])

# train on all params
params_nm, losses_nm = fit_nm_rnn(all_inputs, all_outputs, all_masks,
                                params, optimizer, x0, z0, config['num_full_train_iters'],
                                config['tau_x'], config['tau_z'], wandb_log=False, modulation=True)

#save params_nm, which is a dictionary, using pickle
with open('params_nm.pkl', 'wb') as f:
    pkl.dump(params_nm, f)

pf.plot_loss(losses_nm)
