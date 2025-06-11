from model_functions import *
from config_script import *
import plotting_functions as pf

all_inputs, all_outputs, all_masks = self_timed_movement_task(config['T_start'], config['T_cue'], config['T_wait'], config['T_movement'], config['T'])

# train on all params
params_nm, losses_nm = fit_nm_rnn(all_inputs, all_outputs, all_masks,
                                params, optimizer, x0, z0, config['num_full_train_iters'],
                                config['tau_x'], config['tau_z'], wandb_log=False, modulation=True)

pf.plot_loss(losses_nm)
