import config_script as cs
import model_functions as mf
import plotting_functions as pf
import training_script as trs

import jax.numpy as jnp

all_ys, all_xs, all_zs = mf.test_model(trs.params_nm)

# calculate testing loss
m_ys = jnp.mean(all_ys, 0)  # mean over random seed iterations
tst_inputs, tst_outputs, tst_masks = mf.self_timed_movement_task(cs.test_start_t, cs.config['T_cue'], cs.config['T_wait'], cs.config['T_movement'], cs.config['T'])
testing_loss = jnp.sum(((m_ys - tst_outputs) ** 2) * tst_masks) / jnp.sum(tst_masks)
print("Testing loss:", testing_loss)

pf.plot_output(all_ys)
pf.plot_activity_by_area(all_xs, all_zs)
pf.plot_cue_align_activity(all_xs, all_zs)

v_response_times = mf.get_response_times(all_ys)
pf.plot_response_times(v_response_times)

pf.plot_binned_responses(all_ys, all_xs, all_zs)