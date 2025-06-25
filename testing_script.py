import jax.numpy as jnp
import config_script as cs
import model_functions as mf
import plotting_functions as pf
import training_script as trs

all_ys, all_xs, all_zs = mf.test_model(trs.params_nm)

# calculate testing loss
m_ys = jnp.mean(all_ys, 0)  # mean over random seed iterations
tst_inputs, tst_outputs, tst_masks, tst_reg_indices = mf.self_timed_movement_task(cs.test_start_t, cs.config['T_cue'], cs.config['T_wait'], cs.config['T_movement'], cs.config['T'], null_trial=cs.test_null_trials)
testing_loss = jnp.sum(((m_ys - tst_outputs) ** 2) * tst_masks) / jnp.sum(tst_masks)
print("Testing loss:", testing_loss)

reg_ys = jnp.take(all_ys, tst_reg_indices, 1)

reg_xs_bg = jnp.take(all_xs[0], tst_reg_indices, 1)
reg_xs_c = jnp.take(all_xs[1], tst_reg_indices, 1)
reg_xs_t = jnp.take(all_xs[2], tst_reg_indices, 1)
reg_xs = [reg_xs_bg, reg_xs_c, reg_xs_t]

reg_zs = jnp.take(all_zs, tst_reg_indices, 1)

pf.plot_output(reg_ys)
pf.plot_activity_by_area(reg_xs, reg_zs)
pf.plot_cue_align_activity(reg_xs, reg_zs)

v_response_times = mf.get_response_times(reg_ys)
pf.plot_response_times(v_response_times)

pf.plot_binned_responses(reg_ys, reg_xs, reg_zs)