import pickle as pkl

import model_functions as mf
import plotting_functions as pf

#load params_nm
with open('params_nm.pkl', 'rb') as f:
    params_nm = pkl.load(f)

all_ys, all_xs, all_zs = mf.test_model(params_nm)

# calculate testing loss
testing_loss, tst_reg_indices = mf.calculate_testing_loss(all_ys)
print("Testing loss:", testing_loss)

# select non-null trials for plotting
reg_ys, reg_xs, reg_zs = mf.select_regular_trials(all_ys, all_xs, all_zs, tst_reg_indices)

pf.plot_output(reg_ys)
pf.plot_activity_by_area(reg_xs, reg_zs)
pf.plot_cue_align_activity(reg_xs, reg_zs)

v_response_times = mf.get_response_times(reg_ys)
pf.plot_response_times(v_response_times)

pf.plot_binned_responses(reg_ys, reg_xs, reg_zs)