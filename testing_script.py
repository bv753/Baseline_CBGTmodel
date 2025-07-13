import pickle as pkl

import model_functions as mf
import plotting_functions as pf

#load params_nm
with open('params_nm.pkl', 'rb') as f:
    params_nm = pkl.load(f)

all_ys, all_xs, all_zs = mf.test_model(params_nm)

# calculate testing loss
testing_loss, plt_indices = mf.calculate_testing_loss(all_ys)
print("Testing loss:", testing_loss)

# select trials for plotting
plt_ys, plt_xs, plt_zs = mf.select_trials(all_ys, all_xs, all_zs, plt_indices)

pf.plot_output(plt_ys)
pf.plot_activity_by_area(plt_xs, plt_zs)
pf.plot_cue_align_activity(plt_xs, plt_zs)

v_response_times = mf.get_response_times(plt_ys, plt_indices)
pf.plot_response_times(v_response_times)

pf.plot_binned_responses(plt_ys, plt_xs, plt_zs, plt_indices)