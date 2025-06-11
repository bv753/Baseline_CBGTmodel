import model_functions as mf
import plotting_functions as pf
import training_script as ts

all_ys, all_xs, all_zs = mf.test_model(ts.params_nm)

pf.plot_output(all_ys)
pf.plot_activity_by_area(all_xs, all_zs)
pf.plot_cue_align_activity(all_xs, all_zs)

valid_response_times = mf.get_response_times(all_ys)
pf.plot_response_times(valid_response_times)

pf.plot_binned_responses(all_ys, all_xs, all_zs)
