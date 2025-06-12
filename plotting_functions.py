import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import config_script as cs
import model_functions as mf


def plot_loss(losses_nm):
    loss_curve_nm = [loss[-1] for loss in losses_nm]
    # loss_curve_vanilla = [loss[-1] for loss in losses_vanilla]
    x_axis = np.arange(len(losses_nm)) * 200
    
    plt.cla()
    plt.plot(x_axis, np.log10(loss_curve_nm), label='NM RNN')
    # plt.plot(x_axis, np.log10(loss_curve_vanilla), label='Vanilla RNN')
    plt.ylabel('log10(error)')
    plt.xlabel('iteration')
    plt.legend()
    plt.show()

def plot_output(all_ys):
    # Plot output activity (mean ± SEM)
    # plt.close('all')
    fig = plt.figure(figsize=(4, 3))
    colors = plt.cm.coolwarm(jnp.linspace(0, 1, all_ys.shape[1]))
    mean_ys, sem_ys = mf.compute_mean_sem(all_ys)
    
    for i in range(mean_ys.shape[0]):
        plt.plot(mean_ys[i, :, 0], c=colors[i])
        plt.fill_between(
            jnp.arange(mean_ys.shape[1]),
            mean_ys[i, :, 0] - sem_ys[i, :, 0],
            mean_ys[i, :, 0] + sem_ys[i, :, 0],
            color=colors[i],
            alpha=0.3,
        )
    plt.title(f'Output (mean ± SEM, noise_std={cs.test_noise_std})')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_activity_by_area(all_xs, all_zs):
    # plt.close('all')
    fig = plt.figure(figsize=(8, 6))
    for idx, name in enumerate(['D1', 'D2', 'Cortex', 'Thalamus', 'SNc', 'nm']):
        ax = plt.subplot(2, 3, idx + 1)
        area_activity = jnp.stack(
            [mf.get_brain_area(name, xs_seed, zs_seed) for xs_seed, zs_seed in zip(all_xs, all_zs)]
        )  # shape: (n_seeds, n_conditions, T, N)
        mean_act, sem_act = mf.compute_mean_sem(jnp.mean(area_activity, axis=3))  # Avg across neurons
        colors = plt.cm.coolwarm(jnp.linspace(0, 1, mean_act.shape[0]))
        for i in range(mean_act.shape[0]):
            ax.plot(mean_act[i], c=colors[i], label=f'Condition {i}')
            ax.fill_between(
                jnp.arange(mean_act.shape[1]),
                mean_act[i] - sem_act[i],
                mean_act[i] + sem_act[i],
                color=colors[i],
                alpha=0.3,
            )
        ax.set_title(f'{name}')
    plt.suptitle('Aligned to trial start')
    plt.tight_layout()
    plt.show()

def plot_cue_align_activity(all_xs, all_zs):
    # plt.close('all')
    fig = plt.figure(figsize=(8, 6))
    for idx, name in enumerate(['D1', 'D2', 'Cortex', 'Thalamus', 'SNc', 'nm']):
        ax = plt.subplot(2, 3, idx + 1)
        area_activity = jnp.stack(
            [mf.get_brain_area(name, xs_seed, zs_seed) for xs_seed, zs_seed in zip(all_xs, all_zs)]
        ) # (n_seeds, n_conditions, T, N)
        area_activity = jnp.stack(
            [mf.align_to_cue(area_activity_seed, cs.test_start_t, new_T=50) for area_activity_seed in area_activity]
        )
        mean_act, sem_act = mf.compute_mean_sem(jnp.mean(area_activity, axis=3)) # (n_conditions, T)
        colors = plt.cm.coolwarm(jnp.linspace(0, 1, mean_act.shape[0]))
        for i in range(mean_act.shape[0]):
            ax.plot(mean_act[i], c=colors[i], label=f'Condition {i}')
            ax.fill_between(
                jnp.arange(mean_act.shape[1]),
                mean_act[i] - sem_act[i],
                mean_act[i] + sem_act[i],
                color=colors[i],
                alpha=0.3,
            )
        ymin = jnp.min(mean_act - sem_act)
        ymax = jnp.max(mean_act + sem_act)
        ax.vlines(cs.config['T_cue'], ymin, ymax, linestyles='dashed', label='Cue')
        ax.vlines(cs.config['T_cue'] + cs.config['T_wait'], ymin, ymax, linestyles='dashed', label='Wait')
        ax.vlines(
            cs.config['T_cue'] + cs.config['T_wait'] + cs.config['T_movement'],
            ymin,
            ymax,
            linestyles='dashed',
            label='Movement',
        )
        ax.set_title(f'{name} (aligned to cue)')
    plt.suptitle('Aligned to cue')
    plt.tight_layout()
    plt.show()

def plot_response_times(valid_response_times):
# Plot the distribution
    plt.figure(figsize=(6, 4))
    plt.hist(valid_response_times, bins=20, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel('Response Time')
    plt.ylabel('Frequency')
    plt.title('Distribution of Response Times')
    plt.tight_layout()
    plt.show()
    
    # Sort the response times
    sorted_response_times = jnp.sort(valid_response_times)
    
    # Compute the cumulative proportion of responses
    cumulative_proportion = jnp.arange(1, len(sorted_response_times) + 1) / len(sorted_response_times)
    
    # Plot the cumulative psychometric curve
    plt.figure(figsize=(6, 4))
    plt.plot(sorted_response_times, cumulative_proportion, marker='o', color='blue', alpha=0.7)
    plt.xlabel('Response Time (ms)')
    plt.ylabel('Cumulative Proportion of Responses')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_binned_responses(all_ys, all_xs, all_zs):
    response_times = jnp.full((cs.n_seeds, all_ys.shape[1]), jnp.nan)  # Default to NaN if no response is detected

    for seed_idx in range(cs.n_seeds):
        for condition_idx in range(all_ys.shape[1]):
            cue_end = cs.test_start_t[condition_idx] + cs.config['T_cue']
            post_cue_activity = all_ys[seed_idx, condition_idx, cue_end:]  # Activity after the cue
            response_idx = jnp.argmax(post_cue_activity[:, 0] > 0.5)  # Find first timestep where y > 0.5
            if post_cue_activity[response_idx, 0] > 0.5:
                response_times = response_times.at[seed_idx, condition_idx].set((response_idx) * 0.1)

    # Define the response time bins (left closed, right open)
    bin_boundaries = [1.6, 2.0, 2.2, 2.4, 2.8]
    bin_labels = [f'{bin_boundaries[i]}-{bin_boundaries[i+1]}' for i in range(len(bin_boundaries) - 1)]
    
    # Initialize lists for binning the xs, ys, zs data
    binned_xs = [[[] for n in range(3)] for _ in bin_labels]
    binned_ys = [[] for _ in bin_labels]
    binned_zs = [[] for _ in bin_labels]
    binned_response_times = [[] for _ in bin_labels]
    
    # Assign each trial to a bin based on its response time
    for seed_idx in range(cs.n_seeds):
    
        aligned_xs = [mf.align_to_cue(all_x[seed_idx], cs.test_start_t, new_T=50) for all_x in all_xs]
        aligned_zs = mf.align_to_cue(all_zs[seed_idx], cs.test_start_t, new_T=50)
        aligned_ys = mf.align_to_cue(all_ys[seed_idx], cs.test_start_t, new_T=50)

        for condition_idx in range(all_ys.shape[1]):
            response_time = response_times[seed_idx, condition_idx]
    
            # Find the corresponding bin for the current response time
            for bin_idx, (lower, upper) in enumerate(zip(bin_boundaries[:-1], bin_boundaries[1:])):
                if lower <= response_time < upper:
                    for i in range(3):
                        binned_xs[bin_idx][i].append(aligned_xs[i][condition_idx])
                    binned_ys[bin_idx].append(aligned_ys[condition_idx])
                    binned_zs[bin_idx].append(aligned_zs[condition_idx])
                    binned_response_times[bin_idx].append(response_time)
                    break
    
    # Convert lists to arrays
    binned_xs = [[jnp.array(bin_data) for bin_data in bin_xs] for bin_xs in binned_xs]
    binned_ys = [jnp.array(bin_data) for bin_data in binned_ys]
    binned_zs = [jnp.array(bin_data) for bin_data in binned_zs]
    
    # Print the shapes
    print("Shape of binned_xs:", [bin_data.shape for bin_data in binned_xs[0]])
    print("Shape of binned_ys:", [bin_data.shape for bin_data in binned_ys])
    
    # Plot output activity (mean ± SEM) for each response time bin
    # plt.close('all')
    fig = plt.figure(figsize=(6, 4))
    
    # Plot the activity for each response time bin
    for bin_idx, bin_data in enumerate(binned_ys):
        if len(bin_data) == 0:  # Skip empty bins
            continue
    
        # Compute mean and SEM for the current bin
        mean_ys, sem_ys = mf.compute_mean_sem(bin_data)  # Compute mean and SEM across trials
    
        # Plot each bin with mean ± SEM
        ax = plt.subplot(1, 1, 1)  # Plot on a single axis
        x_axis = 0.1 * jnp.array(range(mean_ys.shape[0]))
        ax.plot(x_axis, mean_ys[:, 0], label=f'{bin_labels[bin_idx]}', c=plt.cm.coolwarm(bin_idx / len(bin_labels)))
    
        # Plot the shaded region representing SEM
        ax.fill_between(
            x_axis,
            mean_ys[:, 0] - sem_ys[:, 0],
            mean_ys[:, 0] + sem_ys[:, 0],
            color=plt.cm.coolwarm(bin_idx / len(bin_labels)),
            alpha=0.3,
        )
    
    ax.set_title(f'Output Activity (mean ± SEM, noise_std={cs.test_noise_std})')
    ax.set_xlabel('Time after cue onset')
    ax.set_ylabel('Activity')
    ax.legend(title="Response Time")
    plt.tight_layout()
    plt.show()

    # Plot activity in each brain area for different response time bins (mean ± SEM) using binned xs and zs
    # plt.close('all')
    fig = plt.figure(figsize=(12, 8))
    
    # Define the brain areas to plot
    brain_areas = ['D1', 'D2', 'Cortex', 'Thalamus', 'SNc', 'nm']
    
    # Loop through each brain area
    for idx, name in enumerate(brain_areas):
        ax = plt.subplot(2, 3, idx + 1)
    
        # Collect the activity data from all bins and align to cue
        for bin_idx in range(len(binned_xs)):
            if len(bin_data) > 0:  # Only process non-empty bins
                aligned_xs = binned_xs[bin_idx]  # tuple of n_trials * T * n_neurons
                aligned_zs = binned_zs[bin_idx]
    
                # Get the brain area activity (aligning to the cue)
                area_activity = mf.get_brain_area(name, aligned_xs, aligned_zs) # trials * T * N
    
                # Compute mean and SEM for the current bin
                mean_area_activity = jnp.mean(area_activity, axis=-1) # trials * T
    
                # Plot each bin with mean ± SEM
                mean_act, sem_act = mf.compute_mean_sem(mean_area_activity) # T
                x_axis = jnp.array(range(mean_act.shape[0]))
                ax.plot(x_axis, mean_act, label=f'{bin_labels[bin_idx]}', c=plt.cm.coolwarm(bin_idx / len(bin_labels)))
                ax.fill_between(
                    x_axis,
                    mean_act - sem_act,
                    mean_act + sem_act,
                    alpha=0.3,
                    color=plt.cm.coolwarm(bin_idx / len(bin_labels)),
                )
    
    
        # Add vertical lines for cue, wait, and movement phases
        ymin = jnp.min(mean_act - sem_act)
        ymax = jnp.max(mean_act + sem_act)
        ax.vlines(cs.config['T_cue'], ymin, ymax, linestyles='dashed')
        ax.vlines(cs.config['T_cue'] + cs.config['T_wait'], ymin, ymax, linestyles='dashed')
        ax.vlines(
            cs.config['T_cue'] + cs.config['T_wait'] + cs.config['T_movement'],
            ymin,
            ymax,
            linestyles='dashed',
        )
    
        # Set titles and labels for each plot
        ax.set_title(f'{name} (aligned to cue)')
        ax.set_xlabel('Time after cue onset')
        ax.set_ylabel('Activity')
        ax.legend(title="Response Time")
    
    plt.suptitle('Aligned to cue (by response time bins)')
    plt.tight_layout()
    plt.show()
