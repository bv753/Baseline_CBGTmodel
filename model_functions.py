import jax
import jax.random as jr
import jax.numpy as jnp
import math

from jax import vmap, jit
from jax import lax
import optax
import config_script as cs

def exc(w):
    return jnp.abs(w)

def inh(w):
    return -jnp.abs(w)

def nln(x):
    # return jnp.maximum(0, x)
    return jnp.tanh(x)
    # return jax.nn.sigmoid(x)

def multiregion_nmrnn(
    params, x_0, z_0, inputs, tau_x, tau_z, modulation=True, opto_stimulation=None, noise_std=0, rng_key=None
):
    """
    Arguments:
    - params
    - x_0: initial states (tuple of x_bg0, x_c0, x_t0)
    - z_0: initial state for x_nm
    - inputs: input sequence (task related signals)
    - tau_x, tau_z: decay constants for x and z
    - modulation: whether to apply neuromodulation
    - opto_stimulation: optional external stimulation
    - noise_std: standard deviation of the noise to add
    - rng_key: JAX random key for reproducibility
    """
    # Unpack initial states
    x_bg0, x_c0, x_t0 = x_0
    x_nm0 = z_0

    # Initialize random keys
    rng_key = jr.PRNGKey(0) if rng_key is None else rng_key
    rng_key, init_key, step_key = jr.split(rng_key, 3)

    # Add noise to initial states
    if noise_std > 0:
        x_bg0 += noise_std * jr.normal(init_key, x_bg0.shape)
        x_c0 += noise_std * jr.normal(init_key, x_c0.shape)
        x_t0 += noise_std * jr.normal(init_key, x_t0.shape)
        x_nm0 += noise_std * jr.normal(init_key, x_nm0.shape)

    J_bg = params['J_bg']
    B_bgc = params['B_bgc']
    J_c = params['J_c']
    B_cu = params['B_cu']
    B_ct = params['B_ct']
    J_t = params['J_t']
    B_tbg = params['B_tbg']
    J_nm = params['J_nm']
    J_nmc = params['J_nmc']
    B_nmc = params['B_nmc']
    m = params['m']
    c = params['c']
    C = params['C']
    rb = params['rb']
    U = params['U']
    V_bg = params['V_bg']
    V_c = params['V_c']

    tau_c = tau_x
    tau_bg = tau_x
    tau_t = tau_x
    tau_nm = tau_z

    num_bg_cells = J_bg.shape[0]
    num_c_cells = J_c.shape[0]
    num_t_cells = J_t.shape[0]
    num_nm_cells = J_nm.shape[0]
    n_d1_cells = num_bg_cells // 2
    n_d2_cells = num_bg_cells - n_d1_cells
    T = inputs.shape[0]

    if opto_stimulation is None:
        opto_stimulation = jnp.zeros((T, num_bg_cells))
    inputs_and_stim = (inputs, opto_stimulation)

    def _step(x_and_z, u_and_stim, step_rng_key):
        x_bg, x_c, x_t, x_nm = x_and_z
        u, stim = u_and_stim

        # Add noise to the states
        if noise_std > 0:
            coef = noise_std / math.sqrt(2 * tau_x)
            x_bg += coef * jr.normal(step_rng_key, x_bg.shape)
            x_c += coef * jr.normal(step_rng_key, x_c.shape)
            x_t += coef * jr.normal(step_rng_key, x_t.shape)
            coefz = noise_std / math.sqrt(2 * tau_z)
            x_nm += coefz * jr.normal(step_rng_key, x_nm.shape)

        # update x_c
        x_c = (1.0 - (1. / tau_c)) * x_c + (1. / tau_c) * J_c @ nln(x_c) # recurrent dynamics
        x_c += (1. / tau_c) * B_cu @ u # external inputs
        x_c += (1. / tau_c) * exc(B_ct) @ nln(x_t) # input from thalamus, excitatory

        if modulation:
            U = jnp.concatenate((jnp.ones((n_d1_cells, 1)), -1 * jnp.ones((n_d2_cells, 1)))) # direct/indirect
            V_bg = jnp.ones((num_bg_cells, 1))
            V_c = jnp.ones((num_c_cells, 1))
            s = jax.nn.sigmoid(exc(m) @ nln(x_nm) + c) # neuromodulatory signal (1D for now)
            G_bg = jnp.exp(s * U @ V_bg.T) # TODO: change to matrix U, V + vector s (for multidimensional NM)
            G_c = jnp.exp(s * U @ V_c.T) # num_bg_cells x num_c_cells
        else:
            G_bg = jnp.ones((num_bg_cells, num_bg_cells))
            G_c = jnp.ones((num_bg_cells, num_c_cells))

        x_bg = (1.0 - (1. / tau_bg)) * x_bg + (1. / tau_bg) * (G_bg * inh(J_bg)) @ nln(x_bg) # recurrent dynamics, inhibitory
        x_bg += (1. / tau_bg) * (G_c * exc(B_bgc)) @ nln(x_c) # input from cortex, excitatory
        x_bg += (1. / tau_bg) * stim # simulate stimulation

        # update x_t
        x_t = (1.0 - (1. / tau_t)) * x_t + (1. / tau_t) * J_t @ nln(x_t) # recurrent dynamics
        tbg = jnp.concatenate((exc(B_tbg[:, : n_d1_cells]), inh(B_tbg[:, n_d1_cells: ])), axis=1) # two subpopulations have the opposite net effects
        x_t += (1. / tau_t) * tbg @ nln(x_bg) # input from BG, inhibitory

        # update x_nm
        x_nm = (1.0 - (1. / tau_nm)) * x_nm + (1. / tau_nm) * J_nm @ nln(x_nm)
        x_nm += (1. / tau_nm) * exc(B_nmc) @ nln(x_c) # input from cortex, excitatory

        # calculate y
        y = exc(C) @ nln(x_t) + rb # output from Thalamus
        return (x_bg, x_c, x_t, x_nm), (y, x_bg, x_c, x_t, x_nm)

    # Generate random keys for each time step
    step_keys = jr.split(step_key, T)

    _, (y, x_bg, x_c, x_t, x_nm) = lax.scan(
        lambda x_and_z, u_and_stim_rng: _step(x_and_z, u_and_stim_rng[:2], u_and_stim_rng[2]),
        (x_bg0, x_c0, x_t0, x_nm0),
        (inputs_and_stim[0], inputs_and_stim[1], step_keys),
    )

    return y, (x_bg, x_c, x_t), x_nm

# Update batched_nm_rnn to accept random keys
batched_nm_rnn = vmap(
    multiregion_nmrnn,
    in_axes=(None, None, None, 0, None, None, None, None, None, 0)  # Add random key as batched input
)

def batched_nm_rnn_loss(params, x0, z0, batch_inputs, tau_x, tau_z, batch_targets, batch_mask, rng_keys, orth_u=True, modulation=True, noise_std=0):
    ys, _, _ = batched_nm_rnn(params, x0, z0, batch_inputs, tau_x, tau_z, modulation, None, noise_std, rng_keys)
    return jnp.sum(((ys - batch_targets) ** 2) * batch_mask) / jnp.sum(batch_mask)

def fit_nm_rnn(inputs, targets, loss_masks, params, optimizer, x0, z0, num_iters, tau_x, tau_z,
               wandb_log=False, orth_u=True, modulation=True, log_interval=200, noise_std=0.1):
    opt_state = optimizer.init(params)
    N_data = inputs.shape[0]

    rng_key = jr.PRNGKey(0)  # Initialize random key

    @jit
    def _step(params_and_opt, _):
        nonlocal rng_key
        (params, opt_state) = params_and_opt

        # Generate random keys for the batch
        rng_key, subkey = jr.split(rng_key)
        batch_rng_keys = jr.split(subkey, N_data)

        # Compute loss and gradients
        loss_value, grads = jax.value_and_grad(batched_nm_rnn_loss)(
            params, x0, z0, inputs, tau_x, tau_z, targets, loss_masks, batch_rng_keys, orth_u=orth_u, modulation=modulation, noise_std=noise_std
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), (params, loss_value)

    losses = []
    best_loss = float('inf')
    best_params = params

    for n in range(num_iters // log_interval):
        (params, _), (_, loss_values) = lax.scan(
            _step, (params, opt_state), None, length=log_interval
        )
        losses.append(loss_values)
        print(f'step {(n + 1) * log_interval}, loss: {loss_values[-1]}')
        #if wandb_log:
        #    wandb.log({'loss': loss_values[-1]})
        if loss_values[-1] < best_loss:
            best_params = params
            best_loss = loss_values[-1]

    return best_params, losses

def self_timed_movement_task(T_start, T_cue, T_wait, T_movement, T, null_trial=False):
    """
    Simulate all possible input/output pairs for the self-timed movement task.

    Arguments:
    T_start: a list of possible time before the cue
    T_cue: duration of the cue
    T_wait: duration of the wait
    T_movement: duration of the movement
    T: total trial time, should not be at least T_start + T_cue + T_wait + T_movement

    Returns:
    inputs: (num_starts, T, 1), binary time series representing the cue
    outputs: (num_starts, T, 1), representing the desired movement
    mask: (num_starts, T, 1), the loss mask
    """

    if null_trial is True:
        # t_start_def = jnp.concatenate((T_start, jnp.zeros(jnp.shape(T_start))), dtype=int)
        t_start_def = jnp.insert(T_start, jr.randint(jr.PRNGKey(0), (len(T_start),), 0, len(T_start)), 0)
    else:
        t_start_def = T_start
    print("Trials ('0': null-trials):", t_start_def)

    num_starts = t_start_def.shape[0]

    def _single(interval_ind):
        t_start = int(t_start_def[interval_ind] / cs.test_start_step)
        t_cue_end = t_start + T_cue
        t_wait_end = t_cue_end + T_wait
        t_movement_end = t_wait_end + T_movement

        # Initialize zero arrays for inputs, outputs, and masks
        inputs = jnp.zeros((1, T, 1))
        outputs = jnp.zeros((1, T, 1))
        mask = jnp.ones((1, T, 1))

        # Dynamically update the slices
        inputs = jax.lax.dynamic_update_slice(inputs, jnp.ones((1, T_cue, 1)), (0, t_start, 0))
        outputs = jax.lax.dynamic_update_slice(outputs, jnp.ones((1, T_movement, 1)), (0, t_wait_end, 0))

        return inputs, outputs, mask

    # stmt output with null_trials
    inputs = jnp.empty((0, T, 1))
    outputs = jnp.empty((0, T, 1))
    masks = jnp.ones((num_starts, T, 1))

    for ind, trial in enumerate(t_start_def):
        if trial == 0:
            input = jnp.zeros((1, T, 1))
            output = jnp.zeros((1, T, 1))
        else:
            input, output, mask = _single(ind)

        inputs = jnp.append(inputs, input, 0)
        outputs = jnp.append(outputs, output, 0)

    reg_indices = jnp.array(jnp.where(t_start_def != 0), )[0]
    print("Regular trial indices:", reg_indices)

    #inputs, outputs, masks = vmap(_single)(jnp.arange(num_starts))
    return inputs, outputs, masks, reg_indices

def get_response_times(all_ys, exclude_nan=True):
    response_times = jnp.full((cs.n_seeds, all_ys.shape[1]), jnp.nan)  # Default to NaN if no response is detected

    for seed_idx in range(cs.n_seeds):
        for condition_idx in range(all_ys.shape[1]):
            cue_end = cs.test_start_t[condition_idx] + cs.config['T_cue']
            post_cue_activity = all_ys[seed_idx, condition_idx, cue_end:]  # Activity after the cue
            response_idx = jnp.argmax(post_cue_activity[:, 0] > 0.5)  # Find first timestep where y > 0.5
            if post_cue_activity[response_idx, 0] > 0.5:
                response_times = response_times.at[seed_idx, condition_idx].set((response_idx) * 0.1)

    # Flatten the response_times array, excluding NaN values
    if exclude_nan:
        valid_response_times = response_times[~jnp.isnan(response_times)].flatten()
    else:
        #replace NaN with T
        valid_response_times = response_times#.flatten()
    return valid_response_times

#function to align trials to cue for plotting
def align_to_cue(data, cue_start, new_T=50):
    """
    align data to the cue
    data: shape (n_conditions, T, N) or (n_conditions, T)
    cue_start: shape (n_conditions,)
    return: shape (n_conditions, new_T, N) or (n_conditions, new_T)
    """
    n_conditions = data.shape[0]
    T = data.shape[1]
    new_data = []

    for i, t in enumerate(cue_start):
        ind_range = jnp.arange(T)
        mask = (ind_range >= t) & (ind_range < t + new_T)
        new_data.append(data[i, mask])

    return jnp.stack(new_data)

#querys the array corresponding to the right brain area from the output
def get_brain_area_(brain_area, xs=None, zs=None):
    if brain_area == 'BG' or brain_area == 'Striatum':
        return jnp.concatenate((get_brain_area('D1', xs, zs), get_brain_area('D2', xs, zs)), axis=0)
    elif brain_area == 'Cortex':
        return xs[1]
    elif brain_area == 'Thalamus':
        return xs[2]
    elif brain_area == 'SNc':
        return zs
    elif brain_area == 'All':
        return jnp.concatenate((xs[0], xs[1], xs[2], zs), axis=0)
    elif brain_area == 'D1':
        return xs[0][:, :, :cs.n_d1_cells]
    elif brain_area == 'D2':
        return xs[0][:, :, cs.n_d1_cells:]
    elif brain_area == 'nm':
        return jax.nn.sigmoid(nln(zs) @ exc(cs.params['m'].T) + cs.params['c'])
    else:
        raise ValueError('Invalid brain area')

#nonlinear transformation of output data to normalize
def get_brain_area(brain_area, xs=None, zs=None):
    return jnp.tanh(get_brain_area_(brain_area, xs, zs))

#calculate the sem for plotting
def sem(data, axis=0):
    return jnp.std(data, axis=axis) / jnp.sqrt(data.shape[axis] - 1)

def test_model(params_nm):
    all_inputs, all_outputs, all_masks, _ = self_timed_movement_task(
    cs.test_start_t, cs.config['T_cue'], cs.config['T_wait'], cs.config['T_movement'], cs.config['T'], null_trial=cs.test_null_trials
    )

    # Collect activity for each seed
    all_ys = []
    all_xs = []
    all_zs = []
    
    for seed in range(cs.n_seeds):
        rng_key = jr.PRNGKey(seed)
        batched_rng_key = jr.split(rng_key, all_inputs.shape[0])
        ys, xs, zs = batched_nm_rnn(
            params_nm, cs.x0, cs.z0, all_inputs, cs.config['tau_x'], cs.config['tau_z'], True, None, cs.test_noise_std, batched_rng_key
        )
        all_ys.append(ys)
        all_xs.append(xs)
        all_zs.append(zs)
    
    # Convert collected data to arrays
    all_ys = jnp.stack(all_ys)  # shape: (n_seeds, n_conditions, T, output_dim)
    all_xs = [jnp.stack(xs) for xs in zip(*all_xs)]  # shape: (n_seeds, n_conditions, T, N_xs)
    all_zs = jnp.stack(all_zs)  # shape: (n_seeds, n_conditions, T, N_zs)
    
    return all_ys, all_xs, all_zs

# Helper function to calculate mean ± SEM
def compute_mean_sem(data):
    return jnp.mean(data, axis=0), sem(data, axis=0)


