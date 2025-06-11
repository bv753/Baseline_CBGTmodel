import jax.random as jr
import jax.numpy as jnp
import math
import optax

def init_params(key, n_bg, n_nm, g_bg, g_nm, input_dim, output_dim):
    # for now assume Th/BG/C are same size, g is the same for all weight matrices
    # n refers to the number of units in the node
    # g refers to a scaling factor for connectivity

    skeys = jr.split(key, 17)

    # bg parameters ("striatum pallidum")
    # auto-recurrence of BG neurons
    # jr.normal generates a (n_bg x n_bg) matrix of random connectivity values
    # to start, using skeys as random value seeds
    J_bg = (g_bg / math.sqrt(n_bg)) * jr.normal(skeys[0], (n_bg, n_bg))

    #BG-cortex connectivity
    B_bgc = (g_bg / math.sqrt(n_bg)) * jr.normal(skeys[1], (n_bg, n_bg))

    # cortex parameters
    J_c = (g_bg / math.sqrt(n_bg)) * jr.normal(skeys[2], (n_bg, n_bg))
    B_cu = (1 / math.sqrt(input_dim)) * jr.normal(skeys[3], (n_bg, input_dim))
    B_ct = (g_bg / math.sqrt(n_bg)) * jr.normal(skeys[4], (n_bg, n_bg))

    # thalamus parameters
    J_t = (g_bg / math.sqrt(n_bg)) * jr.normal(skeys[5], (n_bg, n_bg))
    B_tbg = (g_bg / math.sqrt(n_bg)) * jr.normal(skeys[6], (n_bg, n_bg))

    # "neuromodulatory region" (SNc) parameters
    J_nm = (g_nm / math.sqrt(n_nm)) * jr.normal(skeys[7], (n_nm, n_nm))
    J_nmc = (g_nm / math.sqrt(n_nm)) * jr.normal(skeys[8], (n_nm, n_bg))
    B_nmc = (1 / math.sqrt(n_nm)) * jr.normal(skeys[9], (n_nm, n_bg))
    # understanding: SNc modulation modeled as input to cortex to effectively
    # represent modulation of input to striatum

    m = (1 / math.sqrt(n_nm)) * jr.normal(skeys[10], (1, n_nm))
    c = (1 / math.sqrt(n_nm)) * jr.normal(skeys[11])

    U = (1 / math.sqrt(n_bg)) * jr.normal(skeys[12], (1, n_bg))
    V_bg = (1 / math.sqrt(n_bg)) * jr.normal(skeys[13], (1, n_bg))
    V_c = (1 / math.sqrt(n_bg)) * jr.normal(skeys[14], (1, n_bg))

    # readout params
    C = (1 / math.sqrt(n_bg)) * jr.normal(skeys[15], (output_dim, n_bg))
    rb = (1 / math.sqrt(n_bg)) * jr.normal(skeys[16], (output_dim, ))

    return {
        'J_bg': J_bg,
        'B_bgc': B_bgc,
        'J_c': J_c,
        'B_cu': B_cu,
        'B_ct': B_ct,
        'J_t': J_t,
        'B_tbg': B_tbg,
        'J_nm': J_nm,
        'J_nmc': J_nmc,
        'B_nmc': B_nmc,
        'm': m,
        'c': c,
        'C': C,
        'rb': rb,
        'U': U,
        'V_bg': V_bg,
        'V_c': V_c
    }

# parameters we want to track in wandb
default_config = dict(
    # model parameters
    n_bg = 20,
    n_nm = 5,      # NM (SNc) dimension
    g_bg = 1.4,
    g_nm = 1.4,
    U = 1,      # input dim
    O = 1,      # output dimension
    # Model Hyperparameters
    tau_x = 10,
    tau_z = 100,
    noise_std=0.1,  # Standard deviation of noise
    # Timing (task) parameters
    dt = 10, # ms
    # Data Generation
    T_start = jnp.arange(0, 10, 1),
    T_cue = 10,
    T_wait = 20,
    T_movement = 10,
    T = 70,
    # Training
    num_nm_only_iters = 0,
    num_full_train_iters = 3000,
    keyind = 13,
)

# declare the config
config = default_config

# set up the random key
key = jr.PRNGKey(config['keyind'])

# define a simple optimizer
# optimizer = optax.adam(learning_rate=1e-3)
optimizer = optax.chain(
  optax.clip(1.0), # gradient clipping
  optax.adamw(learning_rate=1e-3),
)

x_bg0 = jnp.ones((config['n_bg'],)) * 0
x_c0 = jnp.ones((config['n_bg'],)) * 0
x_t0 = jnp.ones((config['n_bg'],)) * 0
x0 = (x_bg0, x_c0, x_t0)
z0 = jnp.ones((config['n_nm'],)) * 0

# generate random initial parameters
params = init_params(
    key,
    config['n_bg'], config['n_nm'],
    config['g_bg'], config['g_nm'],
    config['U'], config['O']
)

# test parameters
n_seeds = 40
test_noise_std = 0.2
test_start_t = jnp.arange(0, 10, 2)
