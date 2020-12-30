from collections import namedtuple
from functools import partial

from jax import jit, random, value_and_grad
from jax.experimental.optimizers import adam
import jax.numpy as jnp
from tqdm.auto import trange

from jax_nn import create_mlp

# X: predictors
# Y: response
# W: latent vars that predict response
# Z: latent vars that don't predict response

CSVAE = namedtuple('CSVAE',
                   ['w_encode', 'w_encode_raw',
                    'z_encode', 'z_encode_raw',
                    'x_decode', 'x_decode_raw',
                    'y_decode', 'y_decode_raw',
                    'fit', 'fit_raw',
                    'params', 'n_w_dim', 'n_z_dim'])

W_PRIORS = [
    (jnp.array((0.0, 0.0)), jnp.array((0.1, 0.1))),
    (jnp.array((3.0, 3.0)), jnp.array((1.0, 1.0)))]

BETA = [20, 1, 0.2, 10, 1]
    
def _setup_mlp(rng_key, params):
    network = create_mlp(rng_key, **params)
    predict = jit(network.raw_predict)
    start_params = network.params
    return (network, predict, start_params)


def create_csvae(rng_key, w_enc_params, z_enc_params, x_dec_params, y_dec_params):
    w_encoder, w_encode, w_enc_start_params = _setup_mlp(rng_key, w_enc_params)
    z_encoder, z_encode, z_enc_start_params = _setup_mlp(rng_key, z_enc_params)
    x_decoder, x_decode, x_dec_start_params = _setup_mlp(rng_key, x_dec_params)
    y_decoder, y_decode, y_dec_start_params = _setup_mlp(rng_key, y_dec_params)
    
    params = (w_enc_start_params, z_enc_start_params, x_dec_start_params, y_dec_start_params)
    csvae = CSVAE(
        partial(jit(w_encode), w_enc_start_params), jit(w_encode),
        partial(jit(z_encode), z_enc_start_params), jit(z_encode),
        partial(jit(x_decode), x_dec_start_params), jit(x_decode),
        partial(jit(y_decode), y_dec_start_params), jit(y_decode),
        None, fit,
        params,
        w_enc_params['output_dim'] // 2, z_enc_params['output_dim'] // 2) 
    
    return csvae._replace(fit=partial(fit, csvae))


def w_z_and_kls(params, rng_key, csvae, data):
    x, y = data
    w_encode, z_encode = csvae.w_encode_raw, csvae.z_encode_raw
    x_decode, y_decode = csvae.x_decode_raw, csvae.y_decode_raw
    w_enc_params, z_enc_params, x_dec_params, y_dec_params = params
    
    # W stuff
    w = w_encode(w_enc_params, jnp.hstack((x, y.reshape(-1,1))))
    w_mean, w_var = w[:,:csvae.n_w_dim], jnp.exp(w[:,csvae.n_w_dim:])**2
    w_sample = random.normal(rng_key, w_mean.shape) * w_var**0.5 + w_mean
    
    w_prior_y1_mean = jnp.ones_like(w_mean) * W_PRIORS[0][0]
    w_prior_y1_var = jnp.ones_like(w_var) * W_PRIORS[0][1]
    w_prior_y0_mean = jnp.ones_like(w_mean) * W_PRIORS[1][0]
    w_prior_y0_var = jnp.ones_like(w_var) * W_PRIORS[1][1]
    
    w_kl_y1 = kl(w_mean, w_var, w_prior_y1_mean, w_prior_y1_var)
    w_kl_y0 = kl(w_mean, w_var, w_prior_y0_mean, w_prior_y0_var)
    
    # Z stuff
    z = z_encode(z_enc_params, x)
    z_mean, z_var = z[:,:csvae.n_z_dim], jnp.exp(z[:,csvae.n_z_dim:])**2
    z_sample = random.normal(rng_key, z_mean.shape) * z_var**0.5 + z_mean
    
    z_prior_mean = jnp.zeros_like(z_mean)
    z_prior_var = jnp.ones_like(z_var)
    
    z_kl = kl(z_mean, z_var, z_prior_mean, z_prior_var)
    return (w_sample, z_sample, w_kl_y1, w_kl_y0, z_kl)


def objective_1(params, y_dec_params, rng_key, csvae, data):
    x, y = data
    x_decode, y_decode = csvae.x_decode_raw, csvae.y_decode_raw
    _, _, x_dec_params = params
    
    w, z, w_kl_y1, w_kl_y0, z_kl = w_z_and_kls((*params, y_dec_params), rng_key, csvae, data)
    
    # X stuff
    reconstructed_x = x_decode(x_dec_params, jnp.hstack((w, z)))
    x_reconstruction_loss = ((x - reconstructed_x)**2).sum(axis=1).mean()
    
    # Y stuff
    reconstructed_y = y_decode(y_dec_params, z)
    y_cross_entropy = (jnp.log(reconstructed_y) * reconstructed_y
                             + jnp.log(1 - reconstructed_y) * (1 - reconstructed_y)).mean()
    
    terms = [
        BETA[0] * x_reconstruction_loss,
        BETA[1] * jnp.where(y == 1, w_kl_y1, w_kl_y0).mean(),
        BETA[2] * z_kl.mean(),
        BETA[3] * y_cross_entropy
    # We don't need the last term to do the optimization
    ]
    #print([t for t in terms])
    return jnp.sum(jnp.array(terms))


def objective_2(params, other_params, rng_key, csvae, data):
    y = data[1]
    y_decode = csvae.y_decode_raw
    
    _, z, _, _, _ = w_z_and_kls((*other_params, params), rng_key, csvae, data)
    
    reconstructed_y = y_decode(params, z)
    # Binary cross-entropy loss
    y_reconstruction_loss = jnp.where(y == 1,
                                     jnp.log(reconstructed_y),
                                     jnp.log(1 - reconstructed_y)).mean()
    return -BETA[4] * y_reconstruction_loss
    
    
def fit(csvae, rng_key, data, max_iter=500, step_size=1e-3):
    start_params = csvae.params
    
    obj1_opt_init, obj1_update_params, obj1_get_params = adam(step_size)
    obj1_opt_state = obj1_opt_init(start_params[:3])
    
    obj2_opt_init, obj2_update_params, obj2_get_params = adam(step_size)
    obj2_opt_state = obj2_opt_init(start_params[3])
    
    history = []
    min_loss_params = (1e10, None)
    for i in trange(max_iter, smoothing=0):
        obj1_params = obj1_get_params(obj1_opt_state)
        obj2_params = obj2_get_params(obj2_opt_state)
        rng_key, subkey = random.split(rng_key)
        obj1_loss, obj1_grads = value_and_grad(objective_1)(
            obj1_params, obj2_params, subkey, csvae, data)
        obj2_loss, obj2_grads = value_and_grad(objective_2)(
            obj2_params, obj1_params, subkey, csvae, data)
        obj1_opt_state = obj1_update_params(i, obj1_grads, obj1_opt_state)
        obj2_opt_state = obj2_update_params(i, obj2_grads, obj2_opt_state)
        loss = obj1_loss + obj2_loss
        if loss < min_loss_params[0]:
            min_loss_params = (loss, (*obj1_params, obj2_params))
        history.append((obj1_loss, obj2_loss))
        
    w_enc_params, z_enc_params, x_dec_params, y_dec_params = min_loss_params[1]
    csvae = CSVAE(
        partial(csvae.w_encode_raw, w_enc_params), csvae.w_encode_raw,
        partial(csvae.z_encode_raw, z_enc_params), csvae.z_encode_raw,
        partial(csvae.x_decode_raw, x_dec_params), csvae.x_decode_raw,
        partial(csvae.y_decode_raw, y_dec_params), csvae.y_decode_raw,
        None, csvae.fit_raw,
        min_loss_params[1],
        csvae.n_w_dim, csvae.n_z_dim)
    csvae = csvae._replace(fit=partial(fit, csvae))
    return csvae, history
    
    
# From https://github.com/qq456cvb/CSVAE/blob/master/main.py#L15
@jit
def kl(mu0, var0, mu1, var1):
    std0 = var0**0.5
    std1 = var1**0.5
    return (jnp.log(std1) - jnp.log(std0) + 0.5 * (var0 + (mu0 - mu1) ** 2) / var1 - 0.5).sum(axis=-1)

# From https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
# This doesn't work and could also be simplified.
def kl_my_own(mu0, var0, mu1, var1):
    det0, det1 = var0.prod(), var1.prod()
    k = mu0.shape[1]
    return (
        ((1/var1)*(var0)).sum(axis=1)
         + ((1/var1)*(mu1-mu0)**2).sum(axis=1)
         - k
         + jnp.log(det1/det0)
    ) / 2
    
    