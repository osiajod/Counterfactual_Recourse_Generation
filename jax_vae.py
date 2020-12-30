from collections import namedtuple
from functools import partial

from jax import random, value_and_grad, jit
from jax.experimental.optimizers import adam
import jax.numpy as jnp
import jax.scipy as jsp
from tqdm.auto import trange

from jax_nn import create_mlp

VAE = namedtuple('VAE', [
    'encode', 'raw_encode',
    'decode', 'raw_decode',
    'generate',
    'fit',
    'params', 'n_latent_dims', 'x_var'])

def create_vae(rng_key, enc_params, dec_params):
    encoder = create_mlp(rng_key, **enc_params)
    encode = jit(encoder.raw_predict)
    enc_start_params = encoder.params
    
    decoder = create_mlp(rng_key, **dec_params)
    decode = jit(decoder.raw_predict)
    dec_start_params = decoder.params

    vae = VAE(partial(encode, enc_start_params), encode,
              partial(decode, dec_start_params), decode,
              None,
              None,
              (enc_start_params, dec_start_params),
              enc_params['output_dim'] // 2, None)
    
    return vae._replace(generate=partial(generate_samples, vae),
                 fit=partial(fit, vae))

@jit
def unpack_latent_params(z):
    z_dim = z.shape[1] // 2
    mean = z[:,:z_dim]
    std = jnp.exp(z[:,z_dim:])
    return mean, std

def log_likelihood_data(vae, rng_key, params, data, data_vari, n_latent_samples):
    '''Expected log-likelihood of data under current model'''
    encode, decode = vae.raw_encode, vae.raw_decode
    enc_params, dec_params = params
    latent_params = encode(enc_params, data)
    mean, std = unpack_latent_params(latent_params)
    latent_dim = mean.shape[1]
    n_obs = data.shape[0]
    latent_samples = random.normal(rng_key, shape=(n_latent_samples, n_obs, latent_dim)) * std + mean
    reconstructed_data = decode(dec_params, latent_samples)
    log_likelihood = jsp.stats.norm.logpdf(data, reconstructed_data, data_vari**0.5).sum(axis=-1)
    assert log_likelihood.shape == (n_latent_samples, n_obs)
    return log_likelihood, latent_samples, mean, std

@jit
def log_prob_latent_under_prior(latent_samples):
    ''' Log-probability of latent vars under N(0,1) prior '''
    return jsp.stats.norm.logpdf(latent_samples, 0, 1).sum(axis=-1)

@jit
def log_prob_latent_under_variational(latent_samples, mean, std):
    ''' Log-probability of latent vars under variational dist '''
    return jsp.stats.norm.logpdf(latent_samples, mean, std).sum(axis=-1)

def objective(params, vae, rng_key, data, data_vari, n_latent_samples=50):
    log_likelihood, latent_samples, mean, std = log_likelihood_data(
        vae, rng_key, params, data, data_vari, n_latent_samples)
    log_prior_prob_latent = log_prob_latent_under_prior(latent_samples)
    log_variational_prob_latent = log_prob_latent_under_variational(latent_samples, mean, std)
    
    # Estimated expected value
    elbo = (log_likelihood - log_variational_prob_latent + log_prior_prob_latent).mean()
    return -elbo

@jit
def latent_dim_from_params(params):
    # Locate the number of output layer biases
    return len(params[-1][-2][-1])

def generate_samples(vae, rng_key, n=100):
    latent_samples = random.normal(rng_key, (n, vae.n_latent_dims))
    return vae.decode(latent_samples)

def fit(vae, rng_key, data, data_vari, step_size=1e-3, max_iter=1000):
    '''
    Args:
      *data: array like (obs, features)
    '''
    
    start_params = vae.params
    opt_init, update_params, get_params = adam(step_size)
    opt_state = opt_init(start_params)
    history = []
    min_loss_params = (1e10, None)
    for i in trange(max_iter, smoothing=0):
        params = get_params(opt_state)
        rng_key, subkey = random.split(rng_key)
        loss, grads = value_and_grad(objective)(params, vae, subkey, data, data_vari)
        opt_state = update_params(i, grads, opt_state)
        if loss < min_loss_params[0]:
            min_loss_params = (loss, params)
        history.append(float(loss))
        
    vae = VAE(partial(vae.raw_encode, min_loss_params[1][0]), vae.raw_encode,
              partial(vae.raw_decode, min_loss_params[1][1]), vae.raw_decode,
              None,
              None,
              min_loss_params[1], vae.n_latent_dims, data_vari)
    
    return vae._replace(generate=partial(generate_samples, vae),
                 fit=partial(fit, vae)), history