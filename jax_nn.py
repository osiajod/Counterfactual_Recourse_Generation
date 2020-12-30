from functools import partial
from collections import namedtuple

from jax import value_and_grad, random, jit
from jax.experimental import stax
from jax.experimental.optimizers import adam
import jax.scipy as jsp
import jax.numpy as jnp
from tqdm.auto import trange

Classifier = namedtuple('Classifier', ['predict', 'raw_predict', 'fit', 'raw_fit', 'params'])

def create_mlp(rng_key, input_dim, hidden_widths, activation_fn, output_dim, output_activation_fn=stax.Identity):
    layers = []
    for width in hidden_widths:
        layers.extend([stax.Dense(width), activation_fn])
    layers.extend([stax.Dense(output_dim), output_activation_fn])
    init_model, predict = stax.serial(*layers)
    predict = jit(predict)
    output_shape, start_params = init_model(rng_key, (-1, input_dim))
    classifier = Classifier(
        partial(predict, start_params),
        predict,
        None,
        fit,
        start_params)
    return classifier._replace(fit=partial(fit, classifier))

def binary_crossentropy_loss(params, predict, data):
    inputs, targets = data
    probs = predict(params, inputs)
    eps = jnp.finfo(probs.dtype).eps
    probs = jnp.clip(probs, eps, 1 - eps)
    loss = -(jsp.special.xlogy(targets, probs) + jsp.special.xlogy(1 - targets, 1 - probs)).mean()
    return loss

@jit
def _binary_crossentropy_loss(targets, probs):
    return -(jsp.special.xlogy(targets, probs) + jsp.special.xlogy(1 - targets, 1 - probs)).mean()

def mse_loss(params, predict, data):
    inputs, targets = data
    preds = predict(params, inputs)
    loss = ((preds - targets)**2).mean()
    return loss

def fit(classifier, calc_loss, data, step_size=1e-3, max_iter=1000):
    '''
    Args:
      *calc_loss: function to calculate loss of predictions
      *data: data like (X, y)
    '''
    start_params = classifier.params
    predict = classifier.raw_predict
    opt_init, update_params, get_params = adam(step_size)
    opt_state = opt_init(start_params)
    history = []
    min_loss_params = (1e10, None)
    for i in trange(max_iter, smoothing=0):
        params = get_params(opt_state)
        loss, grads = fit_step(predict, calc_loss, params, data)
        opt_state = update_params(i, grads, opt_state)
        output_layer_weights = grads[-2][0]
        output_layer_weight_mag = ((output_layer_weights.T@output_layer_weights)**0.5)[0][0]
        if loss < min_loss_params[0]:
            min_loss_params = (loss, params)
        history.append((float(loss), float(output_layer_weight_mag)))
    return Classifier(
        partial(predict, min_loss_params[1]),
        predict,
        partial(fit, predict),
        fit,
        min_loss_params[1]), history

def fit_step(predict, calc_loss, params, data):
    loss, grads = value_and_grad(calc_loss)(params, predict, data)
    return loss, grads