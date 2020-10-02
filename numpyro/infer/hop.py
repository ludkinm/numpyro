import warnings

from jax import random, grad, value_and_grad, vmap, device_put

import jax.numpy as jnp

from numpyro.infer.hug import preconditioner, HState
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import ParamInfo, init_to_uniform, initialize_model
from numpyro.util import cond, identity


def hoppy(mu, lam, z, z_grad, z_pe, preconditioner, potential_fn, rng_key):
    # Sample w from the preconditioner
    w = preconditioner.sample(rng_key)

    # unravel z, g and w from tree to vector
    z = preconditioner.unravel(z)
    g = preconditioner.unravel(z_grad)

    # propose new z value using hop
    Sg = preconditioner.condition(z_grad)
    gSg = jnp.dot(g, Sg)
    rho2 = jnp.clip(gSg, a_min=1.0)

    # proposal offset:
    r = (mu * w + (lam - mu) * jnp.dot(g, w) / gSg * Sg) / jnp.sqrt(rho2)

    # store accept ratio
    logar = z_pe - 0.5 * preconditioner._dimension * jnp.log(rho2)\
        + 0.5 * rho2 / mu**2 * preconditioner.inv_dot(r, r)\
        + 0.5 * rho2 / gSg * (mu**2 - lam**2) / (lam**2 * mu**2)\
        * jnp.dot(g, r)**2

    # proposal
    z = preconditioner.ravel(z + r)

    # proposed values
    z_pe, z_grad = value_and_grad(potential_fn)(z)
    z = preconditioner.unravel(z)
    g = preconditioner.unravel(z_grad)
    Sg = preconditioner.condition(z_grad)
    gSg = jnp.dot(g, Sg)
    rho2 = jnp.clip(gSg, a_min=1.0)

    # update accept ratio
    logar = logar - z_pe + 0.5 * preconditioner._dimension * jnp.log(rho2)\
        - 0.5 * rho2 / mu**2 * preconditioner.inv_dot(r, r)\
        - 0.5 * rho2 / gSg * (mu**2 - lam**2) / (lam**2 * mu**2)\
        * jnp.dot(g, r)**2

    logar = jnp.where(jnp.isnan(logar), jnp.inf, logar)
    accept_prob = jnp.clip(jnp.exp(logar), a_max=1.0)

    z = preconditioner.ravel(z)
    return accept_prob, z, z_grad, z_pe


class Hop(MCMCKernel):
    """
    Hop kernel for Markov Monte Carlo inference using fixed lam, mu, and preconditoner.

    :param model: Python callable containing Pyro :mod:`~numpyro.primitives`.
        If model is provided, `potential_fn` will be inferred using the model.
    :param potential_fn: Python callable that computes the potential energy
        given input parameters. The input parameters to `potential_fn` can be
        any python collection type, provided that `init_params` argument to
        `init_kernel` has the same type.
    :param float lam: Scaling in gradient direction. Default=5.0.
    :param float mu: Scaling in directions perpendicular to gradient. Default=1.0.
        value is 1.0.
    :param covar_matrix: The covariance matrix of a preconditioner.
    :param callable init_strategy: a per-site initialization function.
        See :ref:`init_strategy` section for available functions.
    """

    def __init__(self, model=None, potential_fn=None,
                 lam=1.0, mu=1, covar_matrix=None,
                 init_strategy=init_to_uniform):
        if not (model is None) ^ (potential_fn is None):
            raise ValueError(
                'Only one of `model` or `potential_fn` must be specified.')
        self._model = model
        self._potential_fn = potential_fn
        self._lam = lam
        self._mu = mu
        self._covar_matrix = covar_matrix
        self._init_strategy = init_strategy
        # Set on first call to init
        self._postprocess_fn = None

    def init(self, rng_key, num_warmup, init_params=None, model_args=(), model_kwargs={}):
        # non-vectorized
        if rng_key.ndim == 1:
            rng_key, rng_key_init_model = random.split(rng_key)
        # vectorized
        else:
            rng_key, rng_key_init_model = jnp.swapaxes(
                vmap(random.split)(rng_key), 0, 1)

        # If supplied with a model, then there is a function to get most of the "stuff"
        if self._model is not None:
            init_params, model_potential_fn, postprocess_fn, model_trace = initialize_model(
                rng_key, self._model, dynamic_args=True, init_strategy=self._init_strategy,
                model_args=model_args, model_kwargs=model_kwargs)
            if any(v['type'] == 'param' for v in model_trace.values()):
                warnings.warn("'param' sites will be treated as constants during inference. To define "
                              "an improper variable, please use a 'sample' site with log probability "
                              "masked out. For example, `sample('x', dist.LogNormal(0, 1).mask(False)` "
                              "means that `x` has improper distribution over the positive domain.")
            # use the keyword arguments for the model to build the potential function
            kwargs = {} if model_kwargs is None else model_kwargs
            self._potential_fn = model_potential_fn(*model_args, **kwargs)
            self._postprocess_fn = postprocess_fn

        if self._potential_fn and init_params is None:
            raise ValueError('Valid value of `init_params` must be provided with'
                             ' `potential_fn`.')

        # init state
        if isinstance(init_params, ParamInfo):
            z, pe, z_grad = init_params
        else:
            z, pe, z_grad = init_params, None, None
            pe, z_grad = value_and_grad(self._potential_fn)(z)

        # init preconditioner
        self._preconditioner = preconditioner(z, self._covar_matrix)
        self._dimension = self._preconditioner._dimension

        init_state = HState(0, z, pe, z_grad, 0, 0.0, rng_key)

        return device_put(init_state)

    def sample(self, curr_state, model_args, model_kwargs):
        """
        Run Hop from the given :data:`~numpyro.infer.hop.HopState` and return
        the resulting :data:`~numpyro.infer.hop.Hp[State`.

        :param HopState hop_state: Represents the current state.
        :param model_args: Arguments provided to the model.
        :param model_kwargs: Keyword arguments provided to the model.
        :return: Next `hop_state` after running Hop.
        """
        # process input
        model_kwargs = {} if model_kwargs is None else model_kwargs
        # unpack current state
        itr, curr_z, curr_pe, curr_grad, num_steps, _, rng_key = curr_state
        # random state splitting
        rng_key, rng_key_rnd, rng_key_tran = random.split(rng_key, 3)

        if curr_grad is None:
            curr_grad = grad(self._potential_fn)(curr_z)

        # do a hop
        accept_prob, prop_z, prop_grad, prop_pe = hoppy(
            self._mu, self._lam, curr_z, curr_grad, curr_pe, self._preconditioner, self._potential_fn, rng_key_rnd)

        transition = random.bernoulli(rng_key_tran, accept_prob)
        prop_state = HState(itr+1, prop_z, prop_pe, prop_grad, num_steps,
                            accept_prob, rng_key)
        next_state = cond(transition, (prop_state),
                          identity, (curr_state), identity)

        itr = itr + 1

        return HState(itr, next_state.z, next_state.potential_energy, next_state.grad,
                      num_steps, accept_prob, rng_key)

    def postprocess_fn(self, args, kwargs):
        if self._postprocess_fn is None:
            return identity
        return self._postprocess_fn(*args, **kwargs)

    @property
    def model(self):
        return self._model

    @property
    def sample_field(self):
        return 'z'
