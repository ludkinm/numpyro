from jax import random, value_and_grad, vmap, device_put
import jax.numpy as jnp

import numpyro.distributions as dist
from numpyro.infer.hug_util import Preconditioner, HState, to_accept_prob
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import ParamInfo, init_to_uniform, initialize_model
from numpyro.util import cond, identity

# NOTE: potential(s) = -log(posterior(x)) thus there is a sign change on g compared to the paper
# def dot(x, y):
#     return jnp.dot(preconditioner.flatten(x), preconditioner.flatten(y))
# Sample momenta and calculate log_prob from the preconditioner
# w, lpw = preconditioner.sample(rng_key)
# curr_Sg = preconditioner.condition(curr_grad)
# curr_gSg = dot(curr_grad, curr_Sg)
# curr_rho2 = jnp.clip(curr_gSg, a_min=1.0)
# curr_wg = dot(curr_grad, w)
# proposal offset:
# A = (lam - mu) * curr_wg / curr_gSg
# r = tree_multimap(lambda w, Sg: (mu * w + A * Sg) /
#                   jnp.sqrt(curr_rho2), w, curr_Sg)
# proposal
# prop_z = tree_multimap(lambda z, r: z + r, curr_z, r)  # z(n+1)
# prop_pe, prop_grad = value_and_grad(potential_fn)(prop_z)
# prop_Sg = preconditioner.condition(prop_grad)
# prop_gSg = dot(prop_grad, prop_Sg)
# prop_rho2 = jnp.clip(prop_gSg, a_min=1.0)
# prop_wg = dot(prop_grad, w)
# # update accept ratio
# logar = curr_pe - prop_pe + 0.5 * \
#     preconditioner._dimension * jnp.log(prop_rho2/curr_rho2) +\
#     0.5 * (1.0 - prop_rho2/curr_rho2) * preconditioner.inv_dot(r, r) - \
#     0.5 * prop_rho2/curr_rho2 * (lam**2 - mu**2) / mu**2 *\
#     (curr_wg**2/curr_gSg - 1.0/lam**2/prop_gSg * (mu * (prop_wg) +
#                                                   (lam-mu) * (curr_wg) / curr_gSg * dot(curr_grad, prop_Sg))**2)
# logar = jnp.where(jnp.isnan(logar), jnp.inf, logar)
# accept_prob = jnp.clip(jnp.exp(logar), a_max=1.0)
# return accept_prob, prop_z, prop_grad, prop_pe


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
                 lam=1.0, mu=0.5,
                 covar=None, covar_inv=None,
                 init_strategy=init_to_uniform):
        if not (model is None) ^ (potential_fn is None):
            raise ValueError(
                'Only one of `model` or `potential_fn` must be specified.')
        self._model = model
        self._potential_fn = potential_fn
        self._lam = lam
        self._mu = mu
        self._mu2 = mu**2
        self._lam2_minus_mu2 = lam**2 - mu**2
        self._covar = covar
        self._covar_inv = covar_inv
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
        self._preconditioner = Preconditioner(z, self._covar, self._covar_inv)
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
        def proposal_dist(z, g):
            g = -self._preconditioner.flatten(g)
            dim = jnp.size(g)
            rho2 = jnp.clip(jnp.dot(g, g), a_min=1.0)
            covar = (self._mu2 * jnp.eye(dim) + self._lam2_minus_mu2 *
                     jnp.outer(g, g)/jnp.dot(g, g)) / rho2
            return dist.MultivariateNormal(loc=self._preconditioner.flatten(z), covariance_matrix=covar)

        def proposal_density(dist, z):
            return dist.log_prob(self._preconditioner.flatten(z))

        itr, curr_z, curr_pe, curr_grad, num_steps, _, rng_key = curr_state

        rng_key, rng_key_hop, rng_key_ar = random.split(rng_key, 3)

        curr_to_prop = proposal_dist(curr_z, curr_grad)

        prop_z = self._preconditioner.unflatten(
            curr_to_prop.sample(rng_key_hop))

        prop_pe, prop_grad = value_and_grad(self._potential_fn)(prop_z)
        prop_to_curr = proposal_dist(prop_z, prop_grad)

        log_accept_ratio = -prop_pe + curr_pe + \
            proposal_density(prop_to_curr, curr_z) - \
            proposal_density(curr_to_prop, prop_z)
        accept_prob = to_accept_prob(log_accept_ratio)
        transition = random.bernoulli(rng_key_ar, accept_prob)
        next_z, next_pe, next_grad = cond(transition, (prop_z, prop_pe, prop_grad),
                                          identity, (curr_z, curr_pe, curr_grad), identity)

        return HState(itr+1, next_z, next_pe, next_grad, num_steps, accept_prob, rng_key)

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
