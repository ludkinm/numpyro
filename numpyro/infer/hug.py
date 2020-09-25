from collections import namedtuple
import warnings
import math

from jax import grad, random, vmap, device_put
from jax.tree_util import tree_multimap

from jax.flatten_util import ravel_pytree
import jax.numpy as jnp

from numpyro.distributions.util import cholesky_of_inverse
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import ParamInfo, init_to_uniform, initialize_model
from numpyro.util import cond, identity

HugState = namedtuple('HugState', ['itr', 'z', 'r', 'potential_energy', 'energy', 'num_steps',
                                   'accept_prob', 'mean_accept_prob', 'rng_key'])


class preconditioner():
    """
    A class for preconditioning
    param: prototype_var: a prototype for shape and mapping to varaibles
    param: covar: a covariance matrix
    """

    def __init__(self, prototype_var, covar=None):
        flat_var, self.ravel = ravel_pytree(prototype_var)
        self._covar = covar
        if self._covar is None:
            mass_matrix_size = jnp.size(flat_var)
            assert mass_matrix_size is not None
            self._covar = jnp.ones(mass_matrix_size)
            self._covar_sqrt = self._covar
            self._covar_inv = self._covar
        elif self._covar.ndim == 1:
            self._covar_sqrt = jnp.sqrt(self._covar)
            self._covar_inv = 1.0 / self._covar
        elif self._covar.ndim == 2:
            self._covar_sqrt = jnp.linalg.cholesky(self._covar)
            self._covar_inv = jnp.linalg.inverse(self._covar)
        else:
            raise ValueError("covar matrix must be 1 or 2 dimensional.")

    def unravel(self, x):
        if type(x) is dict:
            x, _ = ravel_pytree(x)
        return x

    def sample(self, rng_key):
        """
         generate momentum
         param: prototype_var: A variable with the required shape for a sample
         param: rng_key: random number key to pass to jax random.
         """
        eps = random.normal(rng_key, jnp.shape(self._covar_sqrt)[:1])
        if self._covar.ndim == 1:
            r = jnp.multiply(self._covar_sqrt, eps)
        elif self._covar.ndim == 2:
            r = jnp.dot(self._covar_sqrt, eps)
        return self.ravel(r)

    def condition(self, x):
        """
        Given a parameter, unravel and muliply by the pre-conditioner matrix
        """
        x = self.unravel(x)
        if self._covar.ndim == 2:
            v = jnp.matmul(self._covar, x)
        elif self._covar.ndim == 1:
            v = jnp.multiply(self._covar, x)
        return self.ravel(v)

    def inv_condition(self, x):
        """
        Given a parameter, unravel and muliply by the inverse pre-conditioner matrix
        """
        x = self.unravel(x)
        if self._covar_inv.ndim == 2:
            v = jnp.matmul(self._covar_inv, x)
        elif self._covar_inv.ndim == 1:
            v = jnp.multiply(self._covar_inv, x)
        return self.ravel(v)

    def log_prob(self, x):
        """
        Get the log_prob supposing this is a normal distribution
        """
        x = self.unravel(x)
        if self._covar_inv.ndim == 2:
            v = jnp.matmul(self._covar_inv, x)
        elif self._covar_inv.ndim == 1:
            v = jnp.multiply(self._covar_inv, x)
        return 0.5 * jnp.dot(v, x)


def hug_integrator(potential_fn, preconditioner, step_size):
    """
    Hug integration helper class, stores all the
    :param potential_fn: Python callable that computes the potential energy
        given input parameters. The input parameters to `potential_fn` can be
        any python collection type.
    :param precondtioner: A precondiitoner
    :param float step_size: Size of a single step.
    """

    def step(z, r, flip=True):
        """
        :param: z, r: position, momenta pair
        :param: flip: should the velocity be flipped? No if the last step to save a grad
        :return: new state for the integrator.
        """
        z = tree_multimap(lambda z, r: z + step_size * r, z, r)
        if flip:
            g = grad(potential_fn)(z)
            Mg = preconditioner.condition(g)
            r = tree_multimap(lambda r, z, g, Mg: r - 2 *
                              jnp.dot(r, g) / jnp.dot(g, Mg) * Mg, r, z, g, Mg)
            return (z, r)
        else:
            return (z, r)

    return step


class Hug(MCMCKernel):
    """
    Hug kernel for Markov Monte Carlo inference using fixed step size,
    trajectory length and and mass matrix adaptation.

    :param model: Python callable containing Pyro :mod:`~numpyro.primitives`.
        If model is provided, `potential_fn` will be inferred using the model.
    :param potential_fn: Python callable that computes the potential energy
        given input parameters. The input parameters to `potential_fn` can be
        any python collection type, provided that `init_params` argument to
        `init_kernel` has the same type.
    :param float step_size: Determines the size of a single step taken by the
        hug integrator while computing the trajectory using Hug
        dynamics. If not specified, it will be set to 1.0.
    :param float trajectory_length: Length of a single Hug trajectory. Default
        value is 1.0.
    :param covar_matrix: The covariance matrix of a preconditioner,
        used when bouncing and to sample momenta.
    :param callable init_strategy: a per-site initialization function.
        See :ref:`init_strategy` section for available functions.
    """

    def __init__(self, model=None, potential_fn=None,
                 step_size=1.0, trajectory_length=1, covar_matrix=None,
                 init_strategy=init_to_uniform):
        if not (model is None) ^ (potential_fn is None):
            raise ValueError(
                'Only one of `model` or `potential_fn` must be specified.')
        self._model = model
        self._potential_fn = potential_fn
        self._step_size = min(step_size, trajectory_length)
        self._trajectory_length = trajectory_length
        self._num_bounces = max(1, math.ceil(trajectory_length/step_size))
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
            z, _, _ = init_params
        else:
            z = init_params

        # init preconditioner
        self._preconditioner = preconditioner(z, self._covar_matrix)

        # init stepper
        self._stepper = hug_integrator(
            self._potential_fn, self._preconditioner, self._step_size)

        # init state function
        def init_fn(init_params, rng_key):
            # split rng
            pe = self._potential_fn(z)
            rng_key_hug, rng_key_momentum = random.split(rng_key, 2)
            r = self._preconditioner.sample(rng_key_momentum)
            energy = pe + self._preconditioner.log_prob(r)
            # init state
            init_state = HugState(0, z, r, pe, energy, 0,
                                  0.0, 0.0, rng_key_hug)
            return device_put(init_state)

        if rng_key.ndim == 1:
            init_state = init_fn(init_params, rng_key)
        else:
            init_state = vmap(init_fn)(init_params, rng_key)
            sample_fn = vmap(self._sample_fn, in_axes=(0, None, None))
            self._sample_fn = sample_fn
        return init_state

    def sample(self, curr_state, model_args, model_kwargs):
        """
        Run Hug from the given :data:`~numpyro.infer.hug.HugState` and return
        the resulting :data:`~numpyro.infer.hug.HugState`.

        :param HugState hug_state: Represents the current state.
        :param model_args: Arguments provided to the model.
        :param model_kwargs: Keyword arguments provided to the model.
        :return: Next `hug_state` after running Hug.
        """
        # process input
        model_kwargs = {} if model_kwargs is None else model_kwargs
        # unpack current state
        itr, z, _, curr_pe, _, num_steps, _, mean_accept_prob, rng_key = curr_state
        # random state splitting
        rng_key, rng_key_mom, rng_key_tran = random.split(rng_key, 3)
        # resample momneta
        r = self._preconditioner.sample(rng_key_mom)
        # store current energy
        curr_energy = curr_pe + self._preconditioner.log_prob(r)

        # do bounces
        for i in range(0, self._num_bounces):
            z, r = self._stepper(z, r)
            z, r = self._stepper(z, r, i == self._num_bounces)

        prop_pe = self._potential_fn(z)
        prop_energy = prop_pe + + self._preconditioner.log_prob(r)

        delta_energy = prop_energy - curr_energy
        delta_energy = jnp.where(
            jnp.isnan(delta_energy), jnp.inf, delta_energy)
        accept_prob = jnp.clip(jnp.exp(-delta_energy), a_max=1.0)
        transition = random.bernoulli(rng_key_tran, accept_prob)

        prop_state = HugState(itr+1, z, r, prop_pe, prop_energy,
                              num_steps, accept_prob, 0.0, rng_key)
        next_state = cond(transition, (prop_state),
                          identity, (curr_state), identity)

        itr = itr + 1
        mean_accept_prob = mean_accept_prob + \
            (accept_prob - mean_accept_prob) / itr

        return HugState(itr, next_state.z, next_state.r, next_state.potential_energy,
                        next_state.energy, num_steps, accept_prob, mean_accept_prob, rng_key)

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
