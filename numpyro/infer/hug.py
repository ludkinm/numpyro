import math

from jax import random, vmap, device_put, value_and_grad
import jax.numpy as jnp

from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import ParamInfo, init_to_uniform, initialize_model
from numpyro.util import cond, identity
from numpyro.infer.hug_util import flat_grad, preconditioner, HState, flat_hessian


def huggy(nbounce, step_size, z, z_pe, preconditioner, potential_fn, rng_key):
    # Sample momenta from the preconditioner
    r = preconditioner.sample(rng_key)

    # unravel z and g w from tree to vector
    z = preconditioner.flatten(z)

    # store log accept ratio
    logar = z_pe - preconditioner.log_prob(r)

    # do bounces
    for i in range(0, nbounce):
        z = z + step_size * r
        g = flat_grad(potential_fn, z, preconditioner)
        Sg = preconditioner.condition(g)
        r = r - 2 * jnp.dot(r, g) / jnp.dot(g, Sg) * Sg
        z = z + step_size * r

    # update log accept ratio
    z = preconditioner.unflatten(z)
    z_pe, g = value_and_grad(potential_fn)(z)

    logar = logar - z_pe + preconditioner.log_prob(r)
    logar = jnp.where(jnp.isnan(logar), jnp.inf, logar)
    accept_prob = jnp.clip(jnp.exp(logar), a_max=1.0)

    return accept_prob, z, z_pe, g


def huggy_hess(nbounce, step_size, z, z_pe, potential_fn, rng_key):
    conditioner = preconditioner(z, covar_inv=-flat_hessian(potential_fn, z))
    # Sample momenta from the preconditioner
    r = conditioner.sample(rng_key)
    print(f"{r=}")
    # unravel z and g w from tree to vector
    z = conditioner.flatten(z)
    print(f"{z=}")
    # store log accept ratio
    logar = z_pe - conditioner.log_prob(r)
    print(f"{logar=}")
    # do bounces
    for i in range(0, nbounce):
        print(f"{i=}")
        z = z + step_size * r
        print(f"{z=}")
        conditioner = preconditioner(
            z, covar_inv=-flat_hessian(potential_fn, conditioner.unflatten(z)))
        g = flat_grad(potential_fn, z, conditioner)
        print(f"{g=}")
        Sg = conditioner.condition(g)
        r = r - 2 * jnp.dot(r, g) / jnp.dot(g, Sg) * Sg
        z = z + step_size * r

    # update log accept ratio
    z = conditioner.unflatten(z)
    conditioner = preconditioner(z, covar_inv=-flat_hessian(potential_fn, z))
    z_pe, g = value_and_grad(potential_fn)(z)

    logar = logar - z_pe + conditioner.log_prob(r)
    logar = jnp.where(jnp.isnan(logar), jnp.inf, logar)
    accept_prob = jnp.clip(jnp.exp(logar), a_max=1.0)

    return accept_prob, z, z_pe, g


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
            # use the keyword arguments for the model to build the potential function
            kwargs = {} if model_kwargs is None else model_kwargs
            self._potential_fn = model_potential_fn(*model_args, **kwargs)
            self._postprocess_fn = postprocess_fn

        if self._potential_fn and init_params is None:
            raise ValueError('Valid value of `init_params` must be provided with'
                             ' `potential_fn`.')

        # init state
        print("initializing init_params for hug")
        if isinstance(init_params, ParamInfo):
            z, pe, z_grad = init_params
        else:
            z, pe, z_grad = init_params, None, None
            pe, z_grad = value_and_grad(self._potential_fn)(z)

        # init preconditioner
        print("initializing preconditioner for hug")
        self._preconditioner = preconditioner(z, covar=self._covar_matrix)

        # init state function
        def init_fn(init_params, rng_key):
            # init state
            init_state = HState(0, z, pe, z_grad, 0, 0.0, rng_key)
            return device_put(init_state)

        if rng_key.ndim == 1:
            init_state = init_fn(init_params, rng_key)
        else:
            init_state = vmap(init_fn)(init_params, rng_key)
            sample_fn = vmap(self._sample_fn, in_axes=(0, None, None))
            self._sample_fn = sample_fn

        print("finished init for hug")
        return init_state

    def sample(self, curr_state, model_args, model_kwargs):
        """
        Run Hug from the given :data:`~numpyro.infer.hug.HugState` and return
        the resulting :data:`~numpyro.infer.hug.HugState`.

        :param HugState curr_state: Represents the current state.
        :param model_args: Arguments provided to the model.
        :param model_kwargs: Keyword arguments provided to the model.
        :return: Next `hug_state` after running Hug.
        """
        # process input
        model_kwargs = {} if model_kwargs is None else model_kwargs
        # unpack current state
        itr, curr_z, curr_pe, _, num_steps, _, rng_key = curr_state
        # random state splitting
        rng_key, rng_key_hug, rng_key_tran = random.split(rng_key, 3)

        # do hug
        accept_prob, prop_z, prop_pe, prop_grad = huggy(self._num_bounces, self._step_size, curr_z, curr_pe,
                                                        self._preconditioner, self._potential_fn, rng_key_hug)

        transition = random.bernoulli(rng_key_tran, accept_prob)

        prop_state = HState(itr+1, prop_z, prop_pe, prop_grad,
                            num_steps, accept_prob, rng_key)
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


class HugHess(MCMCKernel):
    """
    Hug kernel for Markov Monte Carlo inference using fixed step size,
    trajectory length and and hessian based conditioner

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
    :param callable init_strategy: a per-site initialization function.
        See :ref:`init_strategy` section for available functions.
    """

    def __init__(self, model=None, potential_fn=None,
                 step_size=1.0, trajectory_length=1,
                 init_strategy=init_to_uniform):
        if not (model is None) ^ (potential_fn is None):
            raise ValueError(
                'Only one of `model` or `potential_fn` must be specified.')
        self._model = model
        self._potential_fn = potential_fn
        self._step_size = min(step_size, trajectory_length)
        self._trajectory_length = trajectory_length
        self._num_bounces = max(1, math.ceil(trajectory_length/step_size))
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
        print("initializing init_params for hug")
        if isinstance(init_params, ParamInfo):
            z, pe, z_grad = init_params
        else:
            z, pe, z_grad = init_params, None, None
            pe, z_grad = value_and_grad(self._potential_fn)(z)

        # init state function
        def init_fn(init_params, rng_key):
            # init state
            init_state = HState(0, z, pe, z_grad, 0, 0.0, rng_key)
            return device_put(init_state)

        if rng_key.ndim == 1:
            init_state = init_fn(init_params, rng_key)
        else:
            init_state = vmap(init_fn)(init_params, rng_key)
            sample_fn = vmap(self._sample_fn, in_axes=(0, None, None))
            self._sample_fn = sample_fn

        print("finished init for hug")
        return init_state

    def sample(self, curr_state, model_args, model_kwargs):
        """
        Run Hug from the given :data:`~numpyro.infer.hug.HugState` and return
        the resulting :data:`~numpyro.infer.hug.HugState`.

        :param HugState curr_state: Represents the current state.
        :param model_args: Arguments provided to the model.
        :param model_kwargs: Keyword arguments provided to the model.
        :return: Next `hug_state` after running Hug.
        """
        # process input
        model_kwargs = {} if model_kwargs is None else model_kwargs
        # unpack current state
        itr, curr_z, curr_pe, _, num_steps, _, rng_key = curr_state
        # random state splitting
        rng_key, rng_key_hug, rng_key_tran = random.split(rng_key, 3)

        # do hug
        accept_prob, prop_z, prop_pe, prop_grad = huggy_hess(self._num_bounces, self._step_size, curr_z, curr_pe,
                                                             self._potential_fn, rng_key_hug)

        transition = random.bernoulli(rng_key_tran, accept_prob)

        prop_state = HState(itr+1, prop_z, prop_pe, prop_grad,
                            num_steps, accept_prob, rng_key)
        next_state = cond(transition, (prop_state),
                          identity, (curr_state), identity)
        itr = itr + 1

        return HState(itr, next_state.z, next_state.potential_energy, next_state.grad,
                      num_steps, accept_prob, rng_key)

    def postprocess_fn(self, args, kwargs):
        if self._postprocess_fn is None:
            return identity
        return self._postprocess_fn(*args, **kwargs)

    @ property
    def model(self):
        return self._model

    @ property
    def sample_field(self):
        return 'z'
