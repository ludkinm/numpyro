from numpyro.infer.mcmc import MCMCKernel
from numpyro.util import identity, cond


class Alternator(MCMCKernel):
    """
    Alternate between two MCMC kernels
    The state type of the kernels must match exactly
    """

    def __init__(self, kernel0, kernel1):
        """
        param: kernel0: an MCMC kernel
        param: kernel1: an MCMC kernel
        """
        self._kernel0 = kernel0
        self._kernel1 = kernel1

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        """
        Initialize the `MCMCKernel` and return an initial state to begin sampling
        from.

        :param random.PRNGKey rng_key: Random number generator key to initialize
            the kernel.
        :param int num_warmup: Number of warmup steps. This can be useful
            when doing adaptation during warmup.
        :param tuple init_params: Initial parameters to begin sampling. The type must
            be consistent with the input type to `potential_fn`.
        :param model_args: Arguments provided to the model.
        :param model_kwargs: Keyword arguments provided to the model.
        :return: The initial state representing the state of the kernel. This can be
            any class that is registered as a
            `pytree <https://jax.readthedocs.io/en/latest/pytrees.html>`_.
        """
        self._kernel1.init(rng_key, num_warmup, init_params,
                           model_args, model_kwargs)
        return self._kernel0.init(rng_key, num_warmup, init_params, model_args, model_kwargs)

    def sample(self, state, model_args, model_kwargs):
        """
        Given the current `state`, return the next `state` using the given
        transition kernel.

        :param state: A `pytree <https://jax.readthedocs.io/en/latest/pytrees.html>`_
            class representing the state for the kernel. For HMC, this is given
            by :data:`~numpyro.infer.hmc.HMCStkernel0ate`. In general, this could be any
            class that supports `getattr`.
        :param model_args: Arguments provided to the model.
        :param model_kwargs: Keyword arguments provided to the model.
        :return: Next `state`.
        """
        return cond(state.itr % 2 == 0,
                    state,
                    lambda s: self._kernel0.sample(
                        s, model_args, model_kwargs),
                    state,
                    lambda s: self._kernel1.sample(s, model_args, model_kwargs))

    @property
    def sample_field(self):
        return 'z'
