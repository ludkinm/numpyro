from jax.flatten_util import ravel_pytree
from collections import namedtuple
import jax.numpy as jnp
from numpyro.distributions.continuous import MultivariateNormal

HState = namedtuple('HState', ['itr', 'z', 'potential_energy', 'grad', 'num_steps',
                               'accept_prob', 'rng_key'])


def to_accept_prob(log_accept_ratio):
    accept_prob = jnp.clip(jnp.exp(log_accept_ratio), a_max=1.0)
    return jnp.where(jnp.isnan(accept_prob), 0.0, accept_prob)


class Preconditioner():
    """
    A class for preconditioning using a multivariate normal
    param: prototype_var: a prototype for shape and mapping to varaibles
    """

    def __init__(self, prototype_var, covar=None, precision=None):
        def flatten(x):
            if type(x) is dict:
                x, _ = ravel_pytree(x)
            return x
        self.flatten = flatten

        z, self.unflatten = ravel_pytree(prototype_var)
        self._dimension = jnp.size(z)
        if covar is None and precision is None:
            covar = jnp.identity(self._dimension)
        self._dist = MultivariateNormal(
            covariance_matrix=covar, precision_matrix=precision)

    def sample(self, rng_key):
        """
         param: rng_key: random number key to pass to jax random.
         """
        r = self._dist.sample(rng_key)
        lp = self._dist.log_prob(r)
        return self.unflatten(r), lp

    def condition(self, x):
        """
        Multiply by covar
        """
        return self.unflatten(jnp.matmul(self.flatten(x), self._dist.covariance_matrix))

    def inv_condition(self, x):
        """
        Multiply by covar_inv
        """
        return self.unflatten(jnp.matmul(self.flatten(x), self._dist.precision_matrix))

    def log_prob(self, x):
        """
        Get the log_prob supposing this is a normal distribution
        """
        return self._dist.log_prob(self.flatten(x))

    def dot(self, x, y):
        """
        Compute x.t() Sigma y where Sigma is the preconditioner matrix
        """
        return jnp.dot(self.flatten(x), self.flatten(self.condition(y)))

    def inv_dot(self, x, y):
        """
        Compute x.t() Sigma y where Sigma is the preconditioner matrix
        """
        return jnp.dot(self.flatten(x), self.flatten(self.inv_condition(y)))
