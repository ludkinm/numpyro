from collections import namedtuple

from jax import grad, random, vmap, device_put, value_and_grad, jacfwd, jacrev
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp

HState = namedtuple('HState', ['itr', 'z', 'potential_energy', 'grad', 'num_steps',
                               'accept_prob', 'rng_key'])


def hessian(f):
    return jacfwd(jacrev(f))


class raveller():
    def __init__(self, prototype_var):
        flat_var, self._ravel = ravel_pytree(prototype_var)
        self._dimension = jnp.size(flat_var)

    def unravel(self, x):
        if type(x) is dict:
            x, _ = ravel_pytree(x)
        return x

    def ravel(self, x):
        return self._ravel(x)

    @property
    def dimension(self):
        return self._dimension


class preconditioner():
    """
    A class for preconditioning
    param: prototype_var: a prototype for shape and mapping to varaibles
    param: covar: a covariance matrix
    """

    def __init__(self, prototype_var, covar=None):
        flat_var, self.ravel = ravel_pytree(prototype_var)
        self._dimension = jnp.size(flat_var)
        self._covar = covar
        if self._covar is None:
            assert self._dimension is not None
            self._covar = jnp.ones(self._dimension)
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
        return r

    def condition(self, x):
        """
        Given a parameter, unravel, muliply by the pre-conditioner matrix, re-ravel
        """
        x = self.unravel(x)
        if self._covar.ndim == 2:
            v = jnp.matmul(self._covar, x)
        elif self._covar.ndim == 1:
            v = jnp.multiply(self._covar, x)
        return v

    def inv_condition(self, x):
        """
        Given a parameter, unravel and muliply by the inverse pre-conditioner matrix
        """
        x = self.unravel(x)
        if self._covar_inv.ndim == 2:
            v = jnp.matmul(self._covar_inv, x)
        elif self._covar_inv.ndim == 1:
            v = jnp.multiply(self._covar_inv, x)
        return v

    def log_prob(self, x):
        """
        Get the log_prob supposing this is a normal distribution
        """
        x = self.unravel(x)
        if self._covar_inv.ndim == 2:
            v = jnp.matmul(self._covar_inv, x)
        elif self._covar_inv.ndim == 1:
            v = jnp.multiply(self._covar_inv, x)
        return -0.5 * jnp.dot(v, x)

    def dot(self, x, y):
        """
        Compute x^\top \Sigma y where \Sigma is the preconditioner denoted by self.
        """
        x = self.unravel(x)
        y = self.unravel(y)
        if self._covar_inv.ndim == 2:
            v = jnp.matmul(self._covar, x)
        elif self._covar_inv.ndim == 1:
            v = jnp.multiply(self._covar, x)
        return jnp.dot(v, y)

    def inv_dot(self, x, y):
        """
        Compute x^\top \Sigma^{-1} y where \Sigma is the preconditioner denoted by self.
        """
        x = self.unravel(x)
        y = self.unravel(y)
        if self._covar_inv.ndim == 2:
            v = jnp.matmul(self._covar_inv, x)
        elif self._covar_inv.ndim == 1:
            v = jnp.multiply(self._covar_inv, x)
        return jnp.dot(v, y)
