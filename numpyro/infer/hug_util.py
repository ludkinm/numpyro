from collections import namedtuple
from jax import random, grad, hessian
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp

HState = namedtuple('HState', ['itr', 'z', 'potential_energy', 'grad', 'num_steps',
                               'accept_prob', 'rng_key'])


def flat_grad(f, z, preconditioner):
    return preconditioner.flatten(grad(f)(preconditioner.unflatten(z)))


def flat_hessian(f, z):
    h = hessian(f)(z)
    # h might be a dictionary if z is a site
    # if we unravel, it has the wrong dimensions...
    if type(z) is dict:
        z, _ = ravel_pytree(z)
        h, _ = ravel_pytree(h)
        if z.size != h.size:
            h = h.reshape(z.size, z.size)
    return h


class preconditioner():
    """
    A class for preconditioning
    param: prototype_var: a prototype for shape and mapping to varaibles
    param: covar: a covariance matrix
    """

    def __init__(self, prototype_var, covar=None, covar_inv=None):
        if covar is not None and covar_inv is not None:
            raise ValueError("Can't provide both covar and invcovar")
        flat_var, self.unflatten = ravel_pytree(prototype_var)
        self._dimension = jnp.size(flat_var)

        # neither set
        if covar is None and covar_inv is None:
            assert self._dimension is not None
            self._covar = jnp.ones(self._dimension)
            self._covar_sqrt = self._covar
            self._covar_inv = self._covar
            self._log_det = 0.0

        if covar is not None:
            self._covar = covar
            if self._covar.ndim == 1:
                self._covar_sqrt = jnp.sqrt(self._covar)
                self._covar_inv = jnp.reciprocal(self._covar)
                self._log_det = jnp.sum(jnp.log(self._covar))
            elif self._covar.ndim == 2:
                self._covar_sqrt = jnp.linalg.cholesky(self._covar)
                self._covar_inv = np.linalg.inv(self._covar)
                self._log_det = jnp.log(jnp.linalg.det(self._covar))
            else:
                raise ValueError("covar matrix must be 1 or 2 dimensional.")

        if covar_inv is not None:
            self._covar_inv = covar_inv
            if self._covar_inv.ndim == 1:
                self._covar = 1.0 / self._covar_inv
                self._covar_sqrt = jnp.sqrt(self._covar)
                self._log_det = jnp.sum(jnp.log(self._covar))
            elif self._covar_inv.ndim == 2:
                self._covar = jnp.linalg.inv(self._covar_inv)
                self._covar_sqrt = jnp.linalg.cholesky(self._covar)
                self._log_det = jnp.log(jnp.linalg.det(self._covar))
            else:
                raise ValueError(
                    "covar_inv matrix must be 1 or 2 dimensional.")

    def flatten(self, x):
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
        if self._covar.ndim == 2:
            v = jnp.matmul(self._covar, x)
        elif self._covar.ndim == 1:
            v = jnp.multiply(self._covar, x)
        return v

    def inv_condition(self, x):
        """
        Given a parameter, unravel and muliply by the inverse pre-conditioner matrix
        """
        if self._covar_inv.ndim == 2:
            v = jnp.matmul(self._covar_inv, x)
        elif self._covar_inv.ndim == 1:
            v = jnp.multiply(self._covar_inv, x)
        return v

    def log_prob(self, x):
        """
        Get the log_prob supposing this is a normal distribution
        """
        if self._covar_inv.ndim == 2:
            v = jnp.matmul(self._covar_inv, x)
        elif self._covar_inv.ndim == 1:
            v = jnp.multiply(self._covar_inv, x)
        return -0.5 * jnp.dot(v, x) - 0.5 * self._log_det

    def dot(self, x, y):
        """
        Compute x^\top \Sigma y where \Sigma is the preconditioner denoted by self.
        """
        if self._covar_inv.ndim == 2:
            v = jnp.matmul(self._covar, x)
        elif self._covar_inv.ndim == 1:
            v = jnp.multiply(self._covar, x)
        return jnp.dot(v, y)

    def inv_dot(self, x, y):
        """
        Compute x^\top \Sigma^{-1} y where \Sigma is the preconditioner denoted by self.
        """
        if self._covar_inv.ndim == 2:
            v = jnp.matmul(self._covar_inv, x)
        elif self._covar_inv.ndim == 1:
            v = jnp.multiply(self._covar_inv, x)
        return jnp.dot(v, y)
