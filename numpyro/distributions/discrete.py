# Source code modified from scipy.stats._discrete_distns.py
#
# Copyright (c) 2001, 2002 Enthought, Inc.
# All rights reserved.
#
# Copyright (c) 2003-2019 SciPy Developers.
# All rights reserved.
from jax.lax import lgamma

from numpyro.distributions.distribution import jax_discrete
import jax.numpy as np


class binom_gen(jax_discrete):
    def _rvs(self, n, p):
        return self._random_state.binomial(n, p, self._size)

    def _argcheck(self, n, p):
        self.b = n
        return (n >= 0) & (p >= 0) & (p <= 1)

    def _logpmf(self, x, n, p):
        k = np.floor(x)
        combiln = (lgamma(n+1) - (lgamma(k+1) + lgamma(n-k+1)))
        return combiln + np.xlogy(k, p) + np.xlog1py(n-k, -p)

    def _pmf(self, x, n, p):
        # binom.pmf(k) = choose(n, k) * p**k * (1-p)**(n-k)
        return np.exp(self._logpmf(x, n, p))

    def _cdf(self, x, n, p):
        k = np.floor(x)
        vals = special.bdtr(k, n, p)
        return vals

    def _sf(self, x, n, p):
        k = floor(x)
        return special.bdtrc(k, n, p)

    def _ppf(self, q, n, p):
        vals = ceil(special.bdtrik(q, n, p))
        vals1 = np.maximum(vals - 1, 0)
        temp = special.bdtr(vals1, n, p)
        return np.where(temp >= q, vals1, vals)

    def _stats(self, n, p, moments='mv'):
        q = 1.0 - p
        mu = n * p
        var = n * p * q
        g1, g2 = None, None
        if 's' in moments:
            g1 = (q - p) / sqrt(var)
        if 'k' in moments:
            g2 = (1.0 - 6*p*q) / var
        return mu, var, g1, g2

    def _entropy(self, n, p):
        k = np.r_[0:n + 1]
        vals = self._pmf(k, n, p)
        return np.sum(entr(vals), axis=0)
