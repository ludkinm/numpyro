# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import biject_to, constraints
from numpyro.distributions.util import is_identically_one, sum_rightmost
from numpyro.infer.autoguide import AutoContinuous


class Reparam(ABC):
    """
    Base class for reparameterizers.
    """
    @abstractmethod
    def __call__(self, name, fn, obs):
        """
        :param str name: A sample site name.
        :param ~numpyro.distributions.Distribution fn: A distribution.
        :param numpy.ndarray obs: Observed value or None.
        :return: A pair (``new_fn``, ``value``).
        """
        return fn, obs

    def _unwrap(self, fn):
        """
        Unwrap Independent(...) distributions.
        """
        event_dim = fn.event_dim
        while isinstance(fn, dist.Independent):
            fn = fn.base_dist
        return fn, event_dim

    def _wrap(self, fn, event_dim):
        """
        Wrap in Independent distributions.
        """
        if fn.event_dim < event_dim:
            fn = fn.to_event(event_dim - fn.event_dim)
        assert fn.event_dim == event_dim
        return fn

    def _unexpand(self, fn):
        """
        Unexpand ExpandedDistribution(...) distributions.
        """
        batch_shape = fn.batch_shape
        if isinstance(fn, dist.ExpandedDistribution):
            fn = fn.base_dist
        return fn, batch_shape


class LocScaleReparam(Reparam):
    """
    Generic decentering reparameterizer [1] for latent variables parameterized
    by ``loc`` and ``scale`` (and possibly additional ``shape_params``).

    This reparameterization works only for latent variables, not likelihoods.

    **References:**

    1. *Automatic Reparameterisation of Probabilistic Programs*,
       Maria I. Gorinova, Dave Moore, Matthew D. Hoffman (2019)

    :param float centered: optional centered parameter. If None (default) learn
        a per-site per-element centering parameter in ``[0,1]``. If 0, fully
        decenter the distribution; if 1, preserve the centered distribution
        unchanged.
    :param shape_params: list of additional parameter names to copy unchanged from
        the centered to decentered distribution.
    :type shape_params: tuple or list
    """
    def __init__(self, centered=None, shape_params=()):
        assert centered is None or isinstance(centered, (int, float))
        assert isinstance(shape_params, (tuple, list))
        assert all(isinstance(name, str) for name in shape_params)
        if isinstance(centered, (int, float)):
            assert 0 <= centered and centered <= 1
        self.centered = centered
        self.shape_params = shape_params

    def __call__(self, name, fn, obs):
        assert obs is None, "LocScaleReparam does not support observe statements"
        centered = self.centered
        if is_identically_one(centered):
            return name, fn, obs
        event_shape = fn.event_shape
        fn, event_dim = self._unwrap(fn)
        fn, batch_shape = self._unexpand(fn)

        # Apply a partial decentering transform.
        params = {key: getattr(fn, key) for key in self.shape_params}
        if self.centered is None:
            centered = numpyro.param("{}_centered".format(name),
                                     jnp.full(event_shape, 0.5),
                                     constraint=constraints.unit_interval)
        params["loc"] = fn.loc * centered
        params["scale"] = fn.scale ** centered
        decentered_fn = type(fn)(**params).expand(batch_shape)

        # Draw decentered noise.
        decentered_value = numpyro.sample("{}_decentered".format(name),
                                          self._wrap(decentered_fn, event_dim))

        # Differentiably transform.
        delta = decentered_value - centered * fn.loc
        value = fn.loc + jnp.power(fn.scale, 1 - centered) * delta

        # Simulate a pyro.deterministic() site.
        return None, value


class TransformReparam(Reparam):
    """
    Reparameterizer for
    :class:`~numpyro.distributions.TransformedDistribution` latent variables.

    This is useful for transformed distributions with complex,
    geometry-changing transforms, where the posterior has simple shape in
    the space of ``base_dist``.

    This reparameterization works only for latent variables, not likelihoods.
    """
    def __call__(self, name, fn, obs):
        assert obs is None, "TransformReparam does not support observe statements"
        fn, batch_shape = self._unexpand(fn)
        assert isinstance(fn, dist.TransformedDistribution)

        # Draw noise from the base distribution.
        # We need to make sure that we have the same batch_shape
        reinterpreted_batch_ndims = fn.event_dim - fn.base_dist.event_dim
        x = numpyro.sample("{}_base".format(name),
                           fn.base_dist.to_event(reinterpreted_batch_ndims).expand(batch_shape))

        # Differentiably transform.
        for t in fn.transforms:
            x = t(x)

        # Simulate a pyro.deterministic() site.
        return None, x


class NeuTraReparam(Reparam):
    """
    Neural Transport reparameterizer [1] of multiple latent variables.

    This uses a trained :class:`~pyro.contrib.autoguide.AutoContinuous`
    guide to alter the geometry of a model, typically for use e.g. in MCMC.
    Example usage::

        # Step 1. Train a guide
        guide = AutoIAFNormal(model)
        svi = SVI(model, guide, ...)
        # ...train the guide...

        # Step 2. Use trained guide in NeuTra MCMC
        neutra = NeuTraReparam(guide)
        model = netra.reparam(model)
        nuts = NUTS(model)
        # ...now use the model in HMC or NUTS...

    This reparameterization works only for latent variables, not likelihoods.
    Note that all sites must share a single common :class:`NeuTraReparam`
    instance, and that the model must have static structure.

    [1] Hoffman, M. et al. (2019)
        "NeuTra-lizing Bad Geometry in Hamiltonian Monte Carlo Using Neural Transport"
        https://arxiv.org/abs/1903.03704

    :param ~numpyro.infer.autoguide.AutoContinuous guide: A guide.
    :param params: trained parameters of the guide.
    """
    def __init__(self, guide, params):
        if not isinstance(guide, AutoContinuous):
            raise TypeError("NeuTraReparam expected an AutoContinuous guide, but got {}"
                            .format(type(guide)))
        self.guide = guide
        self.params = params
        try:
            self.transform = self.guide.get_transform(params)
        except (NotImplementedError, TypeError) as e:
            raise ValueError("NeuTraReparam only supports guides that implement "
                             "`get_transform` method that does not depend on the "
                             "model's `*args, **kwargs`") from e
        self._x_unconstrained = {}

    def _reparam_config(self, site):
        if site["name"] in self.guide.prototype_trace and not site.get("is_observed", False):
            return self

    def reparam(self, fn=None):
        return numpyro.handlers.reparam(fn, config=self._reparam_config)

    def __call__(self, name, fn, obs):
        if name not in self.guide.prototype_trace:
            return fn, obs
        assert obs is None, "NeuTraReparam does not support observe statements"

        log_density = 0.
        if not self._x_unconstrained:  # On first sample site.
            # Sample a shared latent.
            z_unconstrained = numpyro.sample("{}_shared_latent".format(self.guide.prefix),
                                             self.guide.get_base_dist().mask(False))

            # Differentiably transform.
            x_unconstrained = self.transform(z_unconstrained)
            # TODO: find a way to only compute those log_prob terms when needed
            log_density = self.transform.log_abs_det_jacobian(z_unconstrained, x_unconstrained)
            self._x_unconstrained = self.guide._unpack_latent(x_unconstrained)

        # Extract a single site's value from the shared latent.
        unconstrained_value = self._x_unconstrained.pop(name)
        transform = biject_to(fn.support)
        value = transform(unconstrained_value)
        logdet = transform.log_abs_det_jacobian(unconstrained_value, value)
        logdet = sum_rightmost(logdet, jnp.ndim(logdet) - jnp.ndim(value) + len(fn.event_shape))
        log_density = log_density + fn.log_prob(value) + logdet
        numpyro.factor("_{}_log_prob".format(name), log_density)
        return None, value

    def transform_sample(self, latent):
        """
        Given latent samples from the warped posterior (with possible batch dimensions),
        return a `dict` of samples from the latent sites in the model.

        :param latent: sample from the warped posterior (possibly batched).
        :return: a `dict` of samples keyed by latent sites in the model.
        :rtype: dict
        """
        x_unconstrained = self.transform(latent)
        return self.guide._unpack_and_constrain(x_unconstrained, self.params)
