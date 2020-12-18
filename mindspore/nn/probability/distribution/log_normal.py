# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""LogNormal Distribution"""
import numpy as np
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
import mindspore.nn.probability.bijector as msb
import mindspore.nn.probability.distribution as msd
from ._utils.utils import check_distribution_name
from ._utils.custom_ops import exp_generic, log_generic

class LogNormal(msd.TransformedDistribution):
    """
    LogNormal distribution.
    A log-normal (or lognormal) distribution is a continuous probability distribution of a random variable whose
    logarithm is normally distributed. It is constructed as the exponential transformation of a Normal distribution.

    Args:
        loc (int, float, list, numpy.ndarray, Tensor): The mean of the underlying Normal distribution.
        scale (int, float, list, numpy.ndarray, Tensor): The standard deviation of the underlying
          Normal distribution.
        seed (int): the seed used in sampling. The global seed is used if it is None. Default: None.
        dtype (mindspore.dtype): type of the distribution. Default: mstype.float32.
        name (str): the name of the distribution. Default: 'LogNormal'.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:
        `scale` must be greater than zero.
        `dist_spec_args` are `loc` and `scale`.
        `dtype` must be a float type because LogNormal distributions are continuous.

    Examples:
        >>> import mindspore
        >>> import mindspore.context as context
        >>> import mindspore.nn as nn
        >>> import mindspore.nn.probability.distribution as msd
        >>> from mindspore import Tensor
        >>> context.set_context(mode=1)
        >>> # To initialize a LogNormal distribution of `loc` 3.0 and `scale` 4.0.
        >>> n1 = msd.LogNormal(3.0, 4.0, dtype=mindspore.float32)
        >>> # A LogNormal distribution can be initialized without arguments.
        >>> # In this case, `loc` and `scale` must be passed in during function calls.
        >>> n2 = msd.LogNormal(dtype=mindspore.float32)
        >>>
        >>> # Here are some tensors used below for testing
        >>> value = Tensor([1.0, 2.0, 3.0], dtype=mindspore.float32)
        >>> loc_a = Tensor([2.0], dtype=mindspore.float32)
        >>> scale_a = Tensor([2.0, 2.0, 2.0], dtype=mindspore.float32)
        >>> loc_b = Tensor([1.0], dtype=mindspore.float32)
        >>> scale_b = Tensor([1.0, 1.5, 2.0], dtype=mindspore.float32)
        >>>
        >>> # Private interfaces of probability functions corresponding to public interfaces, including
        >>> # `prob`, `log_prob`, `cdf`, `log_cdf`, `survival_function`, and `log_survival`, have the same
        >>> # arguments as follows.
        >>> # Args:
        >>> #     value (Tensor): the value to be evaluated.
        >>> #     loc (Tensor): the loc of distribution. Default: None. If `loc` is passed in as None,
        >>> #         the mean of the underlying Normal distribution will be used.
        >>> #     scale (Tensor): the scale of distribution. Default: None. If `scale` is passed in as None,
        >>> #         the standard deviation of the underlying Normal distribution will be used.
        >>> # Examples of `prob`.
        >>> # Similar calls can be made to other probability functions
        >>> # by replacing 'prob' by the name of the function.
        >>> ans = n1.prob(value)
        >>> print(ans)
        [0.07528435 0.04222769 0.02969363]
        >>> # Evaluate with respect to distribution b.
        >>> ans = n1.prob(value, loc_b, scale_b)
        >>> print(ans)
        [0.24197072 0.13022715 0.0664096 ]
        >>> # `loc` and `scale` must be passed in during function calls since they were not passed in construct.
        >>> ans = n2.prob(value, loc_a, scale_a)
        >>> print(ans)
        [0.12098535 0.08056299 0.06006904]
        >>> # Functions `mean`, `sd`, `var`, and `entropy` have the same arguments.
        >>> # Args:
        >>> #     loc (Tensor): the loc of distribution. Default: None. If `loc` is passed in as None,
        >>> #         the mean of the underlying Normal distribution will be used.
        >>> #     scale (Tensor): the scale of distribution. Default: None. If `scale` is passed in as None,
        >>> #         the standard deviation of the underlying Normal distribution will be used.
        >>> # Example of `mean`. `sd`, `var`, and `entropy` are similar.
        >>> ans = n1.mean()
        >>> print(ans)
        59874.14
        >>> ans = n1.mean(loc_b, scale_b)
        >>> print(ans)
        [ 4.481689  8.372897 20.085537]
        >>> # `loc` and `scale` must be passed in during function calls since they were not passed in construct.
        >>> ans = n2.mean(loc_a, scale_a)
        >>> print(ans)
        [54.59815 54.59815 54.59815]
        >>> # Interfaces of 'kl_loss' and 'cross_entropy' are the same:
        >>> # Args:
        >>> #     dist (str): the type of the distributions. Only "Normal" is supported.
        >>> #     loc_b (Tensor): the loc of distribution b.
        >>> #     scale_b (Tensor): the scale distribution b.
        >>> #     loc_a (Tensor): the loc of distribution a. Default: None. If `loc` is passed in as None,
        >>> #         the mean of the underlying Normal distribution will be used.
        >>> #     scale_a (Tensor): the scale distribution a. Default: None. If `scale` is passed in as None,
        >>> #         the standard deviation of the underlying Normal distribution will be used.
        >>> # Examples of `kl_loss`. `cross_entropy` is similar.
        >>> ans = n1.kl_loss('LogNormal', loc_b, scale_b)
        >>> print(ans)
        [8.113706  2.963615  1.3068528]
        >>> ans = n1.kl_loss('LogNormal', loc_b, scale_b, loc_a, scale_a)
        >>> print(ans)
        [1.3068528  0.32342905 0.125     ]
        >>> # Additional `loc` and `scale` must be passed in since they were not passed in construct.
        >>> ans = n2.kl_loss('LogNormal', loc_b, scale_b, loc_a, scale_a)
        >>> print(ans)
        [1.3068528  0.32342905 0.125     ]
        >>> # Examples of `sample`.
        >>> # Args:
        >>> #     shape (tuple): the shape of the sample. Default: ()
        >>> #     loc (Tensor): the loc of the distribution. Default: None. If `loc` is passed in as None,
        >>> #         the mean of the underlying Normal distribution will be used.
        >>> #     scale (Tensor): the scale of the distribution. Default: None. If `scale` is passed in as None,
        >>> #         the standard deviation of the underlying Normal distribution will be used.
        >>> ans = n1.sample()
        >>> print(ans.shape)
        ()
        >>> ans = n1.sample((2,3))
        >>> print(ans.shape)
        (2, 3)
        >>> ans = n1.sample((2,3), loc_b, scale_b)
        >>> print(ans.shape)
        (2, 3, 3)
        >>> ans = n2.sample((2,3), loc_a, scale_a)
        >>> print(ans.shape)
        (2, 3, 3)
    """

    def __init__(self,
                 loc=None,
                 scale=None,
                 seed=0,
                 dtype=mstype.float32,
                 name="LogNormal"):
        """
        Constructor of LogNormal distribution.
        """
        super(LogNormal, self).__init__(distribution=msd.Normal(loc, scale, dtype=dtype),
                                        bijector=msb.Exp(),
                                        seed=seed, name=name)

        # overwrite default_parameters and parameter_names
        self._reset_parameters()
        self._loc = self._add_parameter(loc, 'loc')
        self._scale = self._add_parameter(scale, 'scale')

        self.log_2pi = np.log(2 * np.pi)

        #ops needed for the class
        self.dtypeop = P.DType()
        self.exp = exp_generic
        self.expm1 = P.Expm1()
        self.log = log_generic
        self.const = P.ScalarToArray()
        self.erf = P.Erf()
        self.fill = P.Fill()
        self.greater = P.Greater()
        self.select = P.Select()
        self.shape = P.Shape()
        self.sq = P.Square()
        self.sqrt = P.Sqrt()
        self.cast = P.Cast()
        self.squeeze = P.Squeeze(0)

    @property
    def loc(self):
        """
        Distribution parameter for the pre-transformed mean
        after casting to self.dtype.
        """
        return self._loc

    @property
    def scale(self):
        """
        Distribution parameter for the pre-transformed standard deviation
        after casting to self.dtype.
        """
        return self._scale

    def _get_dist_type(self):
        return "LogNormal"

    def _get_dist_args(self, loc=None, scale=None):
        if loc is not None:
            self.checktensor(loc, 'loc')
        else:
            loc = self.loc
        if scale is not None:
            self.checktensor(scale, 'scale')
        else:
            scale = self.scale
        return loc, scale

    def extend_repr(self):
        if self.is_scalar_batch:
            s = f'loc = {self.loc}, scale = {self.scale}'
        else:
            s = f'batch_shape = {self.broadcast_shape}'
        return s

    def _mean(self, loc=None, scale=None):
        """
        The mean of the distribution.
        """
        mean, sd = self._check_param_type(loc, scale)
        var = self.distribution("var", mean=mean, sd=sd)
        return self.exp(mean + 0.5 * var)

    def _mode(self, loc=None, scale=None):
        """
        The mode of the distribution.
        """
        mean, sd = self._check_param_type(loc, scale)
        var = self.distribution("var", mean=mean, sd=sd)
        return self.exp(mean - var)

    def _var(self, loc=None, scale=None):
        """
        The varience of the distribution.
        """
        mean, sd = self._check_param_type(loc, scale)
        var = self.distribution("var", mean=mean, sd=sd)
        return self.expm1(var) * self.exp(2. * mean + var)

    def _entropy(self, loc=None, scale=None):
        r"""
        Evaluate entropy.

        .. math::
            H(X) = μ + 0.5 + \log(σ) + 0.5 * \log(2pi)
        """
        mean, sd = self._check_param_type(loc, scale)
        return mean + 0.5 + self.log(sd) + 0.5 * self.log_2pi

    def _cdf(self, value, loc=None, scale=None):
        r"""
        Compute the cdf via the below formula,
        where g is the exp bijector,
        and P is the cdf of the underlying normal dist
        .. math::
            Y = g(X)
            P(Y <= a) = P(X <= g^{-1}(a))
        """
        mean, sd = self._check_param_type(loc, scale)
        inverse_value = self.bijector("inverse", value)
        cdf = self.distribution("cdf", inverse_value, mean, sd)

        # to increase numerical stability, set cdf = 0 when value <= 0
        zeros = self.fill(self.dtypeop(cdf), self.shape(cdf), 0.0)

        return self.select(self.greater(value, 0.), cdf, zeros)

    def _log_prob(self, value, loc=None, scale=None):
        r"""
        Compute the log prob via the below formula,
        where g is the exp bijector,
        and P is the pdf of the underlying normal dist
        .. math::
            Y = g(X)
            Py(a) = Px(g^{-1}(a)) * (g^{-1})'(a)
            \log(Py(a)) = \log(Px(g^{-1}(a))) + \log((g^{-1})'(a))
        """
        mean, sd = self._check_param_type(loc, scale)
        inverse_value = self.bijector("inverse", value)
        unadjust_prob = self.distribution("log_prob", inverse_value, mean, sd)
        log_jacobian = self.bijector("inverse_log_jacobian", value)
        return unadjust_prob + log_jacobian

    def _cross_entropy(self, dist, loc_b, scale_b, loc_a=None, scale_a=None):
        r"""
        Evaluate cross entropy between lognormal distributions.

        Args:
            dist (str): The type of the distributions. Should be "LogNormal" in this case.
            loc_b (Tensor): The loc of distribution b.
            scale_b (Tensor): The scale of distribution b.
            loc_a (Tensor): The loc of distribution a. Default: None.
            scale_a (Tensor): The scale of distribution a. Default: None.
        """
        check_distribution_name(dist, 'LogNormal')
        return self._entropy(loc_a, scale_a) + self._kl_loss(dist, loc_b, scale_b, loc_a, scale_a)

    def _kl_loss(self, dist, loc_b, scale_b, loc_a=None, scale_a=None):
        r"""
        Evaluate LogNormal-LogNormal kl divergence, i.e. KL(a||b).

        Args:
            dist (str): The type of the distributions. Should be "LogNormal" in this case.
            loc_b (Tensor): The loc of distribution b.
            scale_b (Tensor): The scale of distribution b.
            loc_a (Tensor): The loc of distribution a. Default: None.
            scale_a (Tensor): The scale of distribution a. Default: None.

        .. math::
            KL(a||b) = 0.5 * (\fract{MEAN(a)}{STD(b)} - \fract{MEAN(b)}{STD(b)}) ^ 2 +
                       0.5 * EXPM1(2 * (\log(STD(a)) - \log(STD(b))) - (\log(STD(a)) - \log(STD(b)))
        """
        check_distribution_name(dist, 'LogNormal')
        return self.distribution("kl_loss", 'Normal', loc_b, scale_b, loc_a, scale_a)

    def _sample(self, shape=(), loc=None, scale=None):
        r"""
        Generate samples via mapping the samples from the underlying normal dist.
        """
        shape = self.checktuple(shape, 'shape')
        mean, sd = self._check_param_type(loc, scale)
        if shape == ():
            sample_shape = (1,)
        else:
            sample_shape = shape
        org_sample = self.distribution("sample", sample_shape, mean, sd)
        org_sample = self.cast(org_sample, self.dtype)
        value = self.bijector("forward", org_sample)
        if shape == ():
            value = self.squeeze(value)
        return value
