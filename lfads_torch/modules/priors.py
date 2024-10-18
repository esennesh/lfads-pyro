import pyro
import pyro.nn as pnn
import torch
from torch import nn
from pyro.distributions import Independent, Normal, StudentT, kl_divergence
from torch.distributions.transforms import AffineTransform

class Prior(pnn.PyroModule):
    @property
    def density(self):
        raise NotImplementedError

    # Calculates the KL
    def forward(self, *args):
        raise NotImplementedError

    def make_posterior(self, *args):
        raise NotImplementedError

class Null(Prior):
    @property
    def density(self):
        return None

    def make_posterior(self, *args):
        return None

    def forward(self, *args):
        return 0

class MultivariateNormal(Prior):
    def __init__(
        self,
        mean: float,
        variance: float,
        shape: int,
    ):
        super().__init__()
        # Create distribution parameter tensors
        means = torch.ones(shape) * mean
        logvars = torch.log(torch.ones(shape) * variance)
        self.mean = pnn.PyroParam(means)
        self.register_buffer("logvar", logvars)

    @property
    def density(self):
        # Create the prior
        prior_std = torch.exp(0.5 * self.logvar)
        return Independent(Normal(self.mean, prior_std), 1)

    def make_posterior(self, post_mean, post_std):
        return Independent(Normal(post_mean, post_std), 1)

    def forward(self, post_mean, post_std):
        # Create the posterior distribution
        posterior = self.make_posterior(post_mean, post_std)
        # Compute KL analytically
        kl_batch = kl_divergence(posterior, self.density)
        return torch.mean(kl_batch)

class AutoregressiveMultivariateNormal(Prior):
    def __init__(
        self,
        tau: float,
        nvar: float,
        shape: int,
    ):
        super().__init__()
        # Create the distribution parameters
        logtaus = torch.log(torch.ones(shape) * tau)
        lognvars = torch.log(torch.ones(shape) * nvar)
        self.logtaus = pnn.PyroParam(logtaus)
        self.lognvars = pnn.PyroParam(lognvars)

    @property
    def density(self):
        # Compute alpha and process variance
        alphas = torch.exp(-1.0 / torch.exp(self.logtaus))
        logpvars = self.lognvars - torch.log(1 - alphas**2)
        # Create autocorrelative transformation
        transform = AffineTransform(loc=0, scale=alphas)
        # Align previous samples and compute means and stddevs
        prev_samp = torch.roll(sample, shifts=1, dims=1)
        means = transform(prev_samp)
        stddevs = torch.ones_like(means) * torch.exp(0.5 * self.lognvars)
        # Correct the first time point
        means[:, 0] = 0.0
        stddevs[:, 0] = torch.exp(0.5 * logpvars)
        # Create the prior and compute the log-probability
        return Independent(Normal(means, stddevs), 2)

    def make_posterior(self, post_mean, post_std):
        return Independent(Normal(post_mean, post_std), 2)

    def forward(self, post_mean, post_std):
        posterior = self.make_posterior(post_mean, post_std)
        sample = posterior.rsample()
        log_q = posterior.log_prob(sample)
        log_p = self.density.log_prob(sample)
        kl_batch = log_q - log_p
        return torch.mean(kl_batch)

class MultivariateStudentT(Prior):
    def __init__(
        self,
        loc: float,
        scale: float,
        df: int,
        shape: int,
    ):
        super().__init__()
        # Create the distribution parameters
        loc = torch.ones(shape) * scale
        self.loc = pnn.PyroParam(loc, requires_grad=True)
        logscale = torch.log(torch.ones(shape) * scale)
        self.logscale = pnn.PyroParam(logscale, requires_grad=True)
        self.df = df

    @property
    def density(self):
        # Create the prior distribution
        prior_scale = torch.exp(self.logscale)
        return Independent(StudentT(self.df, self.loc, prior_scale), 1)

    def make_posterior(self, post_loc, post_scale):
        # TODO: Should probably be inferring degrees of freedom along with loc and scale
        return Independent(StudentT(self.df, post_loc, post_scale), 1)

    def forward(self, post_loc, post_scale):
        # Create the posterior distribution
        posterior = self.make_posterior(post_loc, post_scale)
        # Approximate KL divergence
        sample = posterior.rsample()
        log_q = posterior.log_prob(sample)
        log_p = self.density.log_prob(sample)
        kl_batch = log_q - log_p
        return torch.mean(kl_batch)
