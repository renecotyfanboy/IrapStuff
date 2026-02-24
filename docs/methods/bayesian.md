---
icon: lucide/chart-scatter
---

# Bayesian Inference

"*Bayesian inference is a method of statistical inference in which Bayes' theorem is used to update the probability for 
a hypothesis as more evidence or information becomes available.*" [Wikipedia](https://en.wikipedia.org/wiki/Bayesian_inference)

## A brieve introduction to Bayesian philosophy

In a Bayesian inference problem, we gather all the a priori  knowledge we have about the problem by defining probability 
distributions for our parameters $\theta$, the so-called **prior distribution**. We then need to build a statistical model, 
which will allow us to estimate how likely a set of parameters would be, conditioned on a given observation $X$, 
the so-called **likelihood** of the problem. From this, by confronting our a priori knowledge with the observations we 
make of the system, we can determine the probability of a parameter set given the data we have. 

The whole process of Bayesian inference can thus be seen as the updating of our a priori knowledge via the additional 
information provided by the observations of the system, driving additional constraints (when the observations are relevant). 

![Bayesian inference in a nutshell](bayesian_inference.png){ width="600"}
/// caption
Bayesian inference in a nutshell : with a parametrized model, *prior* knowledge on their distribution, and some observed,
one can refine this prior knowledge into more stringent constraints in the form of *posterior* distributions. Adapted from the [SBI package](https://sbi.readthedocs.io/en/latest/)
///

## Some maths (very little)

The main goal of a Bayesian inference problem is to compute the posterior distribution of a given parameter $\theta$, 
conditioned on an observation $X$, i.e. the probability $P(\theta \vert X)$. To do so, we rely on the Bayes formula : 

$$ P(\theta \vert X) = \frac{P(X \vert \theta)P(\theta)}{P(X)} $$

*   $P(\theta)$ is the _prior distribution_ of our parameters, which encompasses all the knowledge we have about them. It
can be informative (very confident about the value of a parameter) or not.
*   $P(X \vert \theta)$ is the _likelihood_ of our problem, which translates how likely is an observation given a set of parameters, and carries 
the uncertainties associated to our model and observations.
*   $P(X)$ is called the _evidence_, this quantity is constant for a given problem and simply 
consists in a normalization factor that exists to ensure the integrability of $P(\theta \vert X)$. It is either totally 
useless or the *Graal* of Bayesian inference, depending on you performing parameter estimation or model selection.

!!! note
    The evidence $P(X)$ is the very reason Bayesian inference is hard to perform in the first place, as it hides high-dimension integrals over the whole parameter space.

## Fantastic Samplers ...

A **sampler** is a Markov-Chain/Monte-Carlo (MCMC) method to estimate the posterior distribution of a given problem. It 
launches Markov-Chains that will wander in the parameter space, and after a given time, the samples in the chain will be 
distributed according to the posterior distribution. There is a huge number of samplers used in the state of the art. Most 
of them rely on {proposal + random acceptance} of new states, some of them require computing the gradient of the likelihood 
as a function of parameters. You can find a [nice illustration](https://chi-feng.github.io/mcmc-demo/app.html?algorithm=RandomWalkMH&target=banana) 
of various sampling algorithm, to get a visual idea of how they work. 

In theory, the posterior distribution is independent of the sampler used to approximate it. However, some perform 
better with fancy posterior, some are slower, some are faster. A good sampler ensures that the final samples are lowly 
correlated, and it converges fast to the stationary distribution.  

A brief list of the most used ones : 

* [Metropolis-Hastings](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm) is the most basic sampling 
algorithm you can imagine. At each step, it proposes a new state for the chain. If the new state improves the likelihood, 
then it is accepted. Otherwise, it is rejected or accepted with a probability function of the likelihood difference 
between the two states. This enables the chain to possibly spend time in every possible location of the parameter space, 
while spending most of his time around the maxima of likelihood. 
* [Hamiltonian Monte Carlo](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo) (HMC) is a sampling algorithm where 
the new states are proposed using Hamiltonian dynamic propagation. The new possible states are often "far away" to the 
previous step (when compared to Metropolis-Hastings) which reduce the correlation between successive states while 
keeping a good acceptance rate because of the energy conserving propagation.  
* [No U-Turn Sampler](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo#No_U-Turn_Sampler) (NUTS) is an improvement 
of HMC, which includes an adaptive step which is correlated to the curvature of the posterior distribution around the 
state of the chain. It also contains several fancy features such as a doubling scheme until the achievement of a U-turn 
to ensure quality in the next states. 
* [Nested Sampling Algorithms](https://en.wikipedia.org/wiki/Nested_sampling_algorithm) main goal is to compute the evidence
of a problem, and return the posterior distributions as a byproduct of this computation. It relies on evolutionary strategies 
where the posterior distribution will be learned through redistribution of successive generations of parameters.

!!! danger
    Never code your sampler by yourself : it already exists and your implementation will be **BAD**.

## and where to find them !

There are LOTs of packages for probabilistic programming in Python.
 
*   [PyMC](https://www.pymc.io/welcome.html) which is my favorite, as it is both user-friendly and very sophisticated regarding 
the features it proposes, and implements NUTS-sampler without having to explicitly knowing the gradient of your likelihood 
thanks to autodifferentiation libraries. **This should be your goto package if the problem is easy to write down.**
*  [pyro](https://docs.pyro.ai/en/stable/) which strongly embeds machine learning through `pytorch`, and implements low-level 
variational inference methods.
*  [numpyro](https://num.pyro.ai/en/stable/) which is the one I am using for my PhD thesis, proposes super-fast computation 
and autodifferentiation with `JAX`, and allows for easy integration of `JAX`-written models.

Also, there are samplers for when your likelihood is already written : 

*   [emcee](https://emcee.readthedocs.io/en/stable/) is probably the lightest weight and the most used library for MCMC 
sampling. However, you have to define the prior and likelihood by yourself, which can be pretty tedious. It relies on the 
Affine Invariant Ensemble Sampler (AIES) algorithm to perform the posterior computation
*   [zeus](https://zeus-mcmc.readthedocs.io/en/latest/) is similar as `emcee` in its interface, but uses the 
Ensemble Slice Sampler (ESS) algorithm
*   [multinest](https://dynesty.readthedocs.io/en/latest/) is a library for nested sampling,
*   [dynesty](https://dynesty.readthedocs.io/en/latest/) is a library for dynamic nested sampling, where the number of live points 
is dynamically adjusted over the run.
*   [ultranest](https://johannesbuchner.github.io/UltraNest/index.html) is a meticulous library for nested sampling.
Still a bit tedious as you have to define the prior as transformations of a N-dimensional unitary uniform random variable 
*   [nessai](https://nessai.readthedocs.io/en/latest/) is a machine learning boosted library for nested sampling, using
  normalizing flows to adjust the proposal distribution of the nested sampling algorithm.
*   [nautilus](https://nessai.readthedocs.io/en/latest/) is a machine learning boosted library for nested sampling, using
  MLPs to locally emulate the likelihood function and enhance the proposal.

## Examples with code

Let's do a linear regression on some fake data! First, we generate a bunch of noisy points. 

```py title="Generating the data (in general we don't have access to this step)"
import numpy as np
import matplotlib.pyplot as plt

def generate_fake_data(x, a, b, sigma):
    return a*x + b + np.random.normal(scale=sigma, size=x.shape)

a_ref, b_ref, sigma_ref =  -3, 2, 0.6

x = np.random.uniform(-2, 2, 100)
observed = generate_fake_data(x, a_ref, b_ref, sigma_ref)

plt.scatter(x, observed, color='red');
```

Once this is done, we have some observed points $y_i$. By looking at them, we can see that a linear model with slope $a$ and 
intercept $b$ should be a good fit. In addition, these points are noisy, we can suppose that the noise is Gaussian with 
standard deviation $\sigma$. More formally, our observed data is $Y = (y_1,... y_N)$, and the parameters of our model are 
$\theta = (a,b,\sigma)$.

Under our model, we predict the observed points with $y_i = a x_i + b + \epsilon_i$, 
where $\epsilon_i \sim \mathcal{N}(0,\sigma)$, or $y_i \sim \mathcal{N}(a x_i + b,\sigma)$. The analytical log-likelihood of our model is thus :

$$
\mathcal{L}(a,b,\sigma) = \log P(Y | \theta) = - \sum_{i=1}^N \frac{\left(y_i - (a x_i + b)\right)^2}{2\sigma^2} 
$$

Once we have defined our likelihood, we can define our prior distribution. Let's stick with the following 

* $a \sim \mathcal{U}(-5,0)$
* $b \sim \mathcal{U}(0,5)$
* $\sigma \sim \text{HalfCauchy}(1)$

The two first uniform distributions are quite uninformative, while the HalfCauchy distribution is more adapted to fit 
dispersions as it is strictly positive but favors low values. 

=== "PyMC"

    ```py
    import pymc as pm
    
    # Define the bayesian model
    with pm.Model() as model:
    
        # Model parameters
        a = pm.Uniform("a", -5, 0.)
        b = pm.Uniform("b", 0, 5.)
        sigma = pm.HalfCauchy("sigma", 1)
    
        # Linear model
        mu = a * x + b
    
        # Likelihood
        y = pm.Normal("y", mu, sigma, observed = observed)
    
        # Run the MCMC
        idata = pm.sample(draws=1_000, tune=1_000, cores=4)
    ```

=== "emcee"

    ```py
    import numpy as np
    import emcee
    import arviz as az
    from scipy.stats import uniform, halfcauchy
    
    def log_prior(theta):
        a, b, sigma = theta
        # Note the weird parametrisation from scipy
        prob_a = uniform.logpdf(a, loc=-5, scale=5)
        prob_b = uniform.logpdf(b, loc=0, scale=5)
        prob_sigma = halfcauchy.logpdf(sigma, loc=0, scale=1)
        
        return prob_a + prob_b + prob_sigma
    
    def log_likelihood(theta, x, y):
        a, b, sigma = theta
        mu = a * x + b
    
        return np.sum(-0.5 * np.square((mu - a * x - b) / sigma))
    
    def log_prob(theta, x, y):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, x, observed)
    
    ndim = 3
    nwalkers = 32  # common choice, should be ~ 2*ndim (and usually much larger)
    
    # Initialize walkers within the Uniform priors and a reasonable sigma region
    a0 = uniform.rvs(loc=-5, scale=5, size=nwalkers)
    b0 = uniform.rvs(loc=0, scale=5, size=nwalkers)
    sigma0 = halfcauchy.rvs(loc=0, scale=1, size=nwalkers)
    p0 = np.column_stack([a0, b0, sigma0])
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(x, observed))
    
    # Burn-in (tune)
    tune_steps = 1_000
    state = sampler.run_mcmc(p0, tune_steps, progress=True)
    
    # Reset to drop burn-in samples
    sampler.reset()
    
    # Production (draws)
    draw_steps = 1_000
    sampler.run_mcmc(state, draw_steps, progress=True)
    
    idata = az.from_emcee(sampler, var_names=["a", "b", "sigma"])
    ```

Once the sampling is done, I would recommend to use [`arviz`](https://python.arviz.org/en/stable/index.html) to inspect 
the posterior distributions, the trace of the chains and the posterior predictive checks. For the data visualizations 
and article-ready plots, I suggest [`ChainConsumer`](https://samreay.github.io/ChainConsumer/).

## References

*   [Bayesian Modeling and Computation in Python](https://bayesiancomputationbook.com/welcome.html)
