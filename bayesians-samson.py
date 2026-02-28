import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt

# Define likelihood function
def likelihood_func(datum, mu):
    likelihood_out = sts.norm.pdf(datum, mu, scale=0.1)  
    return likelihood_out / likelihood_out.sum()

# Values of mu
mu = np.linspace(1.05, 1.8, num=50)

# Compute likelihood for datum = 1.7
likelihood_out = likelihood_func(1.7, mu)

# Plot the likelihood
plt.plot(mu, likelihood_out)
plt.title("Likelihood of 1.7m given observation 1.7m")
plt.ylabel("Probability Density / Likelihood")
plt.xlabel("Value of sinus")
plt.show()

# Uniform distribution (shifted up by +1 for clarity)
uniform_dist = sts.uniform.pdf(mu) + 1
uniform_dist = uniform_dist / uniform_dist.sum()  # Normalize so densities sum to 1

# Beta distribution with alpha=2.5, beta=2
beta_dist = sts.beta.pdf(mu, 2.5, 2, loc=1.65, scale=0.2)
beta_dist = beta_dist / beta_dist.sum()

# Plot both distributions
plt.plot(mu, beta_dist, label='Beta Dist')
plt.plot(mu, uniform_dist, label='Uniform Dist')
plt.xlabel('Value of Simu in meters')
plt.ylabel('Probability density')
plt.legend()
plt.show()

# Compute likelihood for datum = 1.9
likelihood_out = likelihood_func(1.9, mu)

# Unnormalized posterior
unnormalized_posterior = likelihood_out * uniform_dist

# Plot Bayesian components
plt.plot(mu, uniform_dist, label="Uniform Prior")
plt.plot(mu, likelihood_out, label="Likelihood")
plt.plot(mu, unnormalized_posterior, label="Unnormalized Posterior")
plt.xlabel(r"$\mu$ in meters")
plt.ylabel("Probability Density")
plt.title("Bayesian Components")
plt.legend()
plt.show()