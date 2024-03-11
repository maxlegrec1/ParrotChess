import chess
import chess.pgn
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, lognorm, expon, gamma, weibull_min
def uniform_density(elo,elo_min = 2500, elo_max = 3000):
    if elo >=2500:
        return 1/(elo_max - elo_min)
    else:
        return 0



f = open("pros.pgn")
samples = []


#use the john von neumann algorithm to sample uniformly from the distribution
for i in tqdm(range(10000)):
    pgn = chess.pgn.read_game(f)
    elo = int(pgn.headers["WhiteElo"])

    samples.append(elo)
sample = np.array(samples)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, expon, gamma, weibull_min, skewnorm,exponweib
from scipy.stats import kstest
import warnings



# Define a list of distributions to test
distributions = [norm, lognorm, expon, gamma, weibull_min, skewnorm, exponweib]

# Ignore warnings that may occur during fitting
warnings.filterwarnings('ignore')

# Iterate over the distributions and fit them to the data
best_fit = None
best_params = None
best_ks = np.inf
for distribution in distributions:
    # Fit the distribution to the data
    params = distribution.fit(sample)
    
    # Calculate the Kolmogorov-Smirnov statistic
    ks, p_value = kstest(sample, distribution.cdf, args=params)
    
    # If this is the best fit so far, update the best fit
    if ks < best_ks:
        best_fit = distribution
        best_params = params
        best_ks = ks

# Print the best fit and its parameters
print("Best fit: ", best_fit.name)
print("Parameters: ", best_params)
print(np.mean(sample))
# Plot the data and the best fit distribution
#plt.hist(sample, bins=150, density=True, alpha=0.5, label='Data')
#x = np.linspace(np.min(sample), np.max(sample), 1000)
#plt.plot(x, best_fit.pdf(x, *best_params), 'r-', label='Best fit')
#plt.legend()
#plt.show()
#Best fit:  weibull_min
#Parameters:  (1.0418159053857048, 2499.9497300298544, 139.5627884452257)

samples = []
#use the john von neumann algorithm to sample uniformly from the distribution
for i in tqdm(range(10000)):
    pgn = chess.pgn.read_game(f)
    elo = int(pgn.headers["WhiteElo"])
    #if random.random() < uniform_density(elo)/(4*best_fit.pdf(elo,*best_params)):
    if random.random() < 1:
        samples.append(elo)
sample = np.array(samples)
print(len(samples))
plt.hist(sample, bins=150, density=True, alpha=0.5, label='Data')
plt.show()