import math

# Parameters
C = 10      # Clip bound
s = 380        # Sample size
T = 60        # Sequence length
epsilon = 1 # Privacy budget ε
delta = 1/560000  # Privacy parameter δ

# Compute Δ
delta_u = (C / (2*s))

import math

def compute_tau(epsilon, delta, T, delta_u):
    """
    Compute the temperature (tau) for Exponential Mechanism with advanced composition.

    Parameters:
    - epsilon (float): total privacy budget ε
    - delta (float): total probability of failure δ
    - T (int): number of composition steps (e.g., number of tokens)
    - delta_u (float): sensitivity Δu of the utility function

    Returns:
    - tau (float): the temperature scale τ for softmax sampling
    """
    if epsilon <= 0 or delta <= 0 or T <= 0 or delta_u <= 0:
        raise ValueError("All inputs must be positive.")

    sqrt_term = math.sqrt(2 * T * math.log(1 / delta))
    tau = (2 * delta_u * sqrt_term) / epsilon
    return tau

print("Tau=", compute_tau(epsilon, delta, T, delta_u))

