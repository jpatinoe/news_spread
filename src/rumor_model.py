import numpy as np
from scipy.integrate import odeint

def rumor_model(y, t, beta, gamma):
    """
    Differential equations for the rumor model.

    Parameters:
        y (list): [I, S, R] fractions
        t (float): time
        beta (float): spreading rate
        gamma (float): stifling rate

    Returns:
        list: [dI/dt, dS/dt, dR/dt]
    """
    I, S, R = y
    dIdt = -beta * I * S
    dSdt = beta * I * S - gamma * S * (S + R)
    dRdt = gamma * S * (S + R)
    return [dIdt, dSdt, dRdt]

def simulate_rumor(beta, gamma, I0=0.99, S0=0.01, R0=0.0, days=30):
    """
    Simulate the rumor spreading model over time.

    Parameters:
        beta (float): spreading rate
        gamma (float): stifling rate
        I0 (float): initial ignorant fraction
        S0 (float): initial spreader fraction
        R0 (float): initial stifler fraction
        days (int): number of days to simulate

    Returns:
        tuple: (time array, result array)
    """
    y0 = [I0, S0, R0]
    t = np.linspace(0, days, days * 10)  # 10 points per day for smooth curves
    result = odeint(rumor_model, y0, t, args=(beta, gamma))
    return t, result

