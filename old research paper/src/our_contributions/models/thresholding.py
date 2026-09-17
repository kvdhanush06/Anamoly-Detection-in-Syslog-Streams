"""
Adaptive Thresholding Modules for Anomaly Detection
Implements Extreme Value Theory (POT) and simple Statistical cutoffs to
replace arbitrary percentile heuristics.
"""
import numpy as np
from scipy.stats import genpareto, norm

def compute_threshold_statistical(errors, k=3.0):
    """
    Fits a simple Gaussian to the errors and returns mean + k * std.
    This assumes errors roughly follow a normal distribution.
    """
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    return mean_err + k * std_err

def compute_threshold_pot(errors, q=0.90, target_fpr=1e-4):
    """
    Peaks-Over-Threshold (POT) using Extreme Value Theory.
    Fits a Generalized Pareto Distribution (GPD) to the tail of the error distribution.
    
    Args:
        errors (np.ndarray): Reconstruction errors from the fitting/normal dataset.
        q (float): Quantile to define the start of the tail (e.g., 0.90 for top 10%).
        target_fpr (float): The target false positive rate (risk level).
    
    Returns:
        float: The dynamic threshold.
    """
    # 1. Determine the threshold u for the tail
    u = np.quantile(errors, q)
    
    # 2. Extract the excesses over u
    exceedances = errors[errors > u] - u
    
    # If no exceedances or very few, fallback to standard percentile
    if len(exceedances) < 10:
        return np.quantile(errors, 0.95)
        
    # 3. Fit Generalized Pareto Distribution (GPD) using Method of Moments
    mean_ex = np.mean(exceedances)
    var_ex = np.var(exceedances)
    if var_ex == 0:
        return np.quantile(errors, 0.95)
        
    c = 0.5 * (1.0 - (mean_ex**2 / var_ex))
    scale = 0.5 * mean_ex * ((mean_ex**2 / var_ex) + 1.0)
    
    # In scipy, shape parameter is typically negative of standard xi if c > 0
    # Let's use standard Extreme Value Theory parameterization (xi)
    xi = -c
    
    # Probability of exceeding u in the original distribution
    n_t = len(errors)
    n_u = len(exceedances)
    p_u = n_u / n_t
    
    # 4. Compute the Value-at-Risk (VaR) / dynamic threshold at the target_fpr
    # Formula: z_q = u + (scale / xi) * (((target_fpr / p_u) ** -xi) - 1)
    if xi == 0:
        # Limit as xi -> 0 is an exponential distribution
        z_q = u - scale * np.log(target_fpr / p_u)
    else:
        z_q = u + (scale / xi) * (((target_fpr / p_u) ** -xi) - 1)
        
    return z_q
