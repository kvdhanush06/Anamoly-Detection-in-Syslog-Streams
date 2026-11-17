import numpy as np
from scipy.stats import genpareto
from sklearn.metrics import precision_recall_fscore_support

def compute_threshold_pot(errors, q=0.90, target_fpr=1e-3):
    """
    Peaks-Over-Threshold (POT) using Extreme Value Theory.
    Fits Generalized Pareto Distribution (GPD) via Method of Moments.
    """
    u = np.quantile(errors, q)
    exceedances = errors[errors > u] - u
    
    if len(exceedances) < 10:
        return np.quantile(errors, 0.995)
        
    mean_ex = np.mean(exceedances)
    var_ex = np.var(exceedances)
    if var_ex == 0:
        return np.quantile(errors, 0.995)
        
    c = 0.5 * (1.0 - (mean_ex**2 / var_ex))
    scale = 0.5 * mean_ex * ((mean_ex**2 / var_ex) + 1.0)
    xi = -c
    
    n_t = len(errors)
    n_u = len(exceedances)
    p_u = n_u / n_t
    
    if xi == 0:
        z_q = u - scale * np.log(target_fpr / p_u)
    else:
        term = (target_fpr / p_u) ** -xi
        z_q = u + (scale / xi) * (term - 1)
        
    return z_q

def compute_threshold_mode_a(val_normal_errors, q=0.90, target_fpr=1e-3, fallback_percentile=99.5):
    """
    Mode A (Label-Free, Headline Threshold):
    Computed ONLY from reconstruction errors of validation NORMAL windows.
    If GPD fit fails or produces NaN/Inf/extreme value outside error bounds,
    falls back automatically to fixed 99.5th percentile of validation normal errors.
    """
    min_err = float(np.min(val_normal_errors))
    max_err = float(np.max(val_normal_errors))
    
    used_fallback = False
    try:
        theta = float(compute_threshold_pot(val_normal_errors, q=q, target_fpr=target_fpr))
        if np.isnan(theta) or np.isinf(theta) or theta <= min_err or theta > max_err * 5.0:
            theta = float(np.percentile(val_normal_errors, fallback_percentile))
            used_fallback = True
    except Exception:
        theta = float(np.percentile(val_normal_errors, fallback_percentile))
        used_fallback = True
        
    return theta, used_fallback

def compute_threshold_mode_b(val_errors, y_val):
    """
    Mode B (Validation-Tuned Threshold):
    Finds threshold that strictly maximizes F1 on validation set (normals + anomalies).
    Uses a grid of 200 candidate quantiles from 80% to 99.9%.
    """
    quantiles = np.linspace(0.80, 0.999, 200)
    candidates = np.quantile(val_errors, quantiles)
    
    best_f1 = -1.0
    best_theta = float(candidates[0])
    
    for theta in candidates:
        y_pred = (val_errors > theta).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_theta = float(theta)
            
    return best_theta, best_f1

