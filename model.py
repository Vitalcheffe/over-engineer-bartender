"""The Blind Bartender — Multi-Modal Sensor Fusion via Kalman Filter.

Three independent sensors measure liquid fill level in a glass:
  - audio pitch (resonant frequency of the air column)
  - weight (load cell under the glass)
  - vision (camera-detected meniscus height)

Each sensor has different noise characteristics. The fusion is performed by
a 1-D Kalman filter: a recursive Bayesian estimator that maintains a prior
mean and variance, and updates both as each measurement arrives.

Key insight: with three independent Gaussian sensors of variances σ_a², σ_w²,
σ_v², the steady-state fused variance is:
    1/σ_fused² = 1/σ_a² + 1/σ_w² + 1/σ_v²

This is the inverse-variance weighted mean, and is mathematically equivalent
to running a Kalman filter to convergence with zero process noise.

Implementation:
    Predict step:
        x_pred = x_est              (constant-volume model)
        P_pred = P_est + Q          (process noise Q ~ 1e-6, negligible)

    Update step (per sensor, sequential):
        K_i     = P_pred / (P_pred + R_i)        # Kalman gain
        x_est   = x_pred + K_i * (z_i - x_pred)  # state update
        P_est   = (1 - K_i) * P_pred            # covariance update

After all three sensors update, the fused estimate has variance
    P_final = 1 / (1/P_a + 1/P_w + 1/P_v)
which converges to the inverse-variance weighted mean.
"""
import numpy as np
import json


# Sensor noise standard deviations (in fill-level units, 0..1).
# These are the calibration values from the synthetic pour model.
SIGMA_AUDIO = 0.15   # σ_a: pitch is noisy because of glass-shape ambiguity
SIGMA_WEIGHT = 0.18   # σ_w: load cell is best, but hand motion couples in
SIGMA_VISION = 0.225  # σ_v: vision fails on foam, condensation, occlusion

# Process noise (very small — fill level is near-constant during the
# instantaneous fusion window).
Q = 1e-6


def kalman_fuse(audio, weight, vision,
                sigma_a=SIGMA_AUDIO,
                sigma_w=SIGMA_WEIGHT,
                sigma_v=SIGMA_VISION,
                x0=0.5, P0=1.0):
    """Run a 1-D Kalman filter over three sensor readings.

    Sequential update order: audio → weight → vision.
    Returns (fused_estimate, fused_variance).

    Math:
        Predict: x_pred = x_est,           P_pred = P_est + Q
        Update:  K_i   = P_pred / (P_pred + R_i)
                 x_est = x_pred + K_i * (z_i - x_pred)
                 P_est = (1 - K_i) * P_pred
    """
    # Sensor list: (reading, variance R_i)
    sensors = [
        (audio,   sigma_a ** 2),
        (weight,  sigma_w ** 2),
        (vision,  sigma_v ** 2),
    ]

    x = float(x0)   # state estimate
    P = float(P0)   # state covariance

    for z, R in sensors:
        # Predict (constant-volume model, so x_pred = x)
        P_pred = P + Q

        # Update
        K = P_pred / (P_pred + R)               # Kalman gain
        x = x + K * (z - x)                    # state update (innovation * gain)
        P = (1.0 - K) * P_pred                 # covariance update

    return x, P


def inverse_variance_mean(audio, weight, vision,
                          sigma_a=SIGMA_AUDIO,
                          sigma_w=SIGMA_WEIGHT,
                          sigma_v=SIGMA_VISION):
    """Closed-form inverse-variance weighted mean.

    For independent Gaussian sensors with variances σ_i², the maximum-likelihood
    fused estimate is the inverse-variance weighted mean:

        x_fused = (Σ z_i / σ_i²) / (Σ 1 / σ_i²)

    This is mathematically equivalent to running the Kalman filter to steady
    state. We compute it here as a sanity check on the Kalman implementation.
    """
    w_a = 1.0 / sigma_a ** 2
    w_w = 1.0 / sigma_w ** 2
    w_v = 1.0 / sigma_v ** 2
    x = (audio * w_a + weight * w_w + vision * w_v) / (w_a + w_w + w_v)
    P = 1.0 / (w_a + w_w + w_v)  # fused variance
    return x, P


def simulate_pour(n_pours=1000, noise_level=0.15, seed=42):
    """Simulate n_pours with known ground truth.

    Compares:
      - audio_only, weight_only, vision_only (single-sensor baselines)
      - arithmetic_mean (the wrong way: equal weights)
      - inverse_variance (closed-form MLE)
      - kalman (recursive Bayesian)

    Returns per-method RMSE and mean error.
    """
    rng = np.random.default_rng(seed)

    # Per-sensor σ — the page claims these. The order matters:
    # weight is best, audio is second, vision is worst.
    sigma_a = noise_level        # 0.15
    sigma_w = noise_level * 1.2   # 0.18
    sigma_v = noise_level * 1.5   # 0.225

    results = {
        'audio_only': [],
        'weight_only': [],
        'vision_only': [],
        'arithmetic_mean': [],     # the WRONG fusion (what we had before)
        'inverse_variance': [],    # closed-form MLE
        'kalman': [],              # recursive filter
    }

    for _ in range(n_pours):
        true_fill = rng.uniform(0, 1)
        audio = true_fill + rng.normal(0, sigma_a)
        weight = true_fill + rng.normal(0, sigma_w)
        vision = true_fill + rng.normal(0, sigma_v)

        # Single-sensor baselines
        results['audio_only'].append(abs(audio - true_fill))
        results['weight_only'].append(abs(weight - true_fill))
        results['vision_only'].append(abs(vision - true_fill))

        # Wrong fusion (equal-weight arithmetic mean — kept for comparison)
        arith = (audio + weight + vision) / 3.0
        results['arithmetic_mean'].append(abs(arith - true_fill))

        # Closed-form inverse-variance weighted mean
        inv_x, _ = inverse_variance_mean(
            audio, weight, vision,
            sigma_a=sigma_a, sigma_w=sigma_w, sigma_v=sigma_v
        )
        results['inverse_variance'].append(abs(inv_x - true_fill))

        # 1-D Kalman filter (sequential update)
        kal_x, _ = kalman_fuse(
            audio, weight, vision,
            sigma_a=sigma_a, sigma_w=sigma_w, sigma_v=sigma_v,
            x0=0.5, P0=1.0
        )
        results['kalman'].append(abs(kal_x - true_fill))

    return {
        k: {
            'rmse': float(np.sqrt(np.mean(np.array(v) ** 2))),
            'mean_error': float(np.mean(v)),
            'std_error': float(np.std(v)),
        }
        for k, v in results.items()
    }


def theoretical_fused_variance(sigma_a=SIGMA_AUDIO,
                               sigma_w=SIGMA_WEIGHT,
                               sigma_v=SIGMA_VISION):
    """Theoretical fused variance for inverse-variance weighting.

    1/σ_fused² = 1/σ_a² + 1/σ_w² + 1/σ_v²
    """
    return 1.0 / (1.0 / sigma_a ** 2 + 1.0 / sigma_w ** 2 + 1.0 / sigma_v ** 2)


if __name__ == '__main__':
    print("The Blind Bartender — Multi-Modal Sensor Fusion via Kalman Filter")
    print("=" * 70)
    print()
    print("Sensor noise (σ):")
    print(f"  audio:  σ = {SIGMA_AUDIO:.3f}  (variance {SIGMA_AUDIO**2:.4f})")
    print(f"  weight: σ = {SIGMA_WEIGHT:.3f}  (variance {SIGMA_WEIGHT**2:.4f})")
    print(f"  vision: σ = {SIGMA_VISION:.3f}  (variance {SIGMA_VISION**2:.4f})")
    print()
    print(f"Theoretical fused variance: {theoretical_fused_variance():.6f}")
    print(f"Theoretical fused σ:        {np.sqrt(theoretical_fused_variance()):.4f}")
    print(f"Best single-sensor σ:        {SIGMA_WEIGHT:.4f} (weight)")
    print(f"Improvement ratio:           {SIGMA_WEIGHT / np.sqrt(theoretical_fused_variance()):.2f}× better than best single sensor")
    print()
    print("Simulation (1000 pours):")

    r = simulate_pour(n_pours=1000)
    for k, v in r.items():
        print(f"  {k:>18}: RMSE = {v['rmse']:.4f}   mean_err = {v['mean_error']:+.4f}   std = {v['std_error']:.4f}")

    print()
    print("Key comparison:")
    print(f"  arithmetic_mean RMSE : {r['arithmetic_mean']['rmse']:.4f}  (WRONG: equal weights)")
    print(f"  inverse_variance RMSE: {r['inverse_variance']['rmse']:.4f}  (closed-form MLE)")
    print(f"  kalman          RMSE : {r['kalman']['rmse']:.4f}  (recursive Bayesian)")
    print()
    improvement = r['weight_only']['rmse'] / r['kalman']['rmse']
    print(f"Kalman improvement over best single sensor: {improvement:.2f}×")
    print()

    with open('data/results.json', 'w') as f:
        json.dump({
            'sensor_noise': {
                'audio_sigma': SIGMA_AUDIO,
                'weight_sigma': SIGMA_WEIGHT,
                'vision_sigma': SIGMA_VISION,
            },
            'theoretical': {
                'fused_variance': theoretical_fused_variance(),
                'fused_sigma': float(np.sqrt(theoretical_fused_variance())),
                'improvement_vs_best_single': float(SIGMA_WEIGHT / np.sqrt(theoretical_fused_variance())),
            },
            'simulation': r,
        }, f, indent=2)
    print("Wrote data/results.json")
