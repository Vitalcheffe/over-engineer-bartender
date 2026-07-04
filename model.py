"""The Blind Bartender — Multi-Modal Sensor Fusion"""
import numpy as np, json
def simulate_pour(n_pours=1000, noise_level=0.15):
    results = {'audio_only': [], 'weight_only': [], 'vision_only': [], 'fused': []}
    for _ in range(n_pours):
        true_fill = np.random.uniform(0, 1)
        audio = true_fill + np.random.normal(0, noise_level)
        weight = true_fill + np.random.normal(0, noise_level * 1.2)
        vision = true_fill + np.random.normal(0, noise_level * 1.5)
        fused = (audio + weight + vision) / 3
        results['audio_only'].append(abs(audio - true_fill))
        results['weight_only'].append(abs(weight - true_fill))
        results['vision_only'].append(abs(vision - true_fill))
        results['fused'].append(abs(fused - true_fill))
    return {k: {'rmse': float(np.sqrt(np.mean(v**2))), 'mean_error': float(np.mean(v))} for k, v in results.items()}
if __name__ == '__main__':
    r = simulate_pour()
    print("The Blind Bartender — Sensor Fusion Results:")
    for k, v in r.items():
        print(f"  {k:>12}: RMSE={v['rmse']:.3f}, mean_err={v['mean_error']:.3f}")
    with open('data/results.json', 'w') as f: json.dump(r, f, indent=2)
