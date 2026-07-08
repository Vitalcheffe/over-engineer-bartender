import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, json, sys, os
sys.path.insert(0, '.')
NAVY='#001F3F'; MUTED='#6B7A8D'; LABEL='#8FA3B1'; BG='#FFFFFF'; RULE='#D6DBE0'
fig, axes = plt.subplots(2,2, figsize=(16,10), constrained_layout=True)
fig.patch.set_facecolor(BG)
# Simple results plot
for ax in axes.flat:
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ax.spines.values(): s.set_color(RULE); s.set_linewidth(0.5)
# Load results
with open('data/results.json') as f: data = json.load(f)
if isinstance(data, dict) and 'results' in data: data = data['results']
elif isinstance(data, dict) and 'history' in data:
    h = data['history']
    if isinstance(h, list) and isinstance(h[0], dict):
        keys = list(h[0].keys())
        x = range(len(h))
        for i, k in enumerate(keys[:4]):
            ax = axes.flat[i]
            vals = [item.get(k, 0) for item in h]
            ax.plot(x, vals, color=NAVY, linewidth=1.5)
            ax.set_title(k, fontsize=13, color=NAVY, fontweight='bold', pad=12)
            ax.set_xlabel('Step', fontsize=10, color=MUTED)
elif isinstance(data, dict):
    keys = list(data.keys())[:4]
    for i, k in enumerate(keys):
        ax = axes.flat[i]
        v = data[k]
        if isinstance(v, dict):
            sub_keys = list(v.keys())[:2]
            for sk in sub_keys:
                sv = v[sk]
                if isinstance(sv, (int, float)):
                    ax.bar(sk, sv, color=NAVY, alpha=0.7)
            ax.set_title(k, fontsize=13, color=NAVY, fontweight='bold', pad=12)
            ax.tick_params(colors=MUTED, labelsize=9)
        elif isinstance(sv, (int, float)):
            ax.bar(k, v, color=NAVY, alpha=0.7)
            ax.set_title(k, fontsize=13, color=NAVY, fontweight='bold', pad=12)
os.makedirs('docs/viz', exist_ok=True)
plt.savefig('docs/viz/analysis-light.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(); print("Saved: docs/viz/analysis-light.png")
