import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def taylor_diagram(stddev, corrcoef, refstd, labels, site):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, polar=True)

    angles = np.arccos(corrcoef)
    ax.plot(0, refstd, 'k*', markersize=15, label='Observed')

    for i in range(len(stddev)):
        ax.plot(angles[i], stddev[i], 'o', label=labels[i])

    max_std = max(max(stddev), refstd) * 1.2
    ax.set_rlim(0, max_std)

    ticks = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1]
    ax.set_xticks(np.arccos(ticks))
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_title(f"Taylor Diagram - {site}", fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.05))
    plt.savefig(f"taylor_{site}.png", bbox_inches='tight')
    plt.show()

# Load your data
df = pd.read_excel("DATA FOR TAYLOR PLOT.xlsx")
df['R'] = np.sqrt(df['R2'])

for site in df['Site'].unique():
    subset = df[df['Site'] == site]
    std_obs = subset['STD_OBS'].iloc[0]
    r_values = subset['R'].tolist()
    std_model = [r * std_obs for r in r_values]
    models = subset['Model'].tolist()

    taylor_diagram(std_model, r_values, std_obs, models, site)
