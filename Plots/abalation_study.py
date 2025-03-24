import numpy as np
import matplotlib.pyplot as plt
x_labels = ['Variable', 'Fixed']
y_labels = ['With PT', 'Without PT']
z_values = np.array([[89.08, 84.23], [91.54, 86.92]])

x = np.repeat(np.arange(len(x_labels)), len(y_labels))
y = np.tile(np.arange(len(y_labels)), len(x_labels))
z = z_values.flatten()

fig = plt.figure(figsize=(8, 6), dpi=600)
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(x, y, z, c=z, cmap='cividis',
                     s=100, edgecolor='k')

ax.set_xlabel('Curvature Type', labelpad=10)
ax.set_ylabel('Transport Mode', labelpad=10)
ax.set_zlabel('AUC')
ax.set_xticks(np.arange(len(x_labels)))
ax.set_xticklabels(x_labels)
ax.set_yticks(np.arange(len(y_labels)))
ax.set_yticklabels(y_labels)

ax.tick_params(
    axis='y',
    which='major',
    pad=5
)

cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
cbar.set_label('AUC Score')

plt.savefig('Plots/plot.png')
