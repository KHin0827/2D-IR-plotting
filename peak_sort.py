
import numpy as np
import matplotlib.pyplot as plt
import csv

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

DataPar = np.loadtxt('2D.par.dat')
DataPer = np.loadtxt('2D.per.dat')

plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# Combine parallel and perpendicular with adjustable weights
Data = DataPar.copy()
Data[:, 3] = 1*DataPar[:, 3] + 0*DataPer[:, 3]   # adjust with the experimentalist's setup

w1Dim = int(np.sqrt(len(Data[:, 0])))
w3Dim = w1Dim

wmin = 1600
wmax = 1700
step = 25

mmax = np.max(np.abs(Data[:, 3]))
Data[:, 3] = Data[:, 3] / mmax
Data[1, 3] = 1
Data[2, 3] = -1

w1 = np.reshape(Data[:, 0], (w1Dim, w3Dim))
w3 = np.reshape(Data[:, 1], (w1Dim, w3Dim))

# ---- peaks ----
mask_peak = (Data[:, 3] > np.percentile(Data[:, 3], 99.3)) & \
            (Data[:, 0] >= wmin) & (Data[:, 0] <= wmax)
table_peak = Data[mask_peak]

# ---- valleys ----
mask_valley = (Data[:, 3] < np.percentile(Data[:, 3], 1)) & \
              (Data[:, 0] >= wmin) & (Data[:, 0] <= wmax)
table_valley = Data[mask_valley]

#ready for quadratic fitting
unique_x, counts = np.unique(table_peak[:, 0], return_counts=True)
valid_x = unique_x[counts >= 3]
table_peak = table_peak[np.isin(table_peak[:, 0], valid_x)]

unique_x, counts = np.unique(table_valley[:, 0], return_counts=True)
valid_x = unique_x[counts >= 3]
table_valley = table_valley[np.isin(table_valley[:, 0], valid_x)]

order_peak = np.lexsort((-table_peak[:, 3], table_peak[:, 0]))
sorted_peak = table_peak[order_peak]

order_valley = np.lexsort((-table_valley[:, 3], table_valley[:, 0]))
sorted_valley = table_valley[order_valley]

PlotMap = np.reshape(Data[:, 3], (w1Dim, w3Dim))

fig = plt.figure()
plt.contourf(w1, w3, PlotMap, 20, cmap=plt.cm.bwr)
plt.xlabel(r'$\omega_1$ [cm$^{-1}$]', fontsize=16)
plt.ylabel(r'$\omega_3$ [cm$^{-1}$]', fontsize=16)
plt.gca().set_aspect('equal')
plt.scatter(sorted_peak[:, 0],   sorted_peak[:, 1],   c='red',   s=1,
            edgecolors='black', label='peaks',   zorder=5)
plt.scatter(sorted_valley[:, 0], sorted_valley[:, 1], c='green', s=1,
            edgecolors='black', label='valleys', zorder=5)
plt.xlim(wmin, wmax)
plt.ylim(wmin, wmax)
plt.plot([wmin, wmax], [wmin, wmax], color='black')
plt.xticks(np.arange(wmin, wmax + 1, step), fontsize=12, rotation=0)
plt.yticks(np.arange(wmin, wmax + 1, step), fontsize=12, rotation=0)
plt.show()
fig.savefig('p+n_contour.eps', bbox_inches='tight', dpi=400)

def fit_quadratic_around_peak_grouped_numpy(Test, col_group=0, mode=1):

    Test = np.asarray(Test, dtype=float)
    groups = np.unique(Test[:, col_group])
    results = []

    for g in groups:
        mask = Test[:, col_group] == g
        cluster = Test[mask]
        x_all = cluster[:, 1]
        y_all = cluster[:, 3]

        if len(x_all) < 3:
            continue

        if mode == 1:
            idx_target = np.argmax(y_all)
        elif mode == 0:
            idx_target = np.argmin(y_all)
        else:
            raise ValueError("mode must be 1 (peak) or 0 (valley)")

        x_target = x_all[idx_target]

        dist = np.abs(x_all - x_target)
        idx_nearest = np.argsort(dist)[:3]
        idx_nearest = idx_nearest[np.argsort(x_all[idx_nearest])]

        x3 = x_all[idx_nearest]
        y3 = y_all[idx_nearest]

        A = np.vstack([x3**2, x3, np.ones_like(x3)]).T
        a, b, c = np.linalg.solve(A, y3)

        if abs(a) > 1e-12:
            x_vertex = -b / (2 * a)
        else:
            x_vertex = np.nan

        results.append([g, x_vertex])

    return np.array(results)

result_peak   = fit_quadratic_around_peak_grouped_numpy(sorted_peak,   col_group=0, mode=1)
result_valley = fit_quadratic_around_peak_grouped_numpy(sorted_valley, col_group=0, mode=0)

print("Peak results:")
print(result_peak)
print("Valley results:")
print(result_valley)


n_clusters = 3

X_peak   = result_peak[:,   0].reshape(-1, 1)
X_valley = result_valley[:, 0].reshape(-1, 1)

kmeans_peak = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels_peak = kmeans_peak.fit_predict(X_peak)

kmeans_valley = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels_valley = kmeans_valley.fit_predict(X_valley)

fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(w1, w3, PlotMap, 20, cmap=plt.cm.bwr)
cbar = fig.colorbar(contour, ax=ax, label='Intensity')

def plot_cluster_regression(data, labels, marker, prefix, ax):
    for i in np.unique(labels):
        cluster_pts = data[labels == i]
        if len(cluster_pts) < 2:
            continue

        X = cluster_pts[:, 0].reshape(-1, 1)
        y = cluster_pts[:, 1]

        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        r2 = model.score(X, y)

        ax.scatter(cluster_pts[:, 0], cluster_pts[:, 1],
                   s=20, marker=marker, edgecolors='black', linewidth=0.8)

        x_line = np.linspace(cluster_pts[:, 0].min(), cluster_pts[:, 0].max(), 50)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, linewidth=2,
                label=f'{prefix} {i+1}: y={slope:.3f}x+{intercept:.3f}, R²={r2:.3f}')

plot_cluster_regression(result_peak,   labels_peak,   'o', 'Peak',   ax)
plot_cluster_regression(result_valley, labels_valley, 's', 'Valley', ax)

ax.set_xlabel(r'$\omega_1$ [cm$^{-1}$]', fontsize=16)
ax.set_ylabel(r'$\omega_3$ [cm$^{-1}$]', fontsize=16)
ax.set_aspect('equal')
ax.set_xlim(wmin, wmax)
ax.set_ylim(wmin, wmax)
ax.plot([wmin, wmax], [wmin, wmax], color='black', linestyle='--', linewidth=1)
ax.set_xticks(np.arange(wmin, wmax + 1, step))
ax.set_yticks(np.arange(wmin, wmax + 1, step))
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
fig.savefig('regression_contour.eps', bbox_inches='tight', dpi=400)

rows = []

def collect_fits(data, labels, name):
    for i in np.unique(labels):
        cluster_pts = data[labels == i]
        if len(cluster_pts) < 2:
            continue

        X = cluster_pts[:, 0].reshape(-1, 1)
        y = cluster_pts[:, 1]

        model = LinearRegression()
        model.fit(X, y)

        rows.append({
            'type'      : name,
            'cluster'   : int(i) + 1,
            'slope'     : model.coef_[0],
            'intercept' : model.intercept_,
            'r2'        : model.score(X, y),
            'x_min'     : cluster_pts[:, 0].min(),
            'x_max'     : cluster_pts[:, 0].max(),
        })

collect_fits(result_peak,   labels_peak,   'ESA')
collect_fits(result_valley, labels_valley, 'GSB')

csv_path = 'cluster_fits.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(
        f,
        fieldnames=['type', 'cluster',
                    'slope', 'intercept', 'r2', 'x_min', 'x_max'],
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({k: (f'{v:.6f}' if isinstance(v, float) else v)
                         for k, v in row.items()})

print(f'Wrote {len(rows)} rows to {csv_path}')
print('Columns: type, cluster, slope, intercept, r2, x_min, x_max')