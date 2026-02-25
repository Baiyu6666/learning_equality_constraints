import numpy as np
import matplotlib.pyplot as plt

# "UDF-like" smooth nonnegative function with zero set = unit circle,
# but with degenerate gradient on the zero set (not a regular level-set representation):
# f(x,y) = (x^2 + y^2 - 1)^2
def f(xy):
    x, y = xy
    r2 = x*x + y*y
    return (r2 - 1.0)**2

def grad(xy):
    x, y = xy
    r2 = x*x + y*y
    # ∇f = 4 (r^2-1) [x, y]
    return np.array([4.0*(r2-1.0)*x, 4.0*(r2-1.0)*y])

def hess(xy):
    x, y = xy
    r2 = x*x + y*y
    # H = 8 xx^T + 4(r^2-1) I
    return np.array([[8*x*x + 4*(r2-1.0), 8*x*y],
                     [8*x*y, 8*y*y + 4*(r2-1.0)]])

def newton_minimize(x0, max_iter=25, tol=1e-10):
    """Plain Newton for unconstrained minimization of f (no line search)."""
    x = np.array(x0, dtype=float)
    traj = [x.copy()]
    for _ in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
        H = hess(x)
        # Solve H p = -g (fallback to least-squares if singular)
        try:
            p = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            p = np.linalg.lstsq(H, -g, rcond=None)[0]
        x = x + p
        traj.append(x.copy())
    return np.array(traj)

# Several initial points: some inside the unit circle, some outside.
starts = [
    (0.20, 0.20),  # inside -> gets trapped at origin (spurious stationary point)
    (0.60, 0.20),  # inside but closer to circle -> converges to circle
    (1.80, 0.20),  # outside -> converges to circle
    (0.05, 0.90),  # near circle -> converges to circle
]

trajs = [newton_minimize(s, max_iter=20) for s in starts]

# --- Plot contours and trajectories ---
xs = np.linspace(-2.0, 2.0, 401)
ys = np.linspace(-2.0, 2.0, 401)
X, Y = np.meshgrid(xs, ys)
Z = (X**2 + Y**2 - 1.0)**2

plt.figure(figsize=(7, 7))

# Contours (log-spaced for visibility)
levels = np.logspace(-4, 1, 12)
plt.contour(X, Y, Z + 1e-16, levels=levels)

# Plot true manifold (unit circle)
theta = np.linspace(0, 2*np.pi, 400)
plt.plot(np.cos(theta), np.sin(theta), linewidth=2, label="True zero set: x^2+y^2=1")

# Plot trajectories
for i, (s, tr) in enumerate(zip(starts, trajs), start=1):
    plt.plot(tr[:,0], tr[:,1], marker="o", linewidth=2, label=f"Newton traj {i} start={s}")
    plt.scatter([s[0]], [s[1]], s=80)

# Highlight the origin (spurious stationary point)
plt.scatter([0], [0], s=120, marker="x", linewidths=3, label="Spurious stationary point (origin)")

plt.axis("equal")
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.title("Unconstrained Newton on f=(x^2+y^2-1)^2 (degenerate 'UDF-like' objective)")
plt.legend(loc="upper right", fontsize=8)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# --- Print diagnostics ---
print("Diagnostics (final point, f, ||grad||, cond(H_final))")
for s, tr in zip(starts, trajs):
    xf = tr[-1]
    ff = f(xf)
    gf = np.linalg.norm(grad(xf))
    Hc = np.linalg.cond(hess(xf))
    xf_tuple = (float(xf[0]), float(xf[1]))
    print(f" start={s} -> x_final={tuple(np.round(xf_tuple, 6))}, f={ff:.3e}, ||g||={gf:.3e}, cond(H_final)={Hc:.3e}")