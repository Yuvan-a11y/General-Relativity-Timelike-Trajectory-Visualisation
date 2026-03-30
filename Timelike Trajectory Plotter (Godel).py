# -*- coding: utf-8 -*-
"""
Stationary-axisymmetric timelike curve / congruence visualizer
==============================================================

Reduced 3-coordinate sector:
    x^0 = t, x^1 = r, x^2 = varphi

Features
--------
1. User-defined symbolic metric input
2. Automatic Christoffel and Riemann computation
3. Timelike worldline evolution:
       d²x^mu/dλ² + Γ^mu_{ab} dx^a/dλ dx^b/dλ = a^mu
4. Optional Jacobi field visualization
5. Optional translucent cylindrical shell at critical radius
6. Easy-to-edit initial conditions

Notes
-----
- λ is taken as proper time for timelike curves.
- If acceleration = 0, this reduces to an ordinary timelike geodesic.
- The Jacobi-displaced nearby curve is a first-order visualization, not a
  separately integrated second worldline.
- This code is general for a user-supplied 3-coordinate reduced sector.
  Full 4D Kerr requires either a 4-coordinate extension or a chosen
  reduced sector (for example, the equatorial plane).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp
from sympy import Matrix, simplify


# ============================================================================
# 1. SYMBOLIC METRIC MODEL
# ============================================================================

class MetricModel:
    def __init__(self, coords, g_sym, name="Metric"):
        self.coords = coords
        self.g_sym = g_sym
        self.name = name
        self.dim = len(coords)

        self.g_inv_sym = simplify(g_sym.inv())
        self.gamma_sym = self._build_christoffel_sym()
        self.riemann_sym = self._build_riemann_sym()

        self.metric_func = sp.lambdify(coords, g_sym, "numpy")
        self.gamma_func = sp.lambdify(coords, self._flatten_gamma(self.gamma_sym), "numpy")
        self.riemann_func = sp.lambdify(coords, self._flatten_riemann(self.riemann_sym), "numpy")

    def _build_christoffel_sym(self):
        n = self.dim
        Gamma = [[[0] * n for _ in range(n)] for _ in range(n)]

        for mu in range(n):
            for a in range(n):
                for b in range(n):
                    expr = 0
                    for nu in range(n):
                        expr += self.g_inv_sym[mu, nu] * (
                            sp.diff(self.g_sym[nu, b], self.coords[a]) +
                            sp.diff(self.g_sym[nu, a], self.coords[b]) -
                            sp.diff(self.g_sym[a, b], self.coords[nu])
                        ) / 2
                    Gamma[mu][a][b] = simplify(expr)

        return Gamma

    def _build_riemann_sym(self):
        n = self.dim
        Gamma = self.gamma_sym  # reuse already-built Christoffels

        R = [[[[0] * n for _ in range(n)] for _ in range(n)] for _ in range(n)]

        for mu in range(n):
            for nu in range(n):
                for a in range(n):
                    for b in range(n):
                        term = (
                            sp.diff(Gamma[mu][nu][b], self.coords[a]) -
                            sp.diff(Gamma[mu][nu][a], self.coords[b])
                        )
                        for lam in range(n):
                            term += Gamma[mu][a][lam] * Gamma[lam][nu][b]
                            term -= Gamma[mu][b][lam] * Gamma[lam][nu][a]
                        R[mu][nu][a][b] = simplify(term)

        return R

    def _flatten_gamma(self, Gamma):
        n = self.dim
        return [Gamma[mu][a][b] for mu in range(n) for a in range(n) for b in range(n)]

    def _flatten_riemann(self, R):
        n = self.dim
        return [R[mu][nu][a][b] for mu in range(n) for nu in range(n) for a in range(n) for b in range(n)]

    def metric(self, x):
        return np.array(self.metric_func(*x), dtype=float)

    def christoffel(self, x):
        return np.array(self.gamma_func(*x), dtype=float).reshape(self.dim, self.dim, self.dim)

    def riemann(self, x):
        return np.array(self.riemann_func(*x), dtype=float).reshape(self.dim, self.dim, self.dim, self.dim)


# ============================================================================
# 2. METRIC INPUT
# ============================================================================

def build_metric_model(metric_name="godel"):
    t, r, varphi = sp.symbols("t r varphi", real=True)

    if metric_name.lower() == "godel":
        g_sym = Matrix([
            [-1, 0, -sp.sqrt(2) * sp.sinh(r)**2],
            [0, 1, 0],
            [-sp.sqrt(2) * sp.sinh(r)**2, 0, sp.sinh(r)**2 - sp.sinh(r)**4]
        ])
        return MetricModel([t, r, varphi], g_sym, name="Gödel reduced sector")

    raise ValueError(f"Unknown metric_name: {metric_name}")


# ============================================================================
# 3. INITIAL CONDITIONS
# ============================================================================

def solve_ut_from_spatial_proper_velocity(model, x0, ur, uphi):
    """
    Given u^r = dr/dτ and u^varphi = dvarphi/dτ, solve for u^t from
    timelike normalization g_{μν}u^μu^ν = -1.
    """
    g = model.metric(x0)

    a = g[0, 0]
    b = 2 * (g[0, 1] * ur + g[0, 2] * uphi)
    c = g[1, 1] * ur**2 + 2 * g[1, 2] * ur * uphi + g[2, 2] * uphi**2 + 1

    disc = b**2 - 4 * a * c
    if disc < 0:
        # This means the chosen initial spatial velocity is not compatible
        # with a future-directed timelike worldline at this starting point.
        raise ValueError(f"Negative normalization discriminant: {disc}")

    roots = [(-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)]
    candidates = [u for u in roots if u > 0]
    if not candidates:
        raise ValueError("No future-directed timelike root found for u^t.")
    return min(candidates)


def solve_u_from_coordinate_rates(model, x0, dr_dt, dvarphi_dt):
    """
    Given coordinate rates dr/dt and dvarphi/dt, solve for the proper-time
    velocity components u^μ = dx^μ/dτ.
    """
    vt = 1.0
    vr = dr_dt
    vphi = dvarphi_dt

    g = model.metric(x0)
    v = np.array([vt, vr, vphi], dtype=float)

    norm = v @ g @ v
    if norm >= 0:
        raise ValueError("Chosen coordinate rates are not timelike at the initial point.")

    factor = 1.0 / np.sqrt(-norm)
    u = factor * v
    return u


def build_initial_velocity(model, x0, mode="proper", ur=0.2, uphi=2.0, dr_dt=0.0, dvarphi_dt=1.0):
    """
    Two ways to specify initial velocity:

    mode = "proper":
        specify ur = dr/dτ and uphi = dvarphi/dτ

    mode = "coordinate":
        specify dr_dt = dr/dt and dvarphi_dt = dvarphi/dt
    """
    if mode == "proper":
        ut = solve_ut_from_spatial_proper_velocity(model, x0, ur, uphi)
        return np.array([ut, ur, uphi], dtype=float)

    if mode == "coordinate":
        return solve_u_from_coordinate_rates(model, x0, dr_dt, dvarphi_dt)

    raise ValueError("mode must be either 'proper' or 'coordinate'")


def build_initial_jacobi_data(model, x0, u0, epsilon=1e-5, direction="radial", enabled=True):
    """
    Build initial Jacobi field xi^μ and eta^μ.

    If enabled=False, returns xi0 = eta0 = 0 so only a single trajectory is plotted.
    """
    n = model.dim

    if not enabled or epsilon == 0.0:
        return np.zeros(n), np.zeros(n)

    g = model.metric(x0)
    Gamma0 = model.christoffel(x0)

    xi0 = np.zeros(n)

    if direction == "radial":
        xi0[1] = epsilon
    elif direction == "angular":
        xi0[2] = epsilon
    else:
        raise ValueError("direction must be 'radial' or 'angular'")

    u_cov = g @ u0
    if abs(u_cov[0]) < 1e-14:
        raise ValueError("Cannot enforce u·xi = 0 because u_0 has vanishing covariant time component.")

    # Enforce initial orthogonality: u_mu xi^mu = 0
    xi0[0] = -np.dot(u_cov[1:], xi0[1:]) / u_cov[0]

    # Initial parallel transport choice for eta^mu
    eta0 = np.zeros(n)
    for mu in range(n):
        eta0[mu] = -sum(
            Gamma0[mu, a, b] * u0[a] * xi0[b]
            for a in range(n)
            for b in range(n)
        )

    return xi0, eta0


# ============================================================================
# 4. ACCELERATION MODELS
# ============================================================================

def zero_acceleration(lam, x, u):
    return np.zeros_like(u)


def constant_coordinate_acceleration(a_vec):
    a_vec = np.array(a_vec, dtype=float)

    def accel(lam, x, u):
        return a_vec.copy()

    return accel


def orthogonalized_constant_acceleration(spatial_vec):
    """
    spatial_vec = [a^r, a^varphi]
    The time component is solved from u_mu a^mu = 0.
    """
    spatial_vec = np.array(spatial_vec, dtype=float)

    def accel_factory(model):
        def accel(lam, x, u):
            g = model.metric(x)
            u_cov = g @ u
            a = np.zeros_like(u)
            a[1:] = spatial_vec

            if abs(u_cov[0]) > 1e-14:
                a[0] = -np.dot(u_cov[1:], a[1:]) / u_cov[0]
            else:
                a[0] = 0.0
            return a

        return accel

    return accel_factory


# ============================================================================
# 5. ODE SYSTEM
# ============================================================================

def make_worldline_odes(model, acceleration=None):
    n = model.dim
    if acceleration is None:
        acceleration = zero_acceleration

    def odes(lam, Y):
        x = Y[0:n]
        u = Y[n:2*n]
        xi = Y[2*n:3*n]
        eta = Y[3*n:4*n]

        Gamma = model.christoffel(x)
        R = model.riemann(x)

        dx = u

        a_ext = acceleration(lam, x, u)

        # Modified geodesic equation:
        # d²x^μ/dλ² + Γ^μ_{ab} u^a u^b = a^μ
        du = np.zeros(n)
        for mu in range(n):
            du[mu] = -sum(
                Gamma[mu, a, b] * u[a] * u[b]
                for a in range(n)
                for b in range(n)
            ) + a_ext[mu]

        # Geodesic deviation with the Riemann convention built above:
        # D²ξ^μ/dλ² = - R^μ_{ναβ} u^ν ξ^α u^β
        # This is used here only as a first-order visualization aid.
        dxi = eta
        deta = np.zeros(n)
        for mu in range(n):
            deta[mu] = -sum(
                R[mu, nu, a, b] * u[nu] * xi[a] * u[b]
                for nu in range(n)
                for a in range(n)
                for b in range(n)
            )

        return np.concatenate([dx, du, dxi, deta])

    return odes


# ============================================================================
# 6. SOLVER
# ============================================================================

def solve_trajectory_family(
    model,
    x0,
    initial_velocity_list,
    lam_span=(0, 5),
    t_eval=None,
    jacobi_enabled=True,
    jacobi_epsilon=1e-5,
    jacobi_direction="radial",
    acceleration=None
):
    if t_eval is None:
        t_eval = np.linspace(lam_span[0], lam_span[1], 600)

    odes = make_worldline_odes(model, acceleration=acceleration)
    solutions = []

    for u0 in initial_velocity_list:
        xi0, eta0 = build_initial_jacobi_data(
            model=model,
            x0=x0,
            u0=u0,
            epsilon=jacobi_epsilon,
            direction=jacobi_direction,
            enabled=jacobi_enabled
        )

        Y0 = np.concatenate([x0, u0, xi0, eta0])

        sol = solve_ivp(
            odes,
            lam_span,
            Y0,
            t_eval=t_eval,
            rtol=1e-10,
            atol=1e-12,
            method="DOP853"
        )
        solutions.append(sol)

    return solutions


# ============================================================================
# 7. PLOTTING
# ============================================================================

def cylindrical_embedding(r, varphi):
    x = r * np.cos(varphi)
    y = r * np.sin(varphi)
    return x, y


def jacobi_to_cartesian_displacement(r, varphi, xi_r, xi_varphi):
    dx = xi_r * np.cos(varphi) - r * xi_varphi * np.sin(varphi)
    dy = xi_r * np.sin(varphi) + r * xi_varphi * np.cos(varphi)
    return dx, dy


def add_translucent_cylindrical_shell(ax, radius, t_min, t_max, color="red", alpha=0.10, n_theta=120, n_z=40):
    theta = np.linspace(0, 2 * np.pi, n_theta)
    z = np.linspace(t_min, t_max, n_z)
    Theta, Z = np.meshgrid(theta, z)

    X = radius * np.cos(Theta)
    Y = radius * np.sin(Theta)

    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0, shade=False, zorder=1)


def plot_trajectory_family(
    solutions,
    title="Timelike Trajectory Congruence",
    show_jacobi=True,
    jacobi_scale=4.0,
    critical_radius=None,
    critical_label=None,
    show_shell=True
):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    colors = ["black", "tab:blue", "tab:green", "tab:purple", "tab:orange", "tab:red"]
    all_t = []

    for i, sol in enumerate(solutions):
        t = sol.y[0]
        r = sol.y[1]
        varphi = sol.y[2]
        all_t.append(t)

        x, y = cylindrical_embedding(r, varphi)

        if i == 0:
            ax.plot(x, y, t, color="black", linewidth=3, label="Main timelike trajectory")
        else:
            c = colors[i % len(colors)]
            ax.plot(x, y, t, color=c, linewidth=2, alpha=0.95)

        if show_jacobi and i == 0:
            xi_r = sol.y[7]
            xi_varphi = sol.y[8]
            dx_j, dy_j = jacobi_to_cartesian_displacement(r, varphi, xi_r, xi_varphi)
            x_dev = x + jacobi_scale * dx_j
            y_dev = y + jacobi_scale * dy_j
            ax.plot(x_dev, y_dev, t, color="dimgray", linewidth=2, label="Jacobi-displaced nearby curve")

    t_min = min(np.min(t) for t in all_t)
    t_max = max(np.max(t) for t in all_t)

    if critical_radius is not None:
        theta = np.linspace(0, 2 * np.pi, 200)
        x_c = critical_radius * np.cos(theta)
        y_c = critical_radius * np.sin(theta)
        z_c = np.zeros_like(theta)

        label = critical_label if critical_label is not None else f"Critical radius ($r_c={critical_radius:.3f}$)"
        ax.plot(x_c, y_c, z_c, color="red", linestyle="--", linewidth=2, label=label, zorder=100)

        if show_shell:
            add_translucent_cylindrical_shell(ax, critical_radius, t_min, t_max, color="red", alpha=0.10)

    t0 = solutions[0].y[0, 0]
    r0 = solutions[0].y[1, 0]
    varphi0 = solutions[0].y[2, 0]
    x0_plot, y0_plot = cylindrical_embedding(r0, varphi0)

    ax.scatter([x0_plot], [y0_plot], [t0], color="magenta", s=25, marker="o", label="Initial point", zorder=130)
    ax.scatter([0], [0], [0], color="black", s=60, marker="x", label="Coordinate origin", zorder=120)

    ax.view_init(elev=20, azim=45)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("t")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ============================================================================
# 8. USER INPUT SECTION
# ============================================================================

if __name__ == "__main__":
    model = build_metric_model("godel")

    # ------------------------------------------------------------------------
    # CHOOSE INITIAL POSITION
    # ------------------------------------------------------------------------
    t0 = 0.0
    # Keep r0 slightly positive because the (r, varphi) chart is singular at r=0
    # even though Gödel spacetime itself is regular there.
    r0 = 1.0e-3
    varphi0 = 0.0
    x0 = np.array([t0, r0, varphi0], dtype=float)

    # ------------------------------------------------------------------------
    # CHOOSE VELOCITY INPUT MODE
    # ------------------------------------------------------------------------
    # mode = "proper"      -> set ur = dr/dτ, uphi = dvarphi/dτ
    # mode = "coordinate"  -> set dr_dt = dr/dt, dvarphi_dt = dvarphi/dt
    velocity_mode = "proper"

    # Multiple timelike trajectories with different initial velocities
    if velocity_mode == "proper":
        velocity_specs = [
            {"ur": 0.20, "uphi": 1.0},
            {"ur": 0.30, "uphi": 3.0},
            {"ur": 0.40, "uphi": 4.0},
            {"ur": 0.50, "uphi": 5.0},
        ]
        initial_velocity_list = [
            build_initial_velocity(model, x0, mode="proper", ur=spec["ur"], uphi=spec["uphi"])
            for spec in velocity_specs
        ]

    elif velocity_mode == "coordinate":
        velocity_specs = [
            {"dr_dt": 0.05, "dvarphi_dt": 0.80},
            {"dr_dt": 0.08, "dvarphi_dt": 1.00},
            {"dr_dt": 0.10, "dvarphi_dt": 1.20},
            {"dr_dt": 0.12, "dvarphi_dt": 1.40},
        ]
        initial_velocity_list = [
            build_initial_velocity(
                model, x0, mode="coordinate",
                dr_dt=spec["dr_dt"], dvarphi_dt=spec["dvarphi_dt"]
            )
            for spec in velocity_specs
        ]

    else:
        raise ValueError("velocity_mode must be 'proper' or 'coordinate'")

    # ------------------------------------------------------------------------
    # JACOBI FIELD OPTIONS
    # ------------------------------------------------------------------------
    jacobi_enabled = True
    jacobi_epsilon = 1e-5
    jacobi_direction = "radial"   # "radial" or "angular"

    # If you want a single trajectory only:
    # jacobi_enabled = False

    # ------------------------------------------------------------------------
    # ACCELERATION OPTIONS
    # ------------------------------------------------------------------------
    # To recover geodesics:
    # acceleration = zero_acceleration
    #
    # Example forced motion:
    # accel_factory = orthogonalized_constant_acceleration([0.20, 0.05])
    # acceleration = accel_factory(model)

    accel_factory = orthogonalized_constant_acceleration([0.0, 0.0])
    acceleration = accel_factory(model)

    # ------------------------------------------------------------------------
    # SOLVE
    # ------------------------------------------------------------------------
    lam_span = (0, 5)
    t_eval = np.linspace(0, 3, 600)

    solutions = solve_trajectory_family(
        model=model,
        x0=x0,
        initial_velocity_list=initial_velocity_list,
        lam_span=lam_span,
        t_eval=t_eval,
        jacobi_enabled=jacobi_enabled,
        jacobi_epsilon=jacobi_epsilon,
        jacobi_direction=jacobi_direction,
        acceleration=acceleration
    )

    # ------------------------------------------------------------------------
    # PLOT
    # ------------------------------------------------------------------------
    r_crit = np.arcsinh(1.0)

    plot_trajectory_family(
        solutions,
        title="Gödel Spacetime: Timelike Trajectory Congruence",
        show_jacobi=jacobi_enabled,
        jacobi_scale=4.0,
        critical_radius=r_crit,
        critical_label=rf"Critical radius ($r_c = {r_crit:.3f}$)",
        show_shell=True
    )