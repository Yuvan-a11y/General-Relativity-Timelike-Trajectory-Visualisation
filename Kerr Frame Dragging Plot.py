# -*- coding: utf-8 -*-
"""
Kerr frame-dragging visualizer (equatorial sector)
==================================================

Purpose
-------
This script is designed specifically for a short Kerr section in a paper.
It visualizes timelike trajectories in the equatorial Kerr geometry and
shows frame dragging in a clean and reproducible way.

Coordinates
-----------
x^0 = t, x^1 = r, x^2 = phi

Main idea
---------
The metric contains a nonzero g_{tphi} term. Even when trajectories are
started with small or zero angular coordinate velocity, the geometry can
still twist their motion. This script is intended to display that effect.

What the plot shows
-------------------
- several timelike trajectories launched from the same initial point
- one nearby trajectory from slightly perturbed initial data
- an optional translucent cylinder at a chosen reference radius r_marker

Suggested use
-------------
- Set r_marker = r_static_equatorial(M) to highlight the equatorial
  stationary-limit radius.
- Or set r_marker = r_plus for the outer horizon.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp
from sympy import Matrix


# ============================================================================
# 1. METRIC MODEL
# ============================================================================

class MetricModel:
    def __init__(self, coords, g_sym, name="Metric"):
        self.coords = coords
        self.g_sym = g_sym
        self.name = name
        self.dim = len(coords)

        self.metric_func = sp.lambdify(coords, g_sym, "numpy")
        self.dg_sym = self._build_metric_derivatives()
        self.dg_func = sp.lambdify(coords, self._flatten_dg(self.dg_sym), "numpy")

    def _build_metric_derivatives(self):
        n = self.dim
        dg = [[[0] * n for _ in range(n)] for _ in range(n)]
        for a in range(n):
            for i in range(n):
                for j in range(n):
                    dg[a][i][j] = sp.diff(self.g_sym[i, j], self.coords[a])
        return dg

    def _flatten_dg(self, dg):
        n = self.dim
        return [dg[a][i][j] for a in range(n) for i in range(n) for j in range(n)]

    def metric(self, x):
        return np.array(self.metric_func(*x), dtype=float)

    def metric_derivatives(self, x):
        return np.array(self.dg_func(*x), dtype=float).reshape(self.dim, self.dim, self.dim)

    def christoffel(self, x):
        """
        Compute Γ^μ_ab numerically:
            Γ^μ_ab = 1/2 g^{μν}(∂_a g_{νb} + ∂_b g_{νa} - ∂_ν g_{ab})
        """
        g = self.metric(x)
        g_inv = np.linalg.inv(g)
        dg = self.metric_derivatives(x)
        n = self.dim

        Gamma = np.zeros((n, n, n), dtype=float)
        for mu in range(n):
            for a in range(n):
                for b in range(n):
                    s = 0.0
                    for nu in range(n):
                        s += g_inv[mu, nu] * (
                            dg[a, nu, b] +
                            dg[b, nu, a] -
                            dg[nu, a, b]
                        )
                    Gamma[mu, a, b] = 0.5 * s
        return Gamma


# ============================================================================
# 2. KERR METRIC (EQUATORIAL SECTOR)
# ============================================================================

def build_kerr_model(M=1.0, a_spin=0.6):
    t, r, phi = sp.symbols("t r phi", real=True)

    Msp = sp.Float(M)
    asp = sp.Float(a_spin)

    # Equatorial Boyer-Lindquist sector: theta = pi/2
    Delta = r**2 - 2 * Msp * r + asp**2

    g_tt = -(1 - 2 * Msp / r)
    g_rr = r**2 / Delta
    g_tphi = -2 * Msp * asp / r
    g_phiphi = r**2 + asp**2 + 2 * Msp * asp**2 / r

    g_sym = Matrix([
        [g_tt, 0, g_tphi],
        [0, g_rr, 0],
        [g_tphi, 0, g_phiphi]
    ])

    return MetricModel([t, r, phi], g_sym, name=f"Kerr equatorial sector (M={M}, a={a_spin})")


# ============================================================================
# 3. REFERENCE RADII
# ============================================================================

def kerr_outer_horizon(M=1.0, a_spin=0.6):
    if abs(a_spin) > M:
        return None
    return M + np.sqrt(M**2 - a_spin**2)


def r_static_equatorial(M=1.0):
    """
    Equatorial stationary-limit radius for Kerr:
        r_static(theta=pi/2) = 2M
    """
    return 2.0 * M


# ============================================================================
# 4. INITIAL CONDITIONS
# ============================================================================

def solve_ut_from_spatial_proper_velocity(model, x0, ur, uphi):
    """
    Given u^r = dr/dτ and u^phi = dphi/dτ, solve for u^t from
    timelike normalization g_{μν}u^μu^ν = -1.
    """
    g = model.metric(x0)

    a = g[0, 0]
    b = 2 * (g[0, 1] * ur + g[0, 2] * uphi)
    c = g[1, 1] * ur**2 + 2 * g[1, 2] * ur * uphi + g[2, 2] * uphi**2 + 1.0

    disc = b**2 - 4 * a * c
    if disc < 0:
        raise ValueError(f"Negative normalization discriminant: {disc}")

    roots = [(-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)]
    candidates = [u for u in roots if u > 0]
    if not candidates:
        raise ValueError("No future-directed timelike root found for u^t.")
    return min(candidates)


def solve_u_from_coordinate_rates(model, x0, dr_dt, dphi_dt):
    """
    Given dr/dt and dphi/dt, construct the proper-time velocity u^μ.
    """
    v = np.array([1.0, dr_dt, dphi_dt], dtype=float)
    g = model.metric(x0)

    norm = v @ g @ v
    if norm >= 0:
        raise ValueError("Chosen coordinate rates are not timelike at the initial point.")

    return v / np.sqrt(-norm)


def build_initial_velocity(model, x0, mode="coordinate", ur=0.0, uphi=0.0, dr_dt=0.0, dphi_dt=0.0):
    if mode == "proper":
        ut = solve_ut_from_spatial_proper_velocity(model, x0, ur, uphi)
        return np.array([ut, ur, uphi], dtype=float)

    if mode == "coordinate":
        return solve_u_from_coordinate_rates(model, x0, dr_dt, dphi_dt)

    raise ValueError("mode must be either 'proper' or 'coordinate'")


def make_nearby_initial_data(x0, u0, epsilon=1e-4, direction="radial", enabled=True):
    """
    Build a second nearby trajectory by perturbing the initial data.
    """
    if not enabled or epsilon == 0.0:
        return None, None

    x1 = x0.copy()
    u1 = u0.copy()

    if direction == "radial":
        x1[1] += epsilon
    elif direction == "angular":
        x1[2] += epsilon
    elif direction == "velocity":
        u1[2] += epsilon
    else:
        raise ValueError("direction must be 'radial', 'angular', or 'velocity'")

    return x1, u1


# ============================================================================
# 5. ACCELERATION MODELS
# ============================================================================

def zero_acceleration(lam, x, u):
    return np.zeros_like(u)


def orthogonalized_constant_acceleration(spatial_vec):
    """
    spatial_vec = [a^r, a^phi]
    Solve a^t from u_mu a^mu = 0.
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
# 6. ODE SYSTEM
# ============================================================================

def make_worldline_odes(model, acceleration=None):
    n = model.dim
    if acceleration is None:
        acceleration = zero_acceleration

    def odes(lam, Y):
        x = Y[0:n]
        u = Y[n:2*n]

        Gamma = model.christoffel(x)
        dx = u
        a_ext = acceleration(lam, x, u)

        du = np.zeros(n)
        for mu in range(n):
            du[mu] = -sum(
                Gamma[mu, a, b] * u[a] * u[b]
                for a in range(n)
                for b in range(n)
            ) + a_ext[mu]

        return np.concatenate([dx, du])

    return odes


# ============================================================================
# 7. SOLVERS
# ============================================================================

def solve_single_trajectory(
    model,
    x0,
    u0,
    lam_span=(0, 10),
    t_eval=None,
    acceleration=None,
    rtol=1e-7,
    atol=1e-9
):
    if t_eval is None:
        t_eval = np.linspace(lam_span[0], lam_span[1], 400)

    odes = make_worldline_odes(model, acceleration=acceleration)
    Y0 = np.concatenate([x0, u0])

    sol = solve_ivp(
        odes,
        lam_span,
        Y0,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        method="DOP853"
    )
    return sol


def solve_trajectory_family(
    model,
    x0,
    initial_velocity_list,
    lam_span=(0, 10),
    t_eval=None,
    nearby_enabled=True,
    nearby_epsilon=1e-2,
    nearby_direction="radial",
    acceleration=None,
    rtol=1e-7,
    atol=1e-9
):
    if t_eval is None:
        t_eval = np.linspace(lam_span[0], lam_span[1], 400)

    main_solutions = []
    nearby_solution = None

    for i, u0 in enumerate(initial_velocity_list):
        sol = solve_single_trajectory(
            model=model,
            x0=x0,
            u0=u0,
            lam_span=lam_span,
            t_eval=t_eval,
            acceleration=acceleration,
            rtol=rtol,
            atol=atol
        )
        main_solutions.append(sol)

        if i == 0 and nearby_enabled:
            x1, u1 = make_nearby_initial_data(
                x0=x0,
                u0=u0,
                epsilon=nearby_epsilon,
                direction=nearby_direction,
                enabled=True
            )
            nearby_solution = solve_single_trajectory(
                model=model,
                x0=x1,
                u0=u1,
                lam_span=lam_span,
                t_eval=t_eval,
                acceleration=acceleration,
                rtol=rtol,
                atol=atol
            )

    return main_solutions, nearby_solution


# ============================================================================
# 8. PLOTTING
# ============================================================================

def cylindrical_embedding(r, phi):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y


def add_translucent_cylindrical_shell(ax, radius, t_min, t_max, color="red", alpha=0.10, n_theta=120, n_z=40):
    theta = np.linspace(0, 2 * np.pi, n_theta)
    z = np.linspace(t_min, t_max, n_z)
    Theta, Z = np.meshgrid(theta, z)

    X = radius * np.cos(Theta)
    Y = radius * np.sin(Theta)

    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0, shade=False, zorder=1)


def plot_trajectory_family(
    main_solutions,
    nearby_solution=None,
    title="Kerr frame-dragging visualization",
    shell_radius=None,
    shell_label=None,
    show_shell=True
):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    colors = ["black", "tab:blue", "tab:green", "tab:purple", "tab:orange"]
    all_t = []

    for i, sol in enumerate(main_solutions):
        t = sol.y[0]
        r = sol.y[1]
        phi = sol.y[2]
        all_t.append(t)

        x, y = cylindrical_embedding(r, phi)

        if i == 0:
            ax.plot(x, y, t, color="black", linewidth=3, label="Main timelike trajectory")
        else:
            c = colors[i % len(colors)]
            ax.plot(x, y, t, color=c, linewidth=2, alpha=0.95)

    if nearby_solution is not None:
        t_n = nearby_solution.y[0]
        r_n = nearby_solution.y[1]
        phi_n = nearby_solution.y[2]
        all_t.append(t_n)

        x_n, y_n = cylindrical_embedding(r_n, phi_n)
        ax.plot(x_n, y_n, t_n, color="dimgray", linewidth=2, label="Nearby trajectory")

    t_min = min(np.min(t) for t in all_t)
    t_max = max(np.max(t) for t in all_t)

    if shell_radius is not None:
        theta = np.linspace(0, 2 * np.pi, 300)
        x_c = shell_radius * np.cos(theta)
        y_c = shell_radius * np.sin(theta)
        z_c = np.zeros_like(theta)

        label = shell_label if shell_label is not None else f"Reference radius ({shell_radius:.3f})"
        ax.plot(x_c, y_c, z_c, color="red", linestyle="--", linewidth=2, label=label, zorder=100)

        if show_shell:
            add_translucent_cylindrical_shell(ax, shell_radius, t_min, t_max, color="red", alpha=0.10)

    # Initial point
    t0 = main_solutions[0].y[0, 0]
    r0 = main_solutions[0].y[1, 0]
    phi0 = main_solutions[0].y[2, 0]
    x0_plot, y0_plot = cylindrical_embedding(r0, phi0)

    ax.scatter([x0_plot], [y0_plot], [t0], color="magenta", s=28, marker="o", label="Initial point", zorder=130)
    ax.scatter([0], [0], [0], color="black", s=60, marker="x", label="Coordinate origin", zorder=120)

    # Make the shell look visually upright
    ax.set_box_aspect((1, 1, 1.8))
    ax.view_init(elev=20, azim=45)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("t")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ============================================================================
# 9. USER INPUT DASHBOARD
# ============================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------------
    # KERR PARAMETERS
    # ------------------------------------------------------------------------
    M = 1.0
    a_spin = 0.6

    model = build_kerr_model(M=M, a_spin=a_spin)

    # ------------------------------------------------------------------------
    # STARTING POINT
    # ------------------------------------------------------------------------
    t0 = 0.0
    r_plus = kerr_outer_horizon(M=M, a_spin=a_spin)
    r_static = r_static_equatorial(M=M)

    # Start outside the horizon
    r0 = 2.6
    phi0 = 0.0
    x0 = np.array([t0, r0, phi0], dtype=float)

    # ------------------------------------------------------------------------
    # VELOCITY MODE
    # ------------------------------------------------------------------------
    # "coordinate" is most intuitive here for frame dragging
    velocity_mode = "coordinate"

    # These are deliberately small angular coordinate velocities.
    # The point is to show the geometry twisting the trajectories.
    velocity_specs = [
        {"dr_dt": -0.015, "dphi_dt": 0.000},
        {"dr_dt": -0.020, "dphi_dt": 0.005},
        {"dr_dt": -0.025, "dphi_dt": 0.010},
        {"dr_dt": -0.030, "dphi_dt": 0.015},
    ]

    initial_velocity_list = [
        build_initial_velocity(
            model, x0,
            mode=velocity_mode,
            dr_dt=spec["dr_dt"],
            dphi_dt=spec["dphi_dt"]
        )
        for spec in velocity_specs
    ]

    # ------------------------------------------------------------------------
    # NEARBY TRAJECTORY
    # ------------------------------------------------------------------------
    nearby_enabled = True
    nearby_epsilon = 0.5
    nearby_direction = "radial"

    # ------------------------------------------------------------------------
    # ACCELERATION
    # ------------------------------------------------------------------------
    # For pure frame dragging, keep this zero.
    acceleration = zero_acceleration

    # ------------------------------------------------------------------------
    # INTEGRATION
    # ------------------------------------------------------------------------
    lam_span = (0, 3)
    t_eval = np.linspace(0, 2.5, 500)

    main_solutions, nearby_solution = solve_trajectory_family(
        model=model,
        x0=x0,
        initial_velocity_list=initial_velocity_list,
        lam_span=lam_span,
        t_eval=t_eval,
        nearby_enabled=nearby_enabled,
        nearby_epsilon=nearby_epsilon,
        nearby_direction=nearby_direction,
        acceleration=acceleration,
        rtol=1e-7,
        atol=1e-9
    )

    # ------------------------------------------------------------------------
    # REFERENCE CYLINDER
    # ------------------------------------------------------------------------
    # Choose what cylinder you want to show:
    #
    # 1) Outer horizon:
    # r_marker = r_plus
    # label = rf"Outer horizon ($r_+ = {r_plus:.3f}$)"
    #
    # 2) Equatorial stationary-limit radius:
    r_marker = r_static
    label = rf"Equatorial stationary limit ($r = {r_marker:.3f}$)"

    # 3) Or set any custom radius by hand:
    # r_marker = 2.3
    # label = rf"Reference radius ($r = {r_marker:.3f}$)"

    # ------------------------------------------------------------------------
    # PLOT
    # ------------------------------------------------------------------------
    plot_trajectory_family(
        main_solutions,
        nearby_solution=nearby_solution,
        title="Kerr spacetime (equatorial sector): frame-dragging visualization",
        shell_radius=r_marker,
        shell_label=label,
        show_shell=True
    )