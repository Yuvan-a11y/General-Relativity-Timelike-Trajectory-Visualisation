# -*- coding: utf-8 -*-
"""
Stationary-axisymmetric metric toolkit
======================================

Reduced 3-coordinate sector:
    x^0 = t, x^1 = r, x^2 = varphi

What this script does
---------------------
1. Accepts a symbolic metric g_{mu nu}(x)
2. Computes:
      - inverse metric
      - Christoffel symbols
      - Riemann tensor
      - Ricci tensor
      - Ricci scalar
      - Einstein tensor
3. Allows evaluation of an energy-density scalar
      rho = T_{mu nu} u^mu u^nu
   from Einstein's equations, with optional cosmological constant.
4. Plots a stationary energy-density surface on a t = const slice.

Edits in this version
---------------------
- Removes the duplicated 0 / 2π seam by using endpoint=False in the angular grid.
- Uses a single solid surface color instead of a colormap, so tiny numerical variations
  do not create a misleading yellow rim.
"""

import numpy as np
import matplotlib.pyplot as plt
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
        self.ricci_sym = self._build_ricci_sym()
        self.ricci_scalar_sym = self._build_ricci_scalar_sym()
        self.einstein_sym = self._build_einstein_sym()

        self.metric_func = sp.lambdify(coords, g_sym, "numpy")
        self.gamma_func = sp.lambdify(coords, self._flatten3(self.gamma_sym), "numpy")
        self.riemann_func = sp.lambdify(coords, self._flatten4(self.riemann_sym), "numpy")
        self.ricci_func = sp.lambdify(coords, self._flatten2(self.ricci_sym), "numpy")
        self.ricci_scalar_func = sp.lambdify(coords, self.ricci_scalar_sym, "numpy")
        self.einstein_func = sp.lambdify(coords, self._flatten2(self.einstein_sym), "numpy")

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
        Gamma = self.gamma_sym
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

    def _build_ricci_sym(self):
        n = self.dim
        R = self.riemann_sym
        Ric = [[0] * n for _ in range(n)]

        for nu in range(n):
            for b in range(n):
                Ric[nu][b] = simplify(sum(R[mu][nu][mu][b] for mu in range(n)))

        return Ric

    def _build_ricci_scalar_sym(self):
        n = self.dim
        Ric = self.ricci_sym
        return simplify(sum(self.g_inv_sym[i, j] * Ric[i][j] for i in range(n) for j in range(n)))

    def _build_einstein_sym(self):
        n = self.dim
        Ric = self.ricci_sym
        Rsc = self.ricci_scalar_sym
        G = [[0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                G[i][j] = simplify(Ric[i][j] - sp.Rational(1, 2) * self.g_sym[i, j] * Rsc)

        return G

    def _flatten2(self, A):
        n = self.dim
        return [A[i][j] for i in range(n) for j in range(n)]

    def _flatten3(self, A):
        n = self.dim
        return [A[i][j][k] for i in range(n) for j in range(n) for k in range(n)]

    def _flatten4(self, A):
        n = self.dim
        return [A[i][j][k][l] for i in range(n) for j in range(n) for k in range(n) for l in range(n)]

    def metric(self, x):
        return np.array(self.metric_func(*x), dtype=float)

    def christoffel(self, x):
        return np.array(self.gamma_func(*x), dtype=float).reshape(self.dim, self.dim, self.dim)

    def riemann(self, x):
        return np.array(self.riemann_func(*x), dtype=float).reshape(self.dim, self.dim, self.dim, self.dim)

    def ricci(self, x):
        return np.array(self.ricci_func(*x), dtype=float).reshape(self.dim, self.dim)

    def ricci_scalar(self, x):
        return float(self.ricci_scalar_func(*x))

    def einstein(self, x):
        return np.array(self.einstein_func(*x), dtype=float).reshape(self.dim, self.dim)


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
# 3. STRESS-ENERGY / ENERGY-DENSITY TOOLS
# ============================================================================

def stress_energy_from_einstein(model, x, Lambda=0.0, eight_pi_G=8.0 * np.pi):
    """
    Einstein equation convention used here:
        G_{mu nu} + Lambda g_{mu nu} = 8 pi G T_{mu nu}

    Returns T_{mu nu}.
    """
    Gmn = model.einstein(x)
    gmn = model.metric(x)
    return (Gmn + Lambda * gmn) / eight_pi_G


def normalize_timelike_vector(model, x, u_guess):
    """
    Normalize a timelike vector u so that g(u,u) = -1.
    """
    g = model.metric(x)
    u_guess = np.array(u_guess, dtype=float)
    norm = u_guess @ g @ u_guess
    if norm >= 0:
        raise ValueError("The supplied observer vector is not timelike.")
    return u_guess / np.sqrt(-norm)


def energy_density(model, x, u, Lambda=0.0, eight_pi_G=8.0 * np.pi):
    """
    rho = T_{mu nu} u^mu u^nu
    """
    Tmn = stress_energy_from_einstein(model, x, Lambda=Lambda, eight_pi_G=eight_pi_G)
    u = np.array(u, dtype=float)
    return float(u @ Tmn @ u)


def static_observer_field(model, x):
    """
    A simple default observer field:
        u ~ partial_t
    normalized where timelike.

    This works only where the coordinate t-direction is timelike.
    """
    u_guess = np.array([1.0, 0.0, 0.0], dtype=float)
    return normalize_timelike_vector(model, x, u_guess)


# ============================================================================
# 4. PLOTTING ENERGY DENSITY ON A STATIONARY SLICE
# ============================================================================

def plot_stationary_energy_density(
    model,
    r_max=1.5,
    nr=250,
    ntheta=220,
    t_slice=0.0,
    Lambda=0.0,
    observer_builder=static_observer_field,
    critical_radius=None,
    title=None,
    zlabel=r"Energy density $\rho$",
    surface_color="#5b2a6e",
    alpha=0.85
):
    """
    Plot rho(r,varphi) on a stationary t = const slice in Cartesian embedding:
        x = r cos(varphi), y = r sin(varphi), z = rho
    """
    r_vals = np.linspace(1e-6, r_max, nr)
    # endpoint=False removes the duplicated 0/2π seam
    theta_vals = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    R, Theta = np.meshgrid(r_vals, theta_vals)

    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    Z = np.zeros_like(R)

    for i in range(ntheta):
        for j in range(nr):
            x_point = np.array([t_slice, R[i, j], Theta[i, j]], dtype=float)

            try:
                u = observer_builder(model, x_point)
                Z[i, j] = energy_density(model, x_point, u, Lambda=Lambda)
            except Exception:
                Z[i, j] = np.nan

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X, Y, Z,
        color=surface_color,
        alpha=alpha,
        linewidth=0,
        antialiased=True,
        shade=False
    )

    if critical_radius is not None:
        theta_c = np.linspace(0.0, 2.0 * np.pi, 300, endpoint=False)
        x_c = critical_radius * np.cos(theta_c)
        y_c = critical_radius * np.sin(theta_c)

        finite_Z = Z[np.isfinite(Z)]
        z_level = np.nanmin(finite_Z) if finite_Z.size else 0.0
        z_c = np.full_like(theta_c, z_level)

        ax.plot(
            x_c, y_c, z_c,
            color="red", linestyle="--", linewidth=1.5,
            label=rf"Critical radius $r_c \approx {critical_radius:.3f}$"
        )

    ax.set_xlabel(r"$X = r\cos\varphi$")
    ax.set_ylabel(r"$Y = r\sin\varphi$")
    ax.set_zlabel(zlabel)

    if title is None:
        title = f"{model.name}: stationary energy density"
    ax.set_title(title)

    if critical_radius is not None:
        ax.legend()

    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.show()


# ============================================================================
# 5. OPTIONAL: PRINT CURVATURE / EINSTEIN TENSOR AT A POINT
# ============================================================================

def print_geometry_summary(model, x):
    print(f"\nMetric model: {model.name}")
    print(f"Point x = {x}\n")

    print("Metric g_mu_nu:")
    print(model.metric(x))
    print()

    print("Ricci tensor R_mu_nu:")
    print(model.ricci(x))
    print()

    print("Ricci scalar R:")
    print(model.ricci_scalar(x))
    print()

    print("Einstein tensor G_mu_nu:")
    print(model.einstein(x))
    print()


# ============================================================================
# 6. RUN EXAMPLE: GÖDEL
# ============================================================================

if __name__ == "__main__":
    model = build_metric_model("godel")

    x_check = np.array([0.0, 0.5, 0.0], dtype=float)
    print_geometry_summary(model, x_check)

    r_crit = np.arcsinh(1.0)

    plot_stationary_energy_density(
        model=model,
        r_max=1.5,
        nr=220,
        ntheta=220,
        t_slice=0.0,
        Lambda=0.0,
        observer_builder=static_observer_field,
        critical_radius=r_crit,
        title=r"Gödel spacetime: energy density $\rho = T_{\mu\nu}u^\mu u^\nu$ on a stationary slice",
        zlabel=r"Energy density $\rho$",
        surface_color="#5b2a6e",
        alpha=0.85
    )