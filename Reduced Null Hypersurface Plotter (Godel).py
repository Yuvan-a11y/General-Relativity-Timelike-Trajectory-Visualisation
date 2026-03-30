# -*- coding: utf-8 -*-
"""
Gödel Spacetime Geodesics - Cylindrical Coordinates with Hypersurface Embedding
================================================================================

Author: Yuvan Raam Chandra
Senior Thesis - Theoretical Physics
November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import sympy as sp
from sympy import Matrix, sinh, simplify, sqrt, cosh


# =========================================================================
# 1. METRIC AND HYPERSURFACE
# =========================================================================

def compute_christoffel():
    """Compute Christoffel symbols symbolically"""
    t, r, phi = sp.symbols("t r phi", real=True)
    
    g = Matrix([
        [-1,  0,  -sp.sqrt(2)*sp.sinh(r)**2],
        [0,   1,  0],
        [-sp.sqrt(2)*sp.sinh(r)**2,  0,  sp.sinh(r)**2 - sp.sinh(r)**4]
    ])
    
    coords = [t, r, phi]
    g_inv = g.inv()
    
    Gamma = [[[0 for _ in range(3)] for _ in range(3)] for _ in range(3)]
    
    for mu in range(3):
        for alpha in range(3):
            for beta in range(3):
                expr = 0
                for nu in range(3):
                    expr += g_inv[mu, nu] * (
                        sp.diff(g[nu, beta], coords[alpha]) +
                        sp.diff(g[nu, alpha], coords[beta]) -
                        sp.diff(g[alpha, beta], coords[nu])
                    ) / 2
                Gamma[mu][alpha][beta] = simplify(expr)
    
    gamma_flat = [Gamma[mu][alpha][beta] 
                  for mu in range(3) for alpha in range(3) for beta in range(3)]
    print("Christoffels:")
    print(gamma_flat)
    return sp.lambdify([t, r, phi], gamma_flat, 'numpy')

CHRISTOFFEL = compute_christoffel()

# =========================================================================
# 5. Hypersurface Function
# =========================================================================

def dt_dphi_future(r):
    """Future sheet: dt/dφ = -√2 sinh²(r) + sinh(r) cosh(r)"""
    sr = np.sinh(r)
    cr = np.cosh(r)
    return -np.sqrt(2) * sr**2 + sr * cr

def dt_dphi_past(r):
    """Past sheet: dt/dφ = -√2 sinh²(r) - sinh(r) cosh(r)"""
    sr = np.sinh(r)
    cr = np.cosh(r)
    return -np.sqrt(2) * sr**2 - sr * cr

# Generate hypersurface
R_VALS = np.linspace(0, 1, 200)
PHI_VALS = np.linspace(0, 2*np.pi, 200)
R_MESH, PHI_MESH = np.meshgrid(R_VALS, PHI_VALS)

T_FUTURE_MESH = np.zeros_like(R_MESH)
T_PAST_MESH = np.zeros_like(R_MESH)

for i in range(len(R_VALS)):
    T_FUTURE_MESH[:, i] = dt_dphi_future(R_VALS[i]) * PHI_VALS
    T_PAST_MESH[:, i] = dt_dphi_past(R_VALS[i]) * PHI_VALS

X_MESH = R_MESH * np.cos(PHI_MESH)
Y_MESH = R_MESH * np.sin(PHI_MESH)

r_crit = np.arcsinh(1.0)
print(f"\n{'─'*90}")
print(f"Critical radius: r_crit = {r_crit:.6f}")
print(f"At r_crit: dt/dφ = {dt_dphi_future(r_crit):.2e}\n")

# =========================================================================
# 6. VISUALIZATION
# =========================================================================

print("="*90)
print("GENERATING VISUALIZATION")
print("="*90)

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')

# Hypersurface
ax.plot_surface(X_MESH, Y_MESH, T_FUTURE_MESH, cmap='viridis', alpha=0.7, 
                edgecolor='none', antialiased=True, zorder=1, label='Future sheet')
ax.plot_surface(X_MESH, Y_MESH, T_PAST_MESH, cmap='plasma', alpha=0.3,
                edgecolor='none', antialiased=True, zorder=1, label='Past sheet')

# Critical radius circle
theta_crit = np.linspace(0, 2*np.pi, 100)
x_crit = r_crit * np.cos(theta_crit)
y_crit = r_crit * np.sin(theta_crit)
ax.plot(x_crit, y_crit, 0*theta_crit, color='red', linewidth=2, linestyle='--',
        label=f'Critical Radius (r={r_crit:.3f})', zorder=100)

# Labels
ax.set_xlabel('\nx = r cos(φ)', fontsize=18, weight='bold')
ax.set_ylabel('\ny = r sin(φ)', fontsize=18, weight='bold')
ax.set_zlabel('\nt (coordinate time)', fontsize=18, weight='bold')

title = 'Gödel Spacetime: Null Hypersurface'
ax.set_title(title, fontsize=22, weight='bold', pad=25)

ax.legend(fontsize=14, loc='upper left', framealpha=0.95)
ax.view_init(elev=30, azim=45)
ax.grid(True, alpha=0.4)

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.savefig('godel_null_plot.png', dpi=50, bbox_inches='tight')

plt.show()