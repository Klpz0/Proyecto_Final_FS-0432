#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def condiciones_frontera(phi):
    """
    Aplica condiciones de frontera según el diagrama:
    - Bordes a 0 V
    - Electrodo izquierdo a +1 V
    - Electrodo derecho a -1 V

    Args:
        phi (ndarray): Matriz 2D del potencial eléctrico.

    Returns:
        (ndarray): Matriz con condiciones de frontera aplicada.
    """
    phi[0, :] = 0.0
    phi[-1, :] = 0.0
    phi[:, 0] = 0.0
    phi[:, -1] = 0.0

    n_filas, n_columnas = phi.shape
    Long_x = n_columnas - 1
    Long_y = n_filas - 1

    electrodo_P = int(0.2 * Long_x)
    electrodo_N = int(0.8 * Long_x)
    inicio_fila = int(0.2 * Long_y)
    final_fila = int(0.8 * Long_y)

    for i in range(inicio_fila, final_fila + 1):
        phi[i, electrodo_P] = 1.0
        phi[i, electrodo_N] = -1.0

    return phi


def laplace_jacobi_sor(phi, omega):
    """
    Realiza una iteración del método de Jacobi con sobre-relajación (SOR)
    para resolver la ecuación de Laplace (sin término fuente).

    Args:
        phi (ndarray): Matriz 2D actual del potencial.
        omega (float): Factor de sobre-relajación (1 < omega < 2).

    Returns:
        (ndarray): Matriz actualizada del potencial tras una iteración.
    """
    phi_new = phi.copy()

    for i in range(1, phi.shape[0] - 1):
        for j in range(1, phi.shape[1] - 1):
            phi_est = 0.25 * (
                phi[i+1, j] + phi[i-1, j] +
                phi[i, j+1] + phi[i, j-1]
            )
            phi_new[i, j] = (1 - omega) * phi[i, j] + omega * phi_est

    phi_new = condiciones_frontera(phi_new)
    return phi_new


def jacobi_sor_relaxation(N, tolerance, omega):
    """
    Ejecuta el método de Jacobi-SOR para resolver la ecuación de Laplace
    en una grilla bidimensional de tamaño (N+1) x (N+1).

    Args:
        N (int): Tamaño de la grilla (produce una grilla de (N+1) × (N+1))
        tolerance (float): Criterio de convergencia.
        omega (float): Factor de sobre-relajación (1 < omega < 2)

    Returns:
        phi (ndarray): Solución final del potencial eléctrico.
        its (int): Número de iteraciones realizadas.
    """
    phi = np.zeros((N + 1, N + 1), dtype=float)
    phi = condiciones_frontera(phi)

    delta = 1.0
    its = 0

    while delta > tolerance:
        its += 1
        phi_new = laplace_jacobi_sor(phi, omega)
        delta = np.max(np.abs(phi - phi_new))
        phi = phi_new

    return phi, its


# Parámetros y ejecución
jacobi_vals, iterations = jacobi_sor_relaxation(100, 1e-5, omega=1.8)
print(f"Iteraciones: {iterations}")

# Visualización
plt.imshow(jacobi_vals, origin="lower", cmap="jet")
plt.colorbar(label="Potencial φ")
plt.title("Solución de la ecuación de Laplace con Jacobi-SOR")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
