#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def segunda_derivada_x(phi):
    """
    Calcula la segunda derivada en la dirección x de la matriz phi.

    La derivada se calcula usando diferencias finitas centradas:
        d²φ/dx² ≈ φ[i+1,j] - 2φ[i,j] + φ[i-1,j]

    Args:
        phi (ndarray): Matriz 2D con los valores del potencial.

    Returns:
        (ndarray): Segunda derivada en x, misma forma que phi.
    """
    d2x = np.zeros_like(phi)
    for i in range(1, phi.shape[0] - 1):
        for j in range(1, phi.shape[1] - 1):
            d2x[i, j] = phi[i+1, j] - 2 * phi[i, j] + phi[i-1, j]
    return d2x


def segunda_derivada_y(phi):
    """
    Calcula la segunda derivada en la dirección y de la matriz phi.

    La derivada se calcula usando diferencias finitas centradas:
        d²φ/dy² ≈ φ[i,j+1] - 2φ[i,j] + φ[i,j-1]

    Args:
        phi (ndarray): Matriz 2D con los valores del potencial.

    Returns:
        (ndarray): Segunda derivada en y, misma forma que phi.
    """
    d2y = np.zeros_like(phi)
    for i in range(1, phi.shape[0] - 1):
        for j in range(1, phi.shape[1] - 1):
            d2y[i, j] = phi[i, j+1] - 2 * phi[i, j] + phi[i, j-1]
    return d2y

def condiciones_frontera(phi):
    """
    Aplica condiciones de frontera según el diagrama:
    - Bordes a 0 V
    - Electrodo izquierdo a +1 V
    - Electrodo derecho a -1 V
    """
    phi[0, :] = 0.0
    phi[-1, :] = 0.0
    phi[:, 0] = 0.0
    phi[:, -1] = 0.0

    n_filas, n_columnas = phi.shape
    Long_x = n_columnas - 1 #longitud x
    Long_y = n_filas - 1 #longitud y

    electrodo_P = int(0.2 * Long_x) #electrodo positivo esta en el 30% desde la izquierda
    electrodo_N = int(0.8 * Long_x) #electrodo positivo esta en el 70% desde la izquierda
    inicio_fila = int(0.2 * Long_y) #inicio vertical en el 20%
    final_fila = int(0.8 * Long_y)  #final vertical en el 80%

    for i in range(inicio_fila, final_fila + 1):
        phi[i, electrodo_P] = 1.0
        phi[i, electrodo_N] = -1.0
    return phi

def laplace(phi):
    """
    Realiza una iteración del método de Jacobi usando el operador de Laplace.

    Esta función calcula el laplaciano de la matriz de potencial `phi`
    sumando sus segundas derivadas en las direcciones x e y (previamente
    calculadas con bucles), y luego actualiza el potencial usando
    slicing solo para los puntos interiores.

    Las condiciones de frontera (por ejemplo, potencial fijo en los bordes
    o electrodos internos) se re-aplican después de la actualización
    para asegurar la estabilidad.

    Args:
        phi (ndarray): arreglo 2D que representa el potencial eléctrico actual.

    Returns:
        (ndarray): arreglo 2D actualizado del potencial luego de una iteración.
    """

    d2x = segunda_derivada_x(phi)
    d2y = segunda_derivada_y(phi)
    laplaciano = d2x + d2y

    phi_new = phi.copy()
    phi_new[1:-1, 1:-1] = phi[1:-1, 1:-1] + 0.25 * laplaciano[1:-1, 1:-1]

    phi_new = condiciones_frontera(phi_new)

    return phi_new


def jacobi_relaxation(N, tolerance):
    """
    Ejecuta el método de relajación de Jacobi para resolver la ecuación de Laplace
    en una grilla bidimensional de tamaño (N+1) x (N+1).

    Args:
        N (int) : Tamaño de la grilla (produce una grilla de (N+1) × (N+1))
        tolerance (float) : Criterio de convergencia para la diferencia máxima entre iteraciones.

    Returns:
        phi (numpy.ndarray): Arreglo 2D con los valores calculados de la función potencial en la grilla.
        its (int): Número de iteraciones realizadas hasta alcanzar la tolerancia.
    """
    phi = np.zeros((N + 1, N + 1), dtype=float)
    phi = condiciones_frontera(phi)

    delta = 1.0
    its = 0

    while delta > tolerance:
        its += 1
        phi_new = laplace(phi)
        delta = np.max(np.abs(phi - phi_new))
        phi = phi_new

    return phi, its
jacobi_vals, iterations = jacobi_relaxation(100, 1e-5)
print(f"Iteraciones: {iterations}")
plt.imshow(jacobi_vals, origin="lower")
plt.colorbar(label="Potencial φ")
plt.jet()
plt.show()
