#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def insertar_cuadro_carga(rho, centro_x, centro_y, tamaño, valor):
    """Inserta una carga cuadrada en la matriz rho.

    Args:
        rho (ndarray): Matriz 2D de densidad de carga.
        centro_x (int): Coordenada x del centro de la carga.
        centro_y (int): Coordenada y del centro de la carga.
        tamaño (int): Tamaño del lado del cuadro de carga.
        valor (float): Valor de la carga a insertar.
    """
    mitad = tamaño // 2
    x0 = centro_x - mitad
    x1 = centro_x + mitad
    y0 = centro_y - mitad
    y1 = centro_y + mitad
    rho[x0:x1, y0:y1] = valor

def resolver_poisson_jacobi_sor(phi, rho, h, tolerancia=1e-5, max_iter=10000, omega=1.5):
    """Resuelve la ecuación de Poisson usando el método de Jacobi-SOR.

    Args:
        phi (ndarray): Matriz 2D inicial del potencial eléctrico.
        rho (ndarray): Matriz 2D de densidad de carga.
        h (float): Tamaño del paso en la malla.
        tolerancia (float): Criterio de convergencia.
        max_iter (int): Número máximo de iteraciones.
        omega (float): Factor de sobre-relajación (1 < omega < 2).

    Returns:
        ndarray: Matriz 2D con la solución del potencial eléctrico.
    """
    nx, ny = phi.shape
    phi_new = phi.copy()
    error = 1.0
    iteracion = 0

    while error > tolerancia and iteracion < max_iter:
        phi_old = phi_new.copy()

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                phi_est = 0.25 * (phi_old[i+1, j] + phi_old[i-1, j] + phi_old[i, j+1] + phi_old[i, j-1] - h**2 * rho[i, j])
                phi_new[i, j] = (1 - omega) * phi_old[i, j] + omega * phi_est

        error = np.max(np.abs(phi_new - phi_old))
        iteracion += 1

        if iteracion % 100 == 0:
            print(f"Iteración {iteracion}, error = {error:.2e}")

    print(f"Convergencia alcanzada en {iteracion} iteraciones con error = {error:.2e}")
    return phi_new

def main():
    """Función principal que configura el problema y grafica la solución."""
    N = 100
    h = 1.0
    phi = np.zeros((N, N))
    rho = np.zeros((N, N))

    # Insertar una carga positiva y una negativa
    insertar_cuadro_carga(rho, N//4, N//2, 10, 1.0)
    insertar_cuadro_carga(rho, 3*N//4, N//2, 10, -1.0)

    # Resolver con método Jacobi-SOR
    phi = resolver_poisson_jacobi_sor(phi, rho, h, tolerancia=1e-5, omega=1.8)

    # Graficar
    plt.imshow(phi, origin='lower', cmap='seismic')
    plt.colorbar(label='Potencial φ')
    plt.title('Solución de la ecuación de Laplace con Jacobi-SOR')
    plt.show()

main()

