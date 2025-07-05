#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def second_derivative_x(phi):
    """
    Calcula la segunda derivada en x usando diferencias finitas centradas.

    Args:
        phi (ndarray): Matriz 2D del potencial.

    Returns:
        ndarray: Segunda derivada en x.
    """
    d2x = np.zeros_like(phi)
    for i in range(1, phi.shape[0] - 1):
        for j in range(1, phi.shape[1] - 1):  # Corregido rango de j
            d2x[i, j] = phi[i+1, j] - 2 * phi[i, j] + phi[i-1, j]
    return d2x

def second_derivative_y(phi):
    """
    Calcula la segunda derivada en y usando diferencias finitas centradas.

    Args:
        phi (ndarray): Matriz 2D del potencial.

    Returns:
        ndarray: Segunda derivada en y.
    """
    d2y = np.zeros_like(phi)
    for i in range(1, phi.shape[0] - 1):  # Corregido rango de i
        for j in range(1, phi.shape[1] - 1):
            d2y[i, j] = phi[i, j+1] - 2 * phi[i, j] + phi[i, j-1]
    return d2y

def condiciones_frontera(phi):
    """
    Aplica condiciones de frontera:
    - Bordes exteriores a 0V
    - Electrodo izquierdo a +1 V (centrado verticalmente)
    - Electrodo derecho a -1 V (centrado verticalmente)
    """
    # Bordes exteriores
    phi[0, :] = 0.0
    phi[-1, :] = 0.0
    phi[:, 0] = 0.0
    phi[:, -1] = 0.0

    # Calcular dimensiones
    n_filas, n_columnas = phi.shape
    Long_x = n_columnas - 1  # Longitud en x
    Long_y = n_filas - 1     # Longitud en y

    # Posiciones relativas (20% y 80% en x, 20% a 80% en y)
    electrodo_P = int(round(0.2 * Long_x))   # Columna izquierda al 20%
    electrodo_N = int(round(0.8 * Long_x))   # Columna derecha al 80%
    inicio_fila = int(round(0.2 * Long_y))  # Inicio vertical al 20%
    final_fila = int(round(0.8 * Long_y))    # Fin vertical al 80%

    # Aplicar voltajes a electrodos
    for i in range(inicio_fila, final_fila + 1):
        phi[i, electrodo_P] = 1.0    # +1V
        phi[i, electrodo_N] = -1.0  # -1V

    return phi

def gauss_seidel_modified(M, omega, tolerance):
    """
    Resuelve la ecuación de Laplace 2D con método de Gauss-Seidel y relajación.

    Args:
        M (int): Tamaño de la grilla (M+1 x M+1 puntos)
        omega (float): Factor de sobre-relajación (1 ≤ omega ≤ 2)
        tolerance (float): Tolerancia para convergencia

    Returns:
        tuple: 
            phi (ndarray): Potencial resultante
            its (int): Número de iteraciones
    """
    # Inicializar matriz con condiciones de frontera
    phi = np.zeros((M + 1, M + 1), dtype=float)
    phi = condiciones_frontera(phi)
    phi_old = phi.copy()
    

    delta = 1.0
    its = 0

    while delta > tolerance:
        its += 1
        # Guardar estado anterior para comparación
        phi_prev = phi.copy()
        
        # Actualizar puntos interiores
        for i in range(1, M):
            for j in range(1, M):
                # Calcular nuevo valor con Gauss-Seidel
                new_val = 0.25 * (
                    phi[i+1, j] + phi[i-1, j] + 
                    phi[i, j+1] + phi[i, j-1]
                )
                # Aplicar sobre-relajación
                phi[i, j] = (1 - omega) * phi[i, j] + omega * new_val
        
        # Reforzar condiciones de frontera
        phi = condiciones_frontera(phi)
        
        # Calcular cambio máximo
        delta = np.max(np.abs(phi - phi_prev))

    return phi, its
# Ejecutar simulación
gauss_seidel, iterations = gauss_seidel_modified(100, 1.5, 1e-5)
print(f"Iteraciones: {iterations}")

# Visualizar resultados
plt.figure(figsize=(8, 6))
plt.imshow(gauss_seidel, origin="lower", cmap='jet')
plt.colorbar(label="Potencial φ")
plt.title("Solución Gauss-Seidel de la Ecuación de Laplace")
plt.xlabel("Posición X")
plt.ylabel("Posición Y")
plt.show()
