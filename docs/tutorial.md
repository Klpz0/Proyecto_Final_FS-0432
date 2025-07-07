# Tutorial: Métodos de relajación

---

## Implementación del metodo de relajación de Jacobi:

El método de relajación de Jacobi es un proceso iterativo utilizado para resolver sistemas de ecuaciones lineales, especialmente los que provienen de la discretización de ecuaciones diferenciales, como la de Laplace. En una grilla discreta de tamaño n×n, cada punto interior de la malla se actualiza en función del promedio de sus vecinos, manteniendo fijos los valores de las fronteras según las condiciones del problema. Matemáticamente se ve de la siguiente manera: 

$\phi_{(x,y)} = \frac{1}{4} \cdot (\phi_{(x + a,y)} + \phi_{(x - a,y)} + \phi_{(x,y + a)} + \phi_{(x,y - a)})$
  
En cada iteración se calcula una nueva matriz con los valores actualizados, sin modificar la matriz original hasta completar todo el barrido. Este enfoque facilita la implementación, ya que no hay dependencias entre los cálculos de los distintos puntos en una misma iteración, sin embargo, resulta costoso computacionalmente. El proceso se detiene una vez se llega a la variación entre valores de los puntos de la grilla estén por de bajo de la tolerancia establecida en los parámetros iniciales establecidos.

???+ note "Método de Jacobi"

    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    def segunda_derivada_x(phi):
        d2x = np.zeros_like(phi)
        for i in range(1, phi.shape[0] - 1):
            for j in range(1, phi.shape[1] - 1):
                d2x[i, j] = phi[i+1, j] - 2 * phi[i, j] + phi[i-1, j]
        return d2x

    def segunda_derivada_y(phi):
        d2y = np.zeros_like(phi)
        for i in range(1, phi.shape[0] - 1):
            for j in range(1, phi.shape[1] - 1):
                d2y[i, j] = phi[i, j+1] - 2 * phi[i, j] + phi[i, j-1]
        return d2y

    def condiciones_frontera(phi):
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

    def laplace(phi):
        d2x = segunda_derivada_x(phi)
        d2y = segunda_derivada_y(phi)
        laplaciano = d2x + d2y

        phi_new = phi.copy()
        phi_new[1:-1, 1:-1] = phi[1:-1, 1:-1] + 0.25 * laplaciano[1:-1, 1:-1]
        phi_new = condiciones_frontera(phi_new)
        return phi_new

    def jacobi_relaxation(N, tolerance):
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
    plt.imshow(jacobi_vals, origin="lower", cmap="jet")
    plt.colorbar(label="Potencial φ")
    plt.show()
    ```

---

## Implementación del método de relajación de Gauss-Seidel

El método de Gauss-Seidel es una optimización del método iterativo de Jacobi que permite una convergencia más rápida al actualizar los valores directamente en la misma memoria durante cada iteración. A diferencia de Jacobi, que calcula los nuevos valores sin modificar los anteriores hasta completar toda la grilla, Gauss-Seidel reutiliza inmediatamente los valores recién actualizados, lo que reduce el número de iteraciones necesarias para alcanzar la convergencia. El esquema básico de actualización para resolver la ecuación de Laplace por diferencias finitas centrales es:

$\phi_{(x,y)} = \frac{1}{4} \cdot (\phi_{(x + a,y)} + \phi_{(x - a,y)} + \phi_{(x,y + a)} + \phi_{(x,y - a)})$

En este método, los valores actualizados se escriben directamente sobre la matriz original, lo que introduce una dependencia entre los cálculos dentro de la misma iteración, dificultando su paralelización directa. Sin embargo, este enfoque reduce el uso de memoria y tiende a converger más rápidamente que Jacobi.

???+ note "Método de Gauss-Seidel"
    
    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    def condiciones_frontera(phi):
        phi[0, :] = 0.0
        phi[-1, :] = 0.0
        phi[:, 0] = 0.0
        phi[:, -1] = 0.0

        n_filas, n_columnas = phi.shape
        Long_x = n_columnas - 1
        Long_y = n_filas - 1

        electrodo_P = int(round(0.2 * Long_x))
        electrodo_N = int(round(0.8 * Long_x))
        inicio_fila = int(round(0.2 * Long_y))
        final_fila = int(round(0.8 * Long_y))

        for i in range(inicio_fila, final_fila + 1):
            phi[i, electrodo_P] = 1.0
            phi[i, electrodo_N] = -1.0

        return phi

    def gauss_seidel_modified(M, omega, tolerance):
        phi = np.zeros((M + 1, M + 1), dtype=float)
        phi = condiciones_frontera(phi)

        delta = 1.0
        its = 0

        while delta > tolerance:
            its += 1
            phi_prev = phi.copy()

            for i in range(1, M):
                for j in range(1, M):
                    new_val = 0.25 * (
                        phi[i+1, j] + phi[i-1, j] + 
                        phi[i, j+1] + phi[i, j-1]
                    )
                    phi[i, j] = (1 - omega) * phi[i, j] + omega * new_val

            phi = condiciones_frontera(phi)
            delta = np.max(np.abs(phi - phi_prev))

        return phi, its

    gauss_seidel, iterations = gauss_seidel_modified(100, 1.5, 1e-5)
    print(f"Iteraciones: {iterations}")

    plt.figure(figsize=(8, 6))
    plt.imshow(gauss_seidel, origin="lower", cmap='jet')
    plt.colorbar(label="Potencial φ")
    plt.title("Solución Gauss-Seidel de la Ecuación de Laplace")
    plt.xlabel("Posición X")
    plt.ylabel("Posición Y")
    plt.show()
    ```
---

## Implementación del método de sobre relajación de Jacobi (over-relaxation):

Una forma de acelerar la convergencia del método de Jacobi es mediante el método de Jacobi modificado, que incrementa el tamaño del paso en cada iteración mediante un parámetro $\omega$. En lugar de avanzar poco a poco, se ajusta la actualización de $\phi$ para hacer pasos más grandes, lo que puede reducir el número de iteraciones necesarias para aproximar la solución. La actualización se define como:

$\phi'(x, y) = (1+\omega) \cdot \left[\frac{1}{4} \cdot (\phi(x + a, y) + \phi(x - a, y) + \phi(x, y + a) + \phi(x, y - a))\right] - \omega \phi(x, y)$

Sin embargo, esta técnica no siempre es estable; la elección adecuada del parámetro $\omega$ es crucial y suele depender del problema específico para garantizar estabilidad y mejorar la velocidad de convergencia.

#### Causas de error del método de sobre-relajación de Jacobi

$$
\phi'(x, y) = \frac{1}{4} \left[ (\omega + 1) \left( \phi(x + a, y) + \phi(x - a, y) + \phi(x, y + a) + \phi(x, y - a) \right) - \omega\phi(x, y) \right]
$$

Este método incluye un término $\omega$ que ayuda a acelerar la convergencia del mismo. Este método numérico no es útil para resolver el problema del capacitor y se debe a dos factores:

1. Radio espectral de la matriz de iteración $M$  
2. Número de condición de la matriz $A$  

Dado un sistema de la forma $A\mathbf{x} = \mathbf{b}$, donde $A$ es una matriz, $\mathbf{x}$ el vector de incógnitas y $\mathbf{b}$ el vector solución, es posible implementar un método iterativo como Jacobi o Gauss-Seidel. No obstante, estos métodos dependen del radio espectral de su matriz de iteración, de forma que la convergencia ocurre si y sólo si:

$$
\text{Convergencia} \iff \rho(M) < 1
$$

(Donde $\rho(M)$ es una medida de la rapidez para corregir errores y converger).

La matriz $M$ se define como la descomposición de la matriz $A$ del sistema inicial y depende de cada método:

- Jacobi:  
  $M = D^{-1}(L + U)$

- Gauss-Seidel:  
  $M = (D - L)^{-1}U$

- SOR (Relajación Sucesiva):  
  $M = [D - \omega L]^{-1} [(1 - \omega)D + \omega U]$

El radio espectral de $M$ es el máximo valor absoluto de sus valores propios.

Por otra parte, el número de condición de la matriz $A$ mide qué tan bien condicionada está. Si $k(A) \gg 1$, la matriz es sensible a perturbaciones y se dice "mal condicionada", generando errores en la solución. Se define como:

$$
k(A) = \| A \| \cdot \| A^T \|
$$

Un sistema mal condicionado es numéricamente inestable. Si $A$ está mal condicionada, la estructura de $M$ hereda esta dificultad y su radio espectral tiende a ser mayor, impidiendo la convergencia.

Explicación de la no convergencia en nuestro problema:  
El método de Jacobi modificado no converge debido a que la matriz de coeficientes $A$ está mal condicionada, con un número de condición del orden de $10^3$.


???+ note "SOR Jacobi"

    ```python
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
    ```
