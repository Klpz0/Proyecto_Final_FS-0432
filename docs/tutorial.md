# Tutorial: Métodos de relajación

---

## Implementación del metodo de relajación de Jacobi:

???+ note "Método de Jacobi"
    
    Este método usa una actualización iterativa para resolver la ecuación de Laplace en 2D.

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

???+ note "Método de Gauss-Seidel"
    
    Este método mejora la convergencia actualizando los valores inmediatamente durante la iteración.

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
    `````
---

## Implementación del método de sobre relajación de Jacobi (over-relaxation)

???+ note "SOR Jacobi"

    ```python
    """Solución numérica de la ecuación de Poisson usando el método de Jacobi-SOR."""

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
    ```
