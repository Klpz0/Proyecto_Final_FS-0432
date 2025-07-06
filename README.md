# Métodos de paralelisación en memoria compartida y distribuida
## Paralelización del Método de Gauss-Seidel en memoria compartida: Red-Black

Para paralelizar el método iterativo de Gauss-Seidel, se implementó la técnica *Red-Black*, una estrategia común en la solución de ecuaciones diferenciales parciales sobre mallas discretas, ideal para la ecuación de Laplace utilizada en este proyecto.

Esta técnica divide los puntos de la grilla en dos subconjuntos intercalados —similar a un tablero de ajedrez— donde cada punto "rojo" solo tiene vecinos "negros" y viceversa. Esto permite actualizar simultáneamente todos los puntos de un mismo color sin dependencias internas.

Cada punto $(i,j)$ se asigna un color según la condición:  
- Rojo si $(i + j) \bmod 2 = 0$
- Negro si $(i + j) \bmod 2 = 1$

El algoritmo se ejecuta en dos pasos por iteración: primero se actualizan todos los puntos rojos usando los valores actuales de sus vecinos negros; luego se actualizan los puntos negros usando los valores recién calculados de los puntos rojos. Así, cada subconjunto se puede procesar en paralelo.

La paralelización se logra asignando a múltiples hilos la actualización de los puntos de un mismo color, lo que en entornos multicore o GPU mejora considerablemente el rendimiento. Para maximizar la eficiencia, se deben cuidar aspectos como el balance de carga, el acceso concurrente a memoria y la sincronización entre fases.

## Posible estrategia de paralelización en memoria distribuida

-

# Métodos numéricos utilizados en la elaboración del proyecto:

En la resolución del problema se hizo uso de métodos iterativos, junto con el modelo de diferencias finitas para aproximar la ecuación de Laplace sobre una malla discreta. Esta aproximación permite obtener una solución numérica al problema de potencial eléctrico en un dominio bidimensional, aplicando condiciones de frontera fijas y cargando valores específicos en ciertas regiones.

## Método de relajación de Jacobi:

El método de relajación de Jacobi es un proceso iterativo utilizado para resolver sistemas de ecuaciones lineales, especialmente los que provienen de la discretización de ecuaciones diferenciales, como la de Laplace. En una grilla discreta de tamaño n×nn×n, cada punto interior de la malla se actualiza en función del promedio de sus vecinos, manteniendo fijos los valores de las fronteras según las condiciones del problema. Matemáticamente se ve de la siguiente manera:

$\phi_{(x,y)} = \frac{1}{4} \cdot (\phi_{(x + a,y)} + \phi_{(x - a,y)} + \phi_{(x,y + a)} + \phi_{(x,y - a)})$

En cada iteración se calcula una nueva matriz con los valores actualizados, sin modificar la matriz original hasta completar todo el barrido. Este enfoque facilita la implementación, ya que no hay dependencias entre los cálculos de los distintos puntos en una misma iteración, sin embargo resulta costoso computacionalmente. El proceso se detiene una vez se llega a la variación entre valores de los puntos de la grilla esté por de bajo de la tolerancia establecida en los parámetros iniciales establecidos.

## Método de relajación de Gauss-Seidel:

El método de Gauss-Seidel es una optimización del método iterativo de Jacobi que permite una convergencia más rápida al actualizar los valores directamente en la misma memoria durante cada iteración. A diferencia de Jacobi, que calcula los nuevos valores sin modificar los anteriores hasta completar toda la grilla, Gauss-Seidel reutiliza inmediatamente los valores recién actualizados, lo que reduce el número de iteraciones necesarias para alcanzar la convergencia. El esquema básico de actualización para resolver la ecuación de Laplace por diferencias finitas centrales es:

$\phi_{(x,y)} = \frac{1}{4} \cdot (\phi_{(x + a,y)} + \phi_{(x - a,y)} + \phi_{(x,y + a)} + \phi_{(x,y - a)})$

En este método, los valores actualizados se escriben directamente sobre la matriz original, lo que introduce una dependencia entre los cálculos dentro de la misma iteración, dificultando su paralelización directa. Sin embargo, este enfoque reduce el uso de memoria y tiende a converger más rápidamente que Jacobi.

## Método de sobre-relajación de Jacobi:

Una forma de acelerar la convergencia del método de Jacobi es mediante el método de Jacobi modificado, que incrementa el tamaño del paso en cada iteración mediante un parámetro $\omega$. En lugar de avanzar poco a poco, se ajusta la actualización de $\phi$ para hacer pasos más grandes, lo que puede reducir el número de iteraciones necesarias para aproximar la solución. La actualización se define como:

$\phi'(x, y) = (1+\omega) \cdot \left[\frac{1}{4} \cdot (\phi(x + a, y) + \phi(x - a, y) + \phi(x, y + a) + \phi(x, y - a))\right] - \omega \phi(x, y)$

Sin embargo, esta técnica no siempre es estable; la elección adecuada del parámetro ωω es crucial y suele depender del problema específico para garantizar estabilidad y mejorar la velocidad de convergencia.

#### Causas de error del método de sobre-relajación de Jacobi
