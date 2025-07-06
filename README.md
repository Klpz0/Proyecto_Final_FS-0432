## Paralelización del Método de Gauss-Seidel mediante Red-Black

Para paralelizar el método iterativo de Gauss-Seidel, se implementó la técnica *Red-Black*, una estrategia común en la solución de ecuaciones diferenciales parciales sobre mallas discretas, ideal para la ecuación de Laplace utilizada en este proyecto.

Esta técnica divide los puntos de la grilla en dos subconjuntos intercalados —similar a un tablero de ajedrez— donde cada punto "rojo" solo tiene vecinos "negros" y viceversa. Esto permite actualizar simultáneamente todos los puntos de un mismo color sin dependencias internas.

Cada punto \((i,j)\) se asigna un color según la condición:  
- Rojo si \((i + j) \bmod 2 = 0\)  
- Negro si \((i + j) \bmod 2 = 1\)

El algoritmo se ejecuta en dos pasos por iteración: primero se actualizan todos los puntos rojos usando los valores actuales de sus vecinos negros; luego se actualizan los puntos negros usando los valores recién calculados de los puntos rojos. Así, cada subconjunto se puede procesar en paralelo.

La paralelización se logra asignando a múltiples hilos la actualización de los puntos de un mismo color, lo que en entornos multicore o GPU mejora considerablemente el rendimiento. Para maximizar la eficiencia, se deben cuidar aspectos como el balance de carga, el acceso concurrente a memoria y la sincronización entre fases.

