# SmolML - Core: Diferenciación Automática y Arrays N-Dimensionales

Bienvenido al núcleo de SmolML! Aquí es donde comienza la magia si quieres entender cómo los modelos de machine learning, especialmente las redes neuronales, aprenden de los datos. Vamos a desglosar dos conceptos fundamentales: cómo las computadoras *calculan* la dirección para el aprendizaje (diferenciación automática) y cómo manejamos los datos multi-dimensionales involucrados.

Esta parte de SmolML se enfoca en manejar estos datos y calcular gradientes automáticamente. ¿Por qué gradientes? ¿Por qué automático? Vamos a profundizar.

## ¿Por qué Necesitamos Gradientes de Todos Modos?

Piensa en enseñar a una computadora a reconocer un gato en una foto. La computadora hace una predicción basándose en su configuración interna actual (parámetros o pesos). Inicialmente, estas configuraciones son aleatorias, así que la predicción probablemente esté mal. Medimos *qué tan mal* usando una "función de pérdida" (las tenemos en `smolml\utils\losses.py`, explicadas en otra sección) – una pérdida menor significa una mejor predicción.

El objetivo es ajustar los parámetros de la computadora para *minimizar* esta pérdida. Pero ¿cómo sabemos *hacia qué dirección* ajustar cada parámetro? ¿Deberíamos aumentarlo? ¿Disminuirlo? ¿En qué cantidad?

![Figure-3-37-Gradient-descent-Algorithm-illustration(1)](https://github.com/user-attachments/assets/93e2df5b-5f02-43d1-a4b9-3e9daeb81a9a)

Aquí es donde entran los **gradientes**. El gradiente de la función de pérdida con respecto a un parámetro específico nos dice la "pendiente" de la pérdida en el valor actual de ese parámetro. Apunta en la dirección del *mayor aumento* de la pérdida. Entonces, si queremos *disminuir* la pérdida, empujamos el parámetro en la dirección *opuesta* al gradiente (la flecha púrpura en la imagen de arriba). El tamaño del gradiente también nos dice qué tan sensible es la pérdida a ese parámetro – un gradiente más grande sugiere que podría necesitarse un ajuste mayor.

Calcular estos gradientes para cada parámetro permite al modelo mejorar iterativamente, paso a paso, reduciendo su error. Este proceso es el corazón del entrenamiento de la mayoría de modelos de ML.

## ¿Por qué Diferenciación "Automática"?

Bien, entonces necesitamos gradientes. Para una función simple como $y = a \times b + c$, podemos encontrar los gradientes ($\frac{\partial y}{\partial a}$, $\frac{\partial y}{\partial b}$, $\frac{\partial y}{\partial c}$) usando reglas básicas de cálculo (como la regla de la cadena).

Pero las redes neuronales modernas son *vastamente* más complejas. Son esencialmente funciones matemáticas gigantes y anidadas con potencialmente millones de parámetros. Calcular todos esos gradientes manualmente es prácticamente imposible e increíblemente propenso a errores.

**La Diferenciación Automática (AutoDiff)** es la solución. Es una técnica donde la computadora misma lleva registro de cada operación matemática realizada, construyendo un grafo computacional. Luego, aplicando la regla de la cadena sistemáticamente hacia atrás a través de este grafo (un proceso llamado **retropropagación**), puede calcular eficientemente el gradiente de la salida final (la pérdida) con respecto a cada entrada y parámetro involucrado.

<div align="center">
  <img src="https://github.com/user-attachments/assets/75372083-69b3-47b4-959d-609d7f426751" width="600">
</div>

## Implementando AutoDiff con `Value`

Esta librería usa la clase `Value` para implementar Diferenciación Automática.

*(Si quieres una inmersión profunda, el concepto de retropropagación y diferenciación automática está muy bien explicado en este [video de Andrej Karpathy](https://www.youtube.com/watch?v=VMj-3S1tku0), ¡muy recomendado!)*

**¿Qué es un `Value`?**

Piensa en un objeto `Value` como un contenedor inteligente para un solo número (un escalar). No solo contiene el número; se prepara para cálculos de gradientes. Almacena:
1.  `data`: El valor numérico actual (ej., 5.0, -3.2).
2.  `grad`: El gradiente de la *salida final* de todo nuestro grafo computacional con respecto a los *datos* específicos de este `Value`. Empieza en 0 y se llena durante la retropropagación.

**Construyendo el Rastro de Cálculo (Grafo Computacional)**

La parte inteligente sucede cuando realizas operaciones matemáticas (como `+`, `*`, `exp`, `tanh`, etc.) usando objetos `Value`. Cada vez que combinas objetos `Value`, creas un *nuevo* objeto `Value`. Este nuevo objeto internamente recuerda:
* Sus propios `data` (el resultado de la operación).
* La operación que lo creó (`_op`).
* Los objetos `Value` originales que fueron sus entradas (`_children` o `_prev`).

Esto implícitamente construye un grafo computacional, paso a paso, rastreando el linaje del cálculo desde las entradas iniciales hasta el resultado final. Por ejemplo:

```python
a = Value(2.0)
b = Value(3.0)
c = Value(4.0)
# d 'sabe' que resultó de a * b
d = a * b  # d._op = "*", d._prev = {a, b}
# y 'sabe' que resultó de d + c
y = d + c  # y._op = "+", y._prev = {d, c}
```

**Retropropagación: Calculando Gradientes Automáticamente**

Entonces, ¿cómo obtenemos los gradientes sin hacer el cálculo nosotros mismos? Llamando al método `.backward()` en el objeto `Value` final (el que representa nuestra pérdida general, como y en el ejemplo simple).

Aquí está el proceso conceptualmente:

1. Empezar al final (`y`). El gradiente de `y` con respecto a sí mismo es 1. Así que, establecemos `y.grad = 1`.

2. `y` sabe que vino de `d + c`. Usando la regla de la cadena para suma ($\frac{\partial y}{\partial d} = 1$, $\frac{\partial y}{\partial c} = 1$), pasa su gradiente hacia atrás. Le dice a `d` que agregue `1 * y.grad` a su gradiente, y le dice a `c` que agregue `1 * y.grad` a su gradiente. Así, `d.grad` se convierte en 1 y `c.grad` se convierte en 1.

3. `d` sabe que vino de `a * b`. Usando la regla de la cadena para multiplicación ($\frac{\partial d}{\partial a} = b$, $\frac{\partial d}{\partial b} = a$), pasa su gradiente recibido (`d.grad`) hacia atrás. Le dice a `a` que agregue `b.data * d.grad` a su gradiente, y le dice a `b` que agregue `a.data * d.grad` a su gradiente. Así `a.grad` se convierte en $3.0 \times 1 = 3.0$ y `b.grad` se convierte en $2.0 \times 1 = 2.0$.

4. Este proceso continúa recursivamente hacia atrás a través del grafo hasta que todos los objetos `Value` que contribuyeron al resultado final tienen sus gradientes calculados.

Cada objeto `Value` almacena una pequeña función (`_backward`) que sabe cómo calcular los gradientes locales para la operación que representa (+, *, tanh, etc.). El método `.backward()` orquesta llamar a estas pequeñas funciones en el orden inverso correcto (usando un ordenamiento topológico) para implementar la regla de la cadena a través de todo el grafo.

Después de llamar `.backward()`, `a.grad`, `b.grad`, y `c.grad` contienen los gradientes, diciéndote exactamente qué tan sensible es la salida final `y` a cambios en las entradas iniciales `a`, `b`, y `c`.

## Manejando Arrays N-Dimensionales de Values con `MLArray`

> **NOTA IMPORTANTE**: `MLArray` es por mucho la clase más compleja de la librería. Si estás implementando esta librería por ti mismo mientras sigues el tutorial, recomiendo simplemente copiar el `mlarray.py` proporcionado por ahora, y hacer tu propia implementación al final, ya que hacerlo desde cero probablemente te tome mucho tiempo (¡y dolores de cabeza!).

Mientras que los objetos `Value` manejan el AutoDiff núcleo para números individuales, el machine learning prospera con vectores, matrices y tensores de dimensiones superiores. Necesitamos una manera eficiente de manejar colecciones de estos objetos Value inteligentes. Este es el trabajo de la clase MLArray.

¿Qué es un `MLArray`?

Piensa en `MLArray` como un array N-dimensional (como arrays de NumPy, pero construido desde cero aquí). Puede ser una lista (1D), una lista-de-listas (matriz 2D), o anidado más profundamente para dimensiones superiores.

<div align="center">
  <img src="https://github.com/user-attachments/assets/29606bb9-fa55-457c-b2ac-120596aebc11" width="600">
</div>

La diferencia crucial de una lista estándar de Python es que cada número que pongas en un MLArray se convierte automáticamente en un objeto `Value`. Esto sucede recursivamente en el método `_process_data` durante la inicialización.

```python
# Crea un MLArray 2x2; 1.0, 2.0, 3.0, 4.0 ahora son objetos Value
my_matrix = MLArray([[1.0, 2.0], [3.0, 4.0]])
```

Esto asegura que cualquier cálculo realizado usando el `MLArray` automáticamente construirá el grafo computacional necesario para la retropropagación.

**Operaciones en Arrays**

La idea de `MLArray` es soportar muchas operaciones numéricas estándar, diseñadas para trabajar sin problemas con los objetos `Value` subyacentes:

- **Operaciones elemento por elemento**: `+`, `-`, `*`, `/`, `**` (potencia), `exp()`, `log()`, `tanh()`, etc.. Cuando sumas dos `MLArray`s, la librería itera a través de elementos correspondientes y realiza la operación `Value.__add__` (o la operación correspondiente, como `__mul__` para multiplicación, y así sucesivamente) en cada par (manejado por `_element_wise_operation`). Esto construye el grafo elemento por elemento.
- **Multiplicación de Matrices**: `matmul()` o el operador `@`. Esto realiza la lógica del producto punto, combinando correctamente las multiplicaciones y sumas de `Value` subyacentes para construir la estructura de grafo apropiada.
- **Reducciones**: `sum()`, `mean()`, `max()`, `min()`, `std()`. Estas operaciones agregan objetos `Value` a través del array o a lo largo de ejes especificados, nuevamente construyendo el grafo correctamente.
- **Manipulación de Forma**: `transpose()` (o `.T`), `reshape()`. Estas reorganizan los objetos `Value` en diferentes formas de array sin cambiar los objetos mismos, preservando las conexiones del grafo.
- **Broadcasting**: Las operaciones entre arrays de formas diferentes pero compatibles (ej., agregar un vector a cada fila de una matriz) se manejan automáticamente, determinando cómo aplicar operaciones elemento por elemento basándose en reglas de broadcasting (`_broadcast_shapes`, `_broadcast_and_apply`).

**Obteniendo Gradientes para Arrays**

Así como con un solo `Value`, puedes realizar una secuencia compleja de operaciones `MLArray`. Típicamente, tu resultado final será un `Value` escalar que representa la pérdida (a menudo logrado usando `.sum()` o `.mean()` al final).

Luego llamas `.backward()` en este `MLArray` escalar final (o el `Value` dentro de él). Esto dispara el proceso de retropropagación a través de todos los objetos `Value` interconectados que fueron parte del cálculo.

Después de que `.backward()` haya corrido, puedes acceder a los gradientes calculados para cualquier `MLArray` involucrado en el cómputo original usando su método `.grad()`. Esto retorna un nuevo `MLArray` de la misma forma, donde cada elemento contiene el gradiente correspondiente al objeto `Value` original.

```python
# Ejemplo usando ml_array.py y value.py

# Datos de entrada (ej., lote de 2 muestras, 2 características cada una)
x = MLArray([[1.0, 2.0], [3.0, 4.0]])
# Pesos (ej., capa lineal simple mapeando 2 características a 1 salida)
w = MLArray([[0.5], [-0.5]])

# --- Paso Hacia Adelante (Construye el grafo) ---
# Multiplicación de matrices
z = x @ w
# Aplicar función de activación (elemento por elemento)
a = z.tanh()
# Calcular pérdida total (ej., suma de activaciones)
L = a.sum() # L ahora es un MLArray escalar conteniendo un Value

# --- Paso Hacia Atrás (Calcula gradientes) ---
L.backward() # Dispara Value.backward() a través del grafo

# --- Inspeccionar Gradientes ---
# Gradiente de L w.r.t. cada elemento en x
print("Gradiente w.r.t. x:")
print(x.grad())
# Gradiente de L w.r.t. cada elemento en w
print("\nGradiente w.r.t. w:")
print(w.grad())
```

**Funciones de Utilidad y Propiedades**

La librería también incluye ayudantes:

- `zeros()`, `ones()`, `randn()`: Crear `MLArray`s llenos de ceros, unos, o números aleatorios estándar.
- `xavier_uniform()`, `xavier_normal()`: Implementan estrategias comunes para inicializar matrices de pesos en redes neuronales.
- `to_list()`: Convertir un `MLArray` de vuelta a una lista estándar de Python (esto extrae el `.data` y descarta información de gradientes).
- `shape`: Una propiedad para obtener rápidamente la tupla de dimensiones del `MLArray`.
- `size()`: Retorna el número total de elementos en el `MLArray`.

## Juntándolo Todo

Con estos dos componentes núcleo:

- Una clase `Value` que encapsula un solo número y habilita diferenciación automática rastreando operaciones y aplicando la regla de la cadena vía retropropagación.
- Una clase `MLArray` que organiza estos objetos `Value` en arrays N-dimensionales y extiende operaciones matemáticas (como multiplicación de matrices, broadcasting) para trabajar sin problemas dentro del framework de AutoDiff.

Tenemos el kit de herramientas esencial para construir y, más importante, entrenar varios modelos de machine learning, como redes neuronales simples, completamente desde cero. ¡Ahora podemos definir capas de red, calcular funciones de pérdida, y usar los gradientes calculados para actualizar parámetros y hacer que nuestros modelos aprendan!