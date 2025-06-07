# SmolML - Redes Neuronales: Retropropagación hasta el límite

Bienvenido a la sección de redes neuronales de SmolML! Habiendo establecido nuestros objetos Value para diferenciación automática y MLArray para manejar datos (ver sección 'core'), ahora podemos construir modelos que aprenden. Esta guía te llevará a través de los conceptos fundamentales, desde una sola neurona hasta una red neuronal completamente entrenable, y cómo se representan en SmolML.

> **IMPORTANTE**: Como nuestra implementación está hecha completamente en Python, manejar la diferenciación automática de una red neuronal completa es muy costoso computacionalmente. Si planeas ejecutar un ejemplo, recomendamos empezar con una red muy pequeña y luego escalar. Crear una red neuronal demasiado grande para tu computadora podría hacer que se congele 🙂 

<div align="center">
  <img src="https://github.com/user-attachments/assets/e5315fca-5dd6-4c9c-9cf3-bf46edfbb40c" width="600">
</div>

## La Neurona: Un Pequeño Tomador de Decisiones

En el corazón de una red neuronal está la neurona (o nodo). Piensa en ella como una pequeña unidad computacional que recibe varias entradas, las procesa, y produce una sola salida.

<div align="center">
  <img src="https://github.com/user-attachments/assets/2f95fdfe-1676-4a0b-9e10-95ecdf9155b6" width="600">
</div>

Esto es lo que una neurona hace conceptualmente:

- **Suma Ponderada (Weighted Sum)**: Cada conexión de entrada a la neurona tiene un peso asociado. La neurona multiplica cada valor de entrada por su peso correspondiente. Estos pesos son cruciales – son lo que la red aprende ajustando durante el entrenamiento, determinando la influencia de cada entrada.
- **Sesgo (Bias)**: La neurona luego agrega un término de sesgo a esta suma ponderada. El sesgo permite a la neurona desplazar su salida hacia arriba o abajo, independiente de sus entradas. Esto ayuda a la red a ajustar datos que no necesariamente pasan por el origen.
- **Función de Activación (Activation Function)**: Finalmente, el resultado de la suma ponderada + sesgo se pasa a través de una función de activación. Esta función introduce no linealidad, lo cual es vital. Sin no linealidad, una pila de múltiples capas se comportaría como una sola capa, limitando la habilidad de la red para aprender patrones complejos. Las funciones de activación comunes incluyen ReLU, Tanh, y Sigmoid.

Mientras SmolML no tiene una clase Neurona independiente para esta sección (ya que frecuentemente es más eficiente trabajar con capas directamente), la lógica de muchas de esas neuronas operando en paralelo está encapsulada dentro de nuestra DenseLayer. Cada característica de salida de una DenseLayer puede pensarse como la salida de una neurona conceptual.

## Capas: Organizando Neuronas

Una sola neurona no es muy poderosa por sí sola. Las redes neuronales organizan neuronas en capas. El tipo más común es una Capa Densa (Dense Layer) (también conocida como Capa Completamente Conectada).

¿Qué hace una capa densa?

En una capa densa, cada neurona en la capa recibe entrada de cada neurona en la capa anterior (o de los datos de entrada cruda si es la primera capa).

Conceptualmente, una capa densa realiza dos pasos principales, construyendo sobre la lógica de la neurona:

1. **Transformación Lineal (Linear Transformation)**: Toma un vector de entrada (o un lote de vectores de entrada) y realiza una multiplicación de matrices con una matriz de pesos (`W`) y agrega un vector de sesgo (`b`).
   - Cada fila en el vector de entrada se conecta a cada columna en la matriz de pesos. Si tienes características input_size y quieres características output_size de esta capa (es decir, neuronas output_size conceptuales), la matriz de pesos `W` tendrá una forma de (input_size, output_size). Cada elemento $W_ij$ es el peso conectando la i-ésima característica de entrada a la j-ésima neurona en la capa.
   - El vector de sesgo b tendrá elementos output_size, uno para cada neurona.
   - Matemáticamente: $z=entrada×W+b$.
   - En SmolML, cuando creas una DenseLayer (de `layer.py`), especificas input_size y output_size. La capa luego inicializa self.weights (nuestro `W`) y self.biases (nuestro `b`) como objetos `MLArray`. Estos son los parámetros entrenables de la capa.

```python
# De layer.py
class DenseLayer:
    def __init__(self, input_size: int, output_size: int, ...):
        self.weights = weight_initializer.initialize(input_size, output_size) # MLArray
        self.biases = zeros(1, output_size) # MLArray
        ...
```

2. **Función de Activación (Activation Function)**: El resultado (`z`) de esta transformación lineal se pasa luego elemento por elemento a través de una función de activación no lineal elegida (ej., ReLU, Tanh).
   - Esto se aplica a la salida de cada neurona conceptual en la capa.
   - En SmolML, especificas la activation_function al crear una DenseLayer, y se aplica en el método forward:

```python
# De layer.py
class DenseLayer:
    ...
    def forward(self, input_data):
        z = input_data @ self.weights + self.biases # Transformación lineal
        return self.activation_function(z)      # Activación
```

El método forward esencialmente define cómo fluyen los datos a través de la capa. Porque `input_data`, `self.weights`, y `self.biases` son `MLArray`s (que usan objetos `Value` internamente), todas las operaciones automáticamente construyen el grafo computacional necesario para retropropagación.

## Redes Neuronales: Apilando Capas

El verdadero poder de las redes neuronales viene de apilar múltiples capas. La salida de una capa se convierte en la entrada de la siguiente. Esto permite a la red aprender características jerárquicas – capas anteriores podrían aprender patrones simples (como bordes en una imagen), mientras capas posteriores los combinan para aprender conceptos más complejos (como formas u objetos).

<div align="center">
  <img src="https://github.com/user-attachments/assets/3979a284-0b29-4110-b6c5-dfe1a13f50b9" width="600">
</div>

### La Clase NeuralNetwork (neural_network.py)

En SmolML, la clase `NeuralNetwork` maneja esta secuencia de capas y orquesta todo el proceso de entrenamiento.

- **Inicialización (__init__)**:
  - Creas una NeuralNetwork proporcionándole una lista de objetos de capa (ej., una secuencia de instancias DenseLayer), una loss_function (para medir qué tan "incorrectas" están las predicciones de la red), y un optimizer (que define cómo actualizar los parámetros de la capa).

```python
# De neural_network.py
class NeuralNetwork:
    def __init__(self, layers: list, loss_function: callable, optimizer: optimizers.Optimizer = optimizers.SGD()):
        self.layers = layers
        self.loss_function = loss_function
        self.optimizer = optimizer
```

- **Paso Hacia Adelante (Forward Pass) (forward)**:
  - El paso hacia adelante de la red es directo: toma los datos de entrada y los pasa secuencialmente a través de cada capa en su lista. La salida de la capa i se convierte en la entrada de la capa i+1.

```python
# De neural_network.py
class NeuralNetwork:
    ...
    def forward(self, input_data):
        for layer in self.layers: # Pasar datos a través de cada capa
            input_data = layer.forward(input_data)
        return input_data # Salida final de la red
```

Este paso hacia adelante encadenado, porque cada método forward de capa usa operaciones MLArray, construye un gran grafo computacional desde la entrada inicial hasta la predicción final de la red.

## Enseñando a la Red: El Loop de Entrenamiento

"Aprender" en una red neuronal significa ajustar los pesos y sesgos en todas sus capas para hacer mejores predicciones. Esto se logra a través de un proceso llamado entrenamiento, que típicamente involucra los siguientes pasos repetidos por muchas épocas (pases a través de todo el dataset):

1. **Paso Hacia Adelante (Forward Pass)**:
   - Alimentar los datos de entrada (`X`) a través de la red usando `network.forward(X)` para obtener predicciones (`y_pred`). Como hemos visto, esto también construye el grafo computacional.

2. **Calcular Pérdida (Compute Loss)**:
   - Comparar las predicciones de la red (`y_pred`) con los valores objetivo reales (`y`) usando la loss_function especificada (ej., Error Cuadrático Medio para regresión, Entropía Cruzada para clasificación).
   - La pérdida es un solo Value (frecuentemente envuelto en un `MLArray`) que cuantifica qué tan mal se desempeñó la red en este lote de datos. Este `Value` de pérdida es el nodo final de nuestro grafo computacional actual.

3. **Paso Hacia Atrás (Backward Pass) (Retropropagación)**:
   - ¡Aquí es donde brilla la magia de nuestros objetos `Value` (de la sección 'core')! Llamamos `loss.backward()`.
   - Este comando dispara el proceso de diferenciación automática. Camina hacia atrás a través del grafo computacional completo (desde la pérdida hasta cada peso y sesgo en cada `DenseLayer`, e incluso la entrada `X`) y calcula el gradiente de la pérdida con respecto a cada uno de estos objetos `Value`. El atributo `.grad` de cada `Value` (y por tanto cada elemento en nuestros parámetros `MLArray`) se llena.
   - Esto nos dice cuánto cambiaría un pequeño cambio en cada `weight` o `bias` la pérdida general.

4. **Actualizar Parámetros (Update Parameters)**:
   - Ahora que sabemos la "dirección de mayor ascenso" para la pérdida (los gradientes), el optimizador interviene. Usa estos gradientes (y su propia lógica interna, como una tasa de aprendizaje) para ajustar los pesos y sesgos en cada capa. El objetivo es empujarlos en la dirección opuesta a sus gradientes para reducir la pérdida.
   - En SmolML, el método `NeuralNetwork.train` itera a través de sus capas y llama `layer.update(self.optimizer, ...)` para cada una. Este método, a su vez, usa el optimizador para modificar layer.weights y layer.biases.

5. **Reiniciar Gradientes (Reset Gradients)**:
   - Los gradientes calculados por `loss.backward()` son acumulados (agregados) al atributo `.grad` de cada `Value`. Antes de la siguiente iteración de entrenamiento (el siguiente paso forward/backward), es absolutamente crucial reiniciar estos gradientes de vuelta a cero.
   - Esto se hace usando el método `.restart()` en los `MLArray`s relevantes (todos los pesos y sesgos en cada capa, y a veces X e y si son parte de grafos persistentes). Si no hiciéramos esto, los gradientes de iteraciones previas influenciarían incorrectamente las actualizaciones en la iteración actual.
   - Verás esto en `NeuralNetwork.train()`:

```python
# Dentro de NeuralNetwork.train() después de actualizaciones de parámetros
X.restart()
y.restart()
for layer in self.layers:
    layer.weights.restart()
    layer.biases.restart()
```

Al repetir cíclicamente estos pasos, la NeuralNetwork gradualmente ajusta sus parámetros DenseLayer, aprovechando el poder de diferenciación automática de Value y MLArray para minimizar la pérdida y "aprender" de los datos.