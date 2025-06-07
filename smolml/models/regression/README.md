# SmolML - Regresión: Prediciendo Valores Continuos

Construyendo sobre los conceptos fundamentales de diferenciación automática (`Value`) y arrays N-dimensionales (`MLArray`) explicados en el SmolML Core, ahora podemos implementar varios modelos de machine learning. Esta sección se enfoca en **modelos de regresión (regression models)**, que se usan para predecir salidas numéricas continuas. Piensa en predecir precios de casas, valores de acciones, o temperatura basándose en características de entrada.

Mientras que las redes neuronales profundas ofrecen poder inmenso, modelos más simples como Regresión Lineal (Linear Regression) o su extensión, Regresión Polinomial (Polynomial Regression), son frecuentemente excelentes puntos de partida, computacionalmente eficientes y altamente interpretables. Comparten el mismo principio fundamental de aprendizaje que las redes complejas: minimizar una función de pérdida (loss function) ajustando parámetros usando descenso de gradiente (gradient descent), todo potenciado por nuestro motor de diferenciación automática usando la clase `Value`.

## Fundamentos de Regresión: Aprendiendo de los Datos

El objetivo en regresión es encontrar una función matemática que mapee características de entrada (como los pies cuadrados de una casa) a una salida continua (como su precio). Esta función tiene parámetros internos (frecuentemente llamados **pesos (weights)** o coeficientes, y un **sesgo (bias)** o intercepto) que determinan su forma exacta.

<div align="center">
  <img src="https://github.com/user-attachments/assets/79874cec-8650-4628-af1f-ca6fdc4debe5" width="600">
</div>

¿Cómo encontramos los *mejores* parámetros?
1.  **Predicción:** Empezamos con parámetros iniciales (frecuentemente aleatorios) y usamos el modelo para hacer predicciones en nuestros datos de entrenamiento.
2.  **Cálculo de Pérdida:** Comparamos estas predicciones con los valores reales conocidos usando una **función de pérdida (loss function)** (como Error Cuadrático Medio - MSE). Esta función cuantifica *qué tan equivocado* está el modelo actualmente. Una pérdida menor es mejor.
3.  **Cálculo de Gradiente:** Así como en la explicación del core, necesitamos saber cómo ajustar cada parámetro para reducir la pérdida. Nuestros objetos `Value` y el concepto de **retropropagación (backpropagation)** automáticamente calculan el **gradiente** de la pérdida con respecto a cada parámetro (pesos y sesgo). Recuerda, el gradiente apunta hacia el mayor *aumento* de pérdida.
4.  **Actualización de Parámetros:** Usamos un **optimizador (optimizer)** (como Descenso de Gradiente Estocástico - SGD) para empujar los parámetros en la dirección *opuesta* a sus gradientes, dando un pequeño paso hacia menor pérdida.
5.  **Iteración:** Repetimos los pasos 1-4 muchas veces (iteraciones o épocas), gradualmente mejorando los parámetros del modelo hasta que la pérdida se minimiza o deja de decrecer significativamente.

Este proceso iterativo permite al modelo de regresión "aprender" la relación subyacente entre las entradas y salidas de los datos.

## La Clase Base `Regression`: Un Framework Común

Para optimizar la implementación de diferentes algoritmos de regresión, en SmolML hicimos una clase base `Regression` (en `regression.py`). Esta clase maneja la estructura común y la lógica del loop de entrenamiento. Modelos específicos como `LinearRegression` heredan de ella.

Así es como funciona:

* **Inicialización (`__init__`)**:
    * Acepta el `input_size` (número de características de entrada esperadas), una `loss_function`, una instancia de `optimizer`, y un `initializer` de pesos.
    * Crucialmente, inicializa los **parámetros entrenables** del modelo:
        * `self.weights`: Un `MLArray` que contiene los coeficientes para cada característica de entrada. Su forma está determinada por `input_size`, y los valores son establecidos por el `initializer`.
        * `self.bias`: Un `MLArray` escalar (inicializado a 1) representando el término de intercepto.
    * Porque `weights` y `bias` son `MLArray`s, inherentemente contienen objetos `Value`. Esto asegura que sean parte del grafo computacional y sus gradientes puedan ser automáticamente calculados durante el entrenamiento.

* **Entrenamiento (`fit`)**:
    * Este método orquesta el loop de descenso de gradiente descrito anteriormente. Para un número especificado de `iterations`:
        1.  **Paso Hacia Adelante (Forward Pass):** Llama `self.predict(X)` (que debe ser implementado por la subclase) para obtener predicciones `y_pred`. Esto construye el grafo computacional para el paso de predicción.
        2.  **Cálculo de Pérdida:** Calcula `loss = self.loss_function(y, y_pred)`. Esta `loss` es el `MLArray` final (usualmente conteniendo un solo `Value`) representando el error general para esta iteración.
        3.  **Paso Hacia Atrás (Backward Pass):** Invoca `loss.backward()`. Esto dispara el proceso de diferenciación automática, calculando los gradientes de la pérdida con respecto a todos los objetos `Value` involucrados, incluyendo aquellos dentro de `self.weights` y `self.bias`.
        4.  **Actualización de Parámetros:** Usa `self.optimizer.update(...)` para ajustar `self.weights` y `self.bias` basándose en sus gradientes calculados (`weights.grad()` y `bias.grad()`) y la lógica del optimizador (ej., tasa de aprendizaje).
        5.  **Reinicio de Gradientes:** Llama `self.restart(X, y)` para poner a cero todos los gradientes (atributos `.grad` de los objetos `Value`) en los parámetros y datos, preparando para la siguiente iteración.

* **Predicción (`predict`)**:
    * Definido en la clase base pero lanza `NotImplementedError`. ¿Por qué? Porque la lógica central de *cómo* hacer una predicción difiere entre tipos de regresión (ej., lineal vs. polinomial). Cada subclase *debe* proporcionar su propio método `predict` definiendo su fórmula matemática específica usando operaciones `MLArray`.

* **Reinicio de Gradientes (`restart`)**:
    * Un auxiliar que simplemente llama el método `.restart()` en los `MLArray`s de `weights`, `bias`, entrada `X`, y objetivo `y`. Esto eficientemente reinicia el atributo `.grad` de todos los objetos `Value` subyacentes a cero.

* **Representación (`__repr__`)**:
    * Proporciona un resumen en cadena bien formateado del modelo configurado, incluyendo su tipo, formas de parámetros, optimizador, función de pérdida y uso estimado de memoria.

## Modelos Específicos Implementados

<div align="center">
  <img src="https://github.com/user-attachments/assets/8b282ca1-7c17-460d-a64c-61b0624627f9" width="600">
</div>

### `LinearRegression`

Este es el modelo de regresión más fundamental. Asume una relación lineal directa entre las características de entrada `X` y la salida `y`. El objetivo es encontrar los mejores pesos `w` y sesgo `b` tal que $y \approx Xw + b$.

* **Implementación (`regression.py`)**:
    * Hereda directamente de `Regression`.
    * Su contribución principal es sobrescribir el método `predict`.
* **Predicción (`predict`)**:
    * Implementa la ecuación lineal: `return X @ self.weights + self.bias`.
    * Toma la entrada `X` (`MLArray`), realiza multiplicación de matrices (`@`) con `self.weights` (`MLArray`), y suma `self.bias` (`MLArray`). Porque `X`, `weights`, y `bias` son todos `MLArray`s conteniendo objetos `Value`, esta línea de código automáticamente construye el grafo computacional necesario para retropropagación.
* **Entrenamiento**:
    * Usa el método `fit` heredado directamente de la clase base `Regression` sin modificación. La clase base maneja todo el loop de entrenamiento usando la lógica `predict` proporcionada por `LinearRegression`.

### `PolynomialRegression`

¿Qué pasa si la relación no es una línea recta? Regresión Polinomial extiende regresión lineal ajustando una curva polinomial (ej., $y \approx w_2 x^2 + w_1 x + b$) a los datos.

* **Implementación (`regression.py`)**:
    * También hereda de `Regression`.
* **La Idea Central**: En lugar de ajustar directamente `X` a `y`, primero *transforma* las características de entrada `X` en características polinomiales (ej., agregando $X^2$, $X^3$, etc.) y luego aplica un modelo de *regresión lineal* estándar a estas *nuevas características transformadas*.
* **Inicialización (`__init__`)**:
    * Toma un argumento adicional `degree`, especificando la potencia más alta a incluir en la transformación de características (ej., `degree=2` significa incluir $X$ y $X^2$).
    * Llama el `__init__` de la clase base, pero el `input_size` pasado a la clase base es efectivamente el número de *características polinomiales*, no el número original de características. Los pesos corresponderán a estas características transformadas.
* **Transformación de Características (`transform_features`)**:
    * Este método crucial toma la entrada original `X` y genera las nuevas características polinomiales. Para una entrada `X` y `degree=d`, calcula $X, X^2, \dots, X^d$ usando operaciones `MLArray` (como multiplicación elemento por elemento `*`) y las concatena en un nuevo `MLArray`. Esto asegura que la transformación también sea potencialmente parte del grafo si es necesario (aunque frecuentemente se pre-calcula).
* **Predicción (`predict`)**:
    1.  Primero llama `X_poly = self.transform_features(X)` para obtener las características polinomiales.
    2.  Luego, realiza una predicción lineal estándar usando estas características transformadas: `return X_poly @ self.weights + self.bias`. Los `self.weights` aquí corresponden a los coeficientes de los términos polinomiales.
* **Entrenamiento (`fit`)**:
    * Sobrescribe ligeramente el método `fit` de la clase base.
    1.  Antes del loop principal, transforma toda la entrada de entrenamiento `X` en `X_poly = self.transform_features(X)`.
    2.  Luego llama el método `fit` de la *clase base* (`super().fit(...)`) pero pasa `X_poly` (en lugar de `X`) como los datos de entrada.
    * El método `fit` heredado luego procede como de costumbre, calculando pérdida basándose en las predicciones de `X_poly`, retropropagando gradientes a través de la parte de predicción lineal *y* el paso de transformación de características, y actualizando los pesos asociados con los términos polinomiales.

## Ejemplo de Uso

Aquí hay un ejemplo conceptual de cómo podrías usar `LinearRegression`:

```python
from smolml.models.regression import LinearRegression
from smolml.core.ml_array import MLArray
import smolml.utils.optimizers as optimizers
import smolml.utils.losses as losses

# Datos de Muestra (ej., 2 características, 3 muestras)
X_data = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
# Valores objetivo (continuos)
y_data = [[3.5], [5.5], [7.5]]

# Convertir a MLArray
X = MLArray(X_data)
y = MLArray(y_data)

# Inicializar el modelo
# Espera 2 características de entrada
model = LinearRegression(input_size=2,
                         optimizer=optimizers.SGD(learning_rate=0.01),
                         loss_function=losses.mse_loss)

# Imprimir resumen inicial del modelo
print(model)

# Entrenar el modelo
print("\nIniciando entrenamiento...")
losses_history = model.fit(X, y, iterations=100, verbose=True, print_every=10)
print("Entrenamiento completo.")

# Imprimir resumen final del modelo (pesos/sesgo habrán cambiado)
print(model)

# Hacer predicciones en nuevos datos
X_new = MLArray([[4.0, 5.0]])
prediction = model.predict(X_new)
print(f"\nPredicción para {X_new.to_list()}: {prediction.to_list()}")
```

## Resumen de Regresión

¡Estas clases de regresión muestran cómo el `Value` y `MLArray` fundamentales que implementamos pueden usarse para diseñar y entrenar modelos clásicos de machine learning! ¡En solo unas pocas líneas de código! ¿No es genial?