# SmolML - ¡El cuarto de utilidades!

¡Bienvenido a los componentes de utilidad de SmolML! Este directorio alberga los bloques de construcción de apoyo requeridos para construir, entrenar y analizar modelos de machine learning dentro de nuestro framework. Piensa en estos módulos como las herramientas de asistencia e ingredientes que usarás repetidamente.

Todo aquí aprovecha el `MLArray` para manejar datos numéricos y diferenciación automática (donde aplique).

## Estructura del Directorio

```
.
├── activation.py       # Funciones no lineales para redes neuronales
├── initializers.py     # Estrategias para establecer pesos iniciales del modelo
├── losses.py           # Funciones para medir error del modelo
├── memory.py           # Utilidades para calcular uso de memoria
└── optimizers.py       # Algoritmos para actualizar pesos del modelo durante entrenamiento
```

---

## Funciones de Activación (`activation.py`)

**¿Por qué las necesitamos?**

Imagina construir una red neuronal. Si solo apilas operaciones lineales (como multiplicaciones de matrices y sumas), toda la red, sin importar qué tan profunda sea, se comporta como una sola transformación *lineal*. Esto limita severamente la habilidad de la red para aprender patrones complejos y no lineales que se encuentran frecuentemente en datos del mundo real (como reconocimiento de imágenes, traducción de idiomas, etc.).

Las **funciones de activación** introducen **no linealidad** en la red, típicamente aplicadas elemento por elemento después de una transformación lineal en una capa. Esto permite a la red aproximar funciones mucho más complicadas.

**Cómo funcionan (generalmente):**

Cada función de activación toma una entrada numérica (frecuentemente la salida de una capa lineal) y le aplica una transformación matemática específica. La mayoría de funciones proporcionadas aquí operan elemento por elemento en un `MLArray`.

**Funciones de Activación Clave Proporcionadas:**

* **`relu(x)` (Unidad Lineal Rectificada):**
    * *Concepto:* Produce la entrada si es positiva, de otro modo produce cero ($f(x) = \max(0, x)$).
    * *Por qué:* Computacionalmente muy eficiente, ayuda a mitigar gradientes que se desvanecen, y es la elección más común para capas ocultas en redes profundas.
    * *Código:* Usa `_element_wise_activation` con `val.relu()`.

* **`leaky_relu(x, alpha=0.01)`:**
    * *Concepto:* Como ReLU, pero permite un gradiente pequeño y no cero para entradas negativas ($f(x) = x$ si $x > 0$, sino $f(x) = \alpha x$).
    * *Por qué:* Intenta arreglar el problema de "ReLU moribundo" donde las neuronas pueden volverse inactivas si consistentemente producen valores negativos.
    * *Código:* Usa `_element_wise_activation` con una lambda personalizada verificando el valor.

* **`elu(x, alpha=1.0)` (Unidad Lineal Exponencial):**
    * *Concepto:* Similar a Leaky ReLU pero usa una curva exponencial para entradas negativas ($f(x) = x$ si $x > 0$, sino $f(x) = \alpha (e^x - 1)$).
    * *Por qué:* Busca tener salidas negativas más cercanas a -1 en promedio, potencialmente acelerando el aprendizaje. Más suave que ReLU/Leaky ReLU.
    * *Código:* Usa `_element_wise_activation` con una lambda personalizada.

* **`sigmoid(x)`:**
    * *Concepto:* Comprime valores de entrada al rango (0, 1) usando la fórmula $f(x) = \frac{1}{1 + e^{-x}}$.
    * *Por qué:* Históricamente popular, frecuentemente usada en la capa de salida para problemas de **clasificación binaria** para interpretar la salida como una probabilidad. Puede sufrir de gradientes que se desvanecen en redes profundas.
    * *Código:* Usa `_element_wise_activation` con la fórmula sigmoid.

* **`softmax(x, axis=-1)`:**
    * *Concepto:* Transforma un vector de números en una distribución de probabilidad (los valores son no negativos y suman 1). Exponencia las entradas y luego las normaliza.
    * *Por qué:* Esencial para la capa de salida en problemas de **clasificación multi-clase**. Cada nodo de salida representa la probabilidad de que la entrada pertenezca a una clase específica. Nota el argumento `axis` determina *a lo largo de qué dimensión* ocurre la normalización.
    * *Código:* Maneja escalares, 1D, y `MLArray`s multi-dimensionales, aplicando la lógica softmax recursivamente a lo largo del `axis` especificado. Incluye mejoras de estabilidad numérica (sustrayendo el valor máximo antes de la exponenciación).

* **`tanh(x)` (Tangente Hiperbólica):**
    * *Concepto:* Comprime valores de entrada al rango (-1, 1) ($f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$).
    * *Por qué:* Casos de uso similares a sigmoid pero el rango de salida centrado en cero a veces puede ser beneficioso. También susceptible a gradientes que se desvanecen.
    * *Código:* Usa `_element_wise_activation` con `val.tanh()`.

* **`linear(x)`:**
    * *Concepto:* Simplemente retorna la entrada sin cambios ($f(x) = x$).
    * *Por qué:* Usada cuando no se necesita no linealidad, por ejemplo, en la capa de salida de un modelo de regresión.

*(Ver `activation.py` para detalles de implementación)*

<div align="center">
  <img src="https://github.com/user-attachments/assets/c610f284-dbf2-4a69-8f88-5433a28276cb" width="600">
</div>

---

## Inicializadores de Pesos (`initializers.py`)

**¿Por qué es importante la inicialización?**

Cuando creas una capa de red neuronal, sus pesos y sesgos necesitan valores iniciales. Elegir estos valores iniciales mal puede obstaculizar drásticamente el entrenamiento:
* **Muy pequeños:** Los gradientes podrían volverse diminutos mientras se propagan hacia atrás (gradientes que se desvanecen), haciendo el aprendizaje extremadamente lento o imposible.
* **Muy grandes:** Los gradientes podrían explotar, llevando a entrenamiento inestable (valores NaN).
* **Simetría:** Si todos los pesos empiezan iguales, las neuronas en la misma capa aprenderán la misma cosa, derrotando el propósito de tener múltiples neuronas.

Los **inicializadores de pesos** proporcionan estrategias para establecer estos pesos iniciales inteligentemente, rompiendo la simetría y manteniendo señales/gradientes en un rango razonable para promover aprendizaje estable y eficiente.

**Cómo funcionan (generalmente):**

Típicamente extraen números aleatorios de distribuciones de probabilidad específicas (como uniforme o normal/Gaussiana) cuyos parámetros (como varianza) están escalados basándose en el número de unidades de entrada (`fan_in`) y/o unidades de salida (`fan_out`) de la capa. Este escalado ayuda a mantener la varianza de la señal mientras pasa a través de las capas.

**Inicializadores Clave Proporcionados:**

* **`WeightInitializer` (Clase Base):** Define la interfaz y proporciona un auxiliar `_create_array` para generar `MLArray`s.
* **`XavierUniform` / `XavierNormal` (Inicialización Glorot):**
    * *Concepto:* Escala la varianza de inicialización basándose tanto en `fan_in` como `fan_out`. Busca mantener la varianza consistente hacia adelante y hacia atrás.
    * *Por qué:* Funciona bien con funciones de activación como `sigmoid` y `tanh`. `XavierUniform` usa una distribución uniforme, `XavierNormal` usa una distribución normal.
    * *Código:* Calcula límites/desviación estándar basándose en $\sqrt{6 / (fan\_in + fan\_out)}$ (Uniforme) o $\sqrt{2 / (fan\_in + fan\_out)}$ (Normal).

* **`HeInitialization` (Inicialización Kaiming):**
    * *Concepto:* Escala la varianza de inicialización basándose principalmente en `fan_in`.
    * *Por qué:* Específicamente diseñada para y funciona bien con `relu` y sus variantes, considerando el hecho de que ReLU pone a cero la mitad de las entradas. Usa una distribución normal.
    * *Código:* Calcula desviación estándar basándose en $\sqrt{2 / fan\_in}$.

*(Ver `initializers.py` para detalles de implementación)*

---

## Funciones de Pérdida (`losses.py`)

**¿Qué es una función de pérdida?**

Durante el entrenamiento, necesitamos una manera de medir qué tan "incorrectas" están las predicciones de nuestro modelo comparadas con los valores objetivo reales (verdad fundamental). Esta medida es la **pérdida** (o costo, o error). El objetivo del entrenamiento es ajustar los parámetros del modelo (pesos/sesgos) para **minimizar** este valor de pérdida.

<div align="center">
  <img src="https://github.com/user-attachments/assets/6fe8332d-904f-45f8-a2f6-9bca50ffd576" width="500">
</div>


**Cómo funcionan:**

Una función de pérdida toma las predicciones del modelo (`y_pred`) y los valores objetivo verdaderos (`y_true`) como entrada y produce un valor escalar único representando el error promedio a través de las muestras. Diferentes funciones de pérdida son adecuadas para diferentes tipos de problemas (regresión vs. clasificación) y tienen diferentes propiedades (ej., sensibilidad a valores atípicos).

**Funciones de Pérdida Clave Proporcionadas:**

* **`mse_loss(y_pred, y_true)` (Error Cuadrático Medio):**
    * *Concepto:* Calcula el promedio de las diferencias cuadradas entre predicciones y valores verdaderos: $L = \frac{1}{N} \sum_{i=1}^{N} (y_{pred, i} - y_{true, i})^2$.
    * *Por qué:* Elección estándar para problemas de **regresión**. Penaliza errores más grandes más fuertemente debido al cuadrado. Sensible a valores atípicos.
    * *Código:* Implementa la fórmula usando operaciones `MLArray`.

* **`mae_loss(y_pred, y_true)` (Error Absoluto Medio):**
    * *Concepto:* Calcula el promedio de las diferencias absolutas entre predicciones y valores verdaderos: $L = \frac{1}{N} \sum_{i=1}^{N} |y_{pred, i} - y_{true, i}|$.
    * *Por qué:* Otra elección común para **regresión**. Menos sensible a valores atípicos comparado con MSE porque los errores no se elevan al cuadrado.
    * *Código:* Implementa la fórmula usando operaciones `MLArray`.

* **`binary_cross_entropy(y_pred, y_true)`:**
    * *Concepto:* Mide la diferencia entre dos distribuciones de probabilidad (la probabilidad predicha y la etiqueta verdadera 0 o 1).
    * *Por qué:* La función de pérdida estándar para problemas de **clasificación binaria** donde el modelo produce una probabilidad (usualmente vía una activación `sigmoid`). Espera valores `y_pred` entre 0 y 1.
    * *Código:* Implementa la fórmula, incluye recorte (`epsilon`) para evitar `log(0)`.

* **`categorical_cross_entropy(y_pred, y_true)`:**
    * *Concepto:* Extiende la entropía cruzada binaria a múltiples clases. Compara la distribución de probabilidad predicha (salida por `softmax`) con la distribución verdadera (usualmente codificada one-hot).
    * *Por qué:* La función de pérdida estándar para problemas de **clasificación multi-clase**. Espera que `y_pred` sea una distribución de probabilidad a través de clases para cada muestra.
    * *Código:* Implementa la fórmula, incluye recorte (`epsilon`), suma a través del eje de clase, luego promedia sobre muestras.

* **`huber_loss(y_pred, y_true, delta=1.0)`:**
    * *Concepto:* Una función de pérdida híbrida que se comporta como MSE para errores pequeños (cuadrática) y como MAE para errores grandes (lineal). El parámetro `delta` controla el punto de transición.
    * *Por qué:* Útil para problemas de **regresión** donde quieres robustez a valores atípicos (como MAE) pero también gradientes más suaves alrededor del mínimo (como MSE).
    * *Código:* Implementa la lógica condicional usando operaciones `MLArray`.

*(Ver `losses.py` para detalles de implementación)*

---

## Optimizadores (`optimizers.py`)

**¿Qué hacen los optimizadores?**

Una vez que hemos calculado la pérdida, sabemos qué tan incorrecto está el modelo. También usamos retropropagación (manejada por la diferenciación automática de `MLArray`) para calcular los **gradientes** – cómo cambia la pérdida con respecto a cada peso y sesgo en el modelo.

El **optimizador** es el algoritmo que usa estos gradientes para realmente *actualizar* los parámetros del modelo (pesos y sesgos) de una manera que busca disminuir la pérdida a lo largo del tiempo.

**Cómo funcionan (generalmente):**

Implementan diferentes reglas de actualización basándose en los gradientes y frecuentemente mantienen "estado" interno (como gradientes pasados o momentum) para mejorar velocidad de convergencia y estabilidad. La idea central es usualmente una variación del **descenso de gradiente**: mover los parámetros ligeramente en la dirección opuesta al gradiente. La `learning_rate` controla el tamaño de estos pasos.

**Optimizadores Clave Proporcionados:**

* **`Optimizer` (Clase Base):** Define la interfaz, requiriendo un método `update`. Almacena la `learning_rate`.
* **`SGD` (Descenso de Gradiente Estocástico):**
    * *Concepto:* El optimizador más simple. Actualiza parámetros directamente opuesto al gradiente, escalado por la tasa de aprendizaje ($\theta = \theta - \alpha \nabla_\theta L$). "Estocástico" usualmente significa que el gradiente se calcula en un mini-lote de datos, no el conjunto de datos completo.
    * *Por qué:* Fácil de entender, pero puede ser lento, quedarse atascado en mínimos locales, u oscilar.
    * *Código:* Implementa la regla de actualización básica.

* **`SGDMomentum`:**
    * *Concepto:* Agrega un término de "momentum" que acumula un promedio exponencialmente decayente de gradientes pasados. Esto ayuda a acelerar el descenso en direcciones consistentes y amortigua oscilaciones ($v = \beta v + \alpha \nabla_\theta L$, $\theta = \theta - v$).
    * *Por qué:* Frecuentemente converge más rápido y de manera más confiable que SGD básico. Introduce `momentum_coefficient` ($\beta$) y mantiene estado de velocidad (`self.velocities`) por parámetro.
    * *Código:* Implementa la regla de actualización de momentum, almacenando velocidades.

* **`AdaGrad` (Gradiente Adaptativo):**
    * *Concepto:* Adapta la tasa de aprendizaje *por parámetro*, usando actualizaciones más pequeñas para parámetros que cambian frecuentemente y actualizaciones más grandes para los infrecuentes. Divide la tasa de aprendizaje por la raíz cuadrada de la suma de gradientes cuadrados pasados ($\theta = \theta - \frac{\alpha}{\sqrt{G + \epsilon}} \nabla_\theta L$).
    * *Por qué:* Bueno para datos dispersos (como en NLP). Sin embargo, la tasa de aprendizaje decrece monótonamente y puede volverse muy pequeña. Mantiene suma de gradientes cuadrados (`self.squared_gradients`) como estado.
    * *Código:* Implementa la regla de actualización AdaGrad, almacenando gradientes cuadrados.

* **`Adam` (Estimación de Momento Adaptativo):**
    * *Concepto:* Combina las ideas de Momentum (usando un promedio exponencialmente decayente de gradientes pasados - 1er momento) y RMSProp/AdaGrad (usando un promedio exponencialmente decayente de gradientes *cuadrados* pasados - 2do momento). Incluye términos de corrección de sesgo para considerar la inicialización.
    * *Por qué:* Frecuentemente considerado un optimizador robusto y efectivo por defecto para muchos problemas. Requiere ajustar `learning_rate`, `exp_decay_gradients` ($\beta_1$), y `exp_decay_squared` ($\beta_2$). Mantiene estimaciones de 1er y 2do momento (`self.gradients_momentum`, `self.squared_gradients_momentum`) y un `timestep`.
    * *Código:* Implementa la regla de actualización Adam con corrección de sesgo, almacenando estimaciones de momento.

*(Ver `optimizers.py` para detalles de implementación)*

---

## Utilidades de Memoria (`memory.py`)

**¿Por qué medir memoria?**

Los modelos de machine learning, especialmente los grandes como redes neuronales profundas o bosques aleatorios complejos, pueden consumir cantidades significativas de memoria (RAM). Entender la huella de memoria de tus estructuras de datos (`Value`, `MLArray`) y modelos (`NeuralNetwork`, `DecisionTree`, etc.) es crucial para:
* **Planificación de Recursos:** Asegurar que tu hardware pueda manejar el tamaño del modelo.
* **Depuración:** Identificar cuellos de botella de memoria o uso inesperado.
* **Optimización:** Comparar la eficiencia de memoria de diferentes arquitecturas de modelos o implementaciones.

**Cómo funcionan:**

Estas funciones de utilidad usan el `sys.getsizeof` incorporado de Python para estimar el uso de memoria de objetos. Para objetos complejos como `MLArray` o estructuras de modelos (que contienen objetos anidados o listas), recorren recursivamente los componentes y suman sus tamaños.

**Utilidades Clave Proporcionadas:**

* **`format_size(size_bytes)`:** Convierte conteos de bytes crudos en formatos legibles por humanos (KB, MB, GB).
* **`calculate_value_size(value)`:** Estima el tamaño de un solo objeto `smolml.core.value.Value` (incluyendo sus datos, grad, etc.).
* **`calculate_mlarray_size(arr)`:** Estima el tamaño de un `MLArray`, incluyendo las listas anidadas/objetos `Value` que contiene.
* **`calculate_neural_network_size(model)`:** Estima el tamaño total de una `NeuralNetwork`, incluyendo sus capas (pesos, sesgos) y estado del optimizador.
* **(Otras funciones similares):** `calculate_decision_node_size`, `calculate_regression_size`, `calculate_decision_tree_size`, `calculate_random_forest_size` estiman memoria para otros tipos de modelos.

**Nota:** Estas proporcionan *estimaciones*. El uso real de memoria puede estar influenciado por la gestión interna de memoria de Python, referencias de objetos compartidos, etc.

*(Ver `memory.py` para detalles de implementación)*