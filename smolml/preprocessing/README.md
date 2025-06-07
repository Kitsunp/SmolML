# SmolML - Preprocesamiento: Haz que tus datos sean significativos

Antes de alimentar nuestros preciosos datos a muchos algoritmos de machine learning, frecuentemente hay un paso crucial de preprocesamiento: **Escalado de Características**. ¿Por qué? ¡Porque los algoritmos pueden ser bastante sensibles a la *escala* o *rango* de nuestras características de entrada!

## ¿Por qué Molestarse en Escalar? ¡Hablemos de Números!

Imagina que tienes un conjunto de datos para predecir precios de casas con características como:
* `size_sqft`: rango de 500 a 5000 pies cuadrados
* `num_bedrooms`: rango de 1 a 5
* `distance_to_school_km`: rango de 0.1 a 10 km

Ahora, considera algoritmos que usan distancias (como K-Means) o dependen del descenso de gradiente (como Regresión Lineal o Redes Neuronales).
* **Algoritmos Basados en Distancia:** Si calculas la distancia entre dos casas, una diferencia de 1000 pies cuadrados eclipsará numéricamente una diferencia de 2 habitaciones, solo porque los *números* son más grandes. El algoritmo podría pensar erróneamente que `size_sqft` es vastamente más importante, únicamente debido a su rango más grande.
* **Algoritmos Basados en Gradiente:** Características con escalas vastamente diferentes pueden causar que el proceso de optimización (encontrar los mejores pesos del modelo) sea lento e inestable. Piensa en tratar de encontrar el fondo de un valle donde un lado es increíblemente empinado (característica de rango grande) y el otro es muy suave (característica de rango pequeño) – ¡es complicado!

<div align="center">
  <img src="https://github.com/user-attachments/assets/2930477a-a175-41b0-a802-bdaa5ff04bbc" width="600">
</div>

**El Objetivo:** El escalado de características lleva todas las características a un campo de juego numérico similar. Esto previene que características con valores más grandes dominen el proceso de aprendizaje solo por su escala, frecuentemente llevando a convergencia de entrenamiento más rápida y a veces incluso mejor rendimiento del modelo.

SmolML proporciona dos escaladores comunes, construidos usando nuestro `MLArray`.

## `StandardScaler`: Media Cero, Varianza Unitaria

<div align="center">
  <img src="https://github.com/user-attachments/assets/dda0fe2b-5e9f-4fc2-a5c6-61db874e2d88" width="850">
</div>

Esta es una de las técnicas de escalado más populares, frecuentemente llamada **normalización z-score**.

**El Concepto:** Transforma los datos para cada característica de modo que tenga:
* Una **media ($\mu$) de 0**.
* Una **desviación estándar ($\sigma$) de 1**.

**Cómo Funciona:** Para cada valor $x$ en una característica, aplica la fórmula:
$$ z = \frac{x - \mu}{\sigma} $$
* **Restar la media ($\mu$)**: Esto centra los datos para esa característica alrededor de cero.
* **Dividir por la desviación estándar ($\sigma$)**: Esto escala los datos de modo que tenga una desviación estándar de 1, significando que la "dispersión" de los datos se vuelve consistente a través de las características.

**Implementación (clase `StandardScaler` en `scalers.py`):**

Es un proceso de dos pasos:

1.  **`fit(self, X)`:** Este método aprende los parámetros necesarios *de tus datos de entrenamiento*.
    * Calcula la media (`self.mean`) y desviación estándar (`self.std`) para *cada columna de característica* (usando `X.mean(axis=0)` y `X.std(axis=0)` de nuestro `MLArray`).
    * Almacena estos valores calculados de `mean` y `std` dentro del objeto escalador.
    * *Atención:* También incluye lógica para manejar casos donde una característica tiene desviación estándar cero (es decir, todos los valores son iguales). En este caso, establece la desviación estándar a 1 para evitar división por cero durante el paso de transformación.

2.  **`transform(self, X)`:** Este método aplica el escalado usando los parámetros *previamente aprendidos*.
    * Toma cualquier conjunto de datos `X` (podría ser tus datos de entrenamiento otra vez, o nuevos datos de prueba/predicción) y aplica la fórmula z-score: `(X - self.mean) / self.std`.
    * El resultado son tus datos escalados, ¡listos para tu modelo!

## `MinMaxScaler`: Comprimiendo en [0, 1]

<div align="center">
  <img src="https://github.com/user-attachments/assets/f2153e47-bf00-482e-9784-567a462b96e1" width="850">
</div>

Otra técnica común, especialmente útil cuando quieres características limitadas dentro de un rango específico.

**El Concepto:** Transforma los datos para cada característica de modo que todos los valores caigan ordenadamente dentro del rango **[0, 1]**.

**Cómo Funciona:** Para cada valor $x$ en una característica, aplica la fórmula:
$$ x_{escalado} = \frac{x - x_{min}}{x_{max} - x_{min}} $$
* **Restar el mínimo ($x_{min}$)**: Esto desplaza los datos de modo que el valor mínimo se convierte en 0.
* **Dividir por el rango ($x_{max} - x_{min}$)**: Esto escala los datos proporcionalmente de modo que el valor máximo se convierte en 1. Todos los otros valores caen en algún lugar intermedio.

**Implementación (clase `MinMaxScaler` en `scalers.py`):**

Proceso similar de dos pasos:

1.  **`fit(self, X)`:** Aprende los parámetros de los datos de entrenamiento.
    * Encuentra el valor mínimo (`self.min`) y máximo (`self.max`) para *cada columna de característica* (usando `X.min(axis=0)` y `X.max(axis=0)` de `MLArray`).
    * Almacena estos valores `min` y `max`.

2.  **`transform(self, X)`:** Aplica el escalado usando los parámetros aprendidos.
    * Toma cualquier conjunto de datos `X` y aplica la fórmula Min-Max: `(X - self.min) / (self.max - self.min)`.
    * ¡Voilà! Tus datos ahora están escalados entre 0 y 1.

## La Regla de Oro: ¡Ajustar en Entrenamiento, Transformar Entrenamiento y Prueba!

¡Esto es súper importante!

* Deberías **únicamente** llamar al método `fit()` o `fit_transform()` en tus **datos de entrenamiento**. El escalador necesita aprender la media/std o min/max *únicamente* de los datos en los que tu modelo entrenará.
* Luego usas el *mismo escalador ajustado* (con los parámetros aprendidos) para llamar `transform()` en tus **datos de entrenamiento** Y tus **datos de prueba/validación/predicción nueva**.

¿Por qué? Quieres aplicar la *misma transformación exacta* a todos tus datos, basándose únicamente en lo que el modelo aprendió durante el entrenamiento. Ajustar el escalador en los datos de prueba sería una forma de "fuga de datos" – permitir que tu paso de preprocesamiento vea datos que no debería ver aún.

**Conveniencia:** Ambos escaladores tienen un método `fit_transform(self, X)` que simplemente llama `fit(X)` seguido de `transform(X)` en los mismos datos – perfecto para aplicar escalado a tu conjunto de entrenamiento de una vez.

## Ejemplo de Uso

Escalemos algunos datos simples usando `StandardScaler`:

```python
from smolml.preprocessing import StandardScaler, MinMaxScaler
from smolml.core.ml_array import MLArray

# Datos de muestra (ej., 3 muestras, 2 características con diferentes escalas)
X_train_data = [[10, 100],
                [20, 150],
                [30, 120]]

# Nuevos datos para predecir más tarde (¡deben escalarse de la misma manera!)
X_new_data = [[15, 110],
              [25, 160]]

# Convertir a MLArray
X_train = MLArray(X_train_data)
X_new = MLArray(X_new_data)

# --- Usando StandardScaler ---
print("--- Usando StandardScaler ---")
scaler_std = StandardScaler()

# Ajustar UNA VEZ en datos de entrenamiento, luego transformarlos
X_train_scaled_std = scaler_std.fit_transform(X_train)

# Transformar los datos NUEVOS usando el MISMO escalador (ya ajustado)
X_new_scaled_std = scaler_std.transform(X_new)

print("Datos de Entrenamiento Originales:\n", X_train)
print("Datos de Entrenamiento Escalados (StandardScaler):\n", X_train_scaled_std)
print("\nDatos Nuevos Originales:\n", X_new)
print("Datos Nuevos Escalados (StandardScaler):\n", X_new_scaled_std)
print(f"Media Aprendida: {scaler_std.mean}")
print(f"Desviación Estándar Aprendida: {scaler_std.std}")


# --- Usando MinMaxScaler ---
print("\n--- Usando MinMaxScaler ---")
scaler_minmax = MinMaxScaler()

# Ajustar UNA VEZ en datos de entrenamiento, luego transformarlos
X_train_scaled_mm = scaler_minmax.fit_transform(X_train)

# Transformar los datos NUEVOS usando el MISMO escalador (ya ajustado)
X_new_scaled_mm = scaler_minmax.transform(X_new)

print("Datos de Entrenamiento Originales:\n", X_train)
print("Datos de Entrenamiento Escalados (MinMaxScaler):\n", X_train_scaled_mm)
print("\nDatos Nuevos Originales:\n", X_new)
print("Datos Nuevos Escalados (MinMaxScaler):\n", X_new_scaled_mm)
print(f"Mínimo Aprendido: {scaler_minmax.min}")
print(f"Máximo Aprendido: {scaler_minmax.max}")
```

## ¡Escalando!

El escalado de características es un paso fundamental en el pipeline de machine learning para muchos algoritmos. Al llevar características a una escala común usando técnicas como estandarización (`StandardScaler`) o normalización (`MinMaxScaler`), frecuentemente puedes mejorar la velocidad de convergencia y rendimiento de tu modelo. ¡SmolML proporciona estas herramientas esenciales, aprovechando el poder computacional de MLArray para hacer el preprocesamiento directo! No olvides la regla de oro: ¡ajustar solo en entrenamiento, transformar todo!