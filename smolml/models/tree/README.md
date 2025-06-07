# SmolML - Modelos de Árboles: ¡Decisiones, Decisiones!

Bienvenido a la *rama* de SmolML que trata con **Modelos Basados en Árboles (Tree-Based Models)**! A diferencia de los modelos que vimos en **Regresión** (que dependen de ecuaciones suaves y descenso de gradiente), los Árboles de Decisión (Decision Trees) y su poderoso hermano, los Bosques Aleatorios (Random Forests), hacen predicciones aprendiendo una serie de **reglas de decisión (decision rules)** explícitas de los datos. Piensa en ello como construir un diagrama de flujo sofisticado para clasificar un email como spam o no spam, o para predecir el precio de una casa.

Estos modelos son increíblemente versátiles, manejando tanto problemas de **clasificación (classification)** (prediciendo categorías) como de **regresión (regression)** (prediciendo valores numéricos). No necesitan escalado de características y pueden capturar relaciones complejas y no lineales. ¡Vamos a profundizar en cómo funcionan!

## Árboles de Decisión: El Enfoque de Diagrama de Flujo

Imagina que estás tratando de decidir si deberías jugar tenis hoy. Podrías preguntar:
1.  ¿Es soleado el pronóstico?
    * Sí -> ¿Es alta la humedad?
        * Sí -> No Jugar
        * No -> ¡Jugar!
    * No -> ¿Está lloviendo?
        * Sí -> No Jugar
        * No -> ¡Jugar!

¡Esa es la esencia de un **Árbol de Decisión (Decision Tree)**! Es una estructura que divide recursivamente los datos basándose en preguntas simples sobre las características de entrada.

<div align="center">
  <img src="https://github.com/user-attachments/assets/0b805169-fa57-4097-80e0-e841ea3246af" width="600">
</div>

**Cómo se Construye (El método `fit`):**

La magia sucede en el método `fit` de la clase `DecisionTree` (ver `decision_tree.py`). Construye la estructura del árbol, representada por objetos `DecisionNode` interconectados, usando un proceso llamado **particionamiento recursivo (recursive partitioning)**:

1.  **Empezar con todos los datos:** Comenzar en el nodo raíz con todo tu conjunto de datos de entrenamiento.
2.  **Encontrar la Mejor Pregunta:** La tarea central es encontrar la *mejor* característica y el *mejor* valor umbral para dividir los datos actuales en dos grupos (ramas izquierda y derecha). ¿Qué es "mejor"? Una división que hace que los grupos resultantes sean lo más "puros" u homogéneos posible respecto a la variable objetivo (ej., todas las muestras en un grupo pertenecen a la misma clase, o tienen valores numéricos muy similares).
    * **¿Cómo? Los métodos `_find_best_split` y `_calculate_gain`:** El árbol prueba *cada división posible* (cada característica, cada valor único como umbral) y evalúa qué tan "más puros" son los grupos resultantes comparados con el grupo padre.
        * **Para Clasificación:** Típicamente usa **Entropía (Entropy)** (una medida de desorden) y calcula la **Ganancia de Información (Information Gain)** (cuánto decrece la entropía después de la división). Una ganancia mayor significa una mejor división. (Ver `_information_gain`).
        * **Para Regresión:** Típicamente usa **Varianza (Variance)** o **Error Cuadrático Medio (MSE)** y calcula cuánto se reduce esta métrica por la división. Una reducción mayor significa una mejor división. (Ver `_mse_reduction`).
3.  **Dividir los Datos:** Aplicar la mejor división encontrada, dividiendo los datos en dos subconjuntos.
4.  **Repetir Recursivamente:** Tratar cada subconjunto como un nuevo problema y repetir los pasos 2 y 3 para las ramas izquierda y derecha, creando nodos hijos (el método `_grow_tree` se llama a sí mismo).
5.  **Parar de Dividir (Crear un Nodo Hoja):** La recursión se detiene, y se crea un **nodo hoja (leaf node)** (un `DecisionNode` con un `value` pero sin hijos) cuando se cumplen ciertas condiciones:
    * El nodo es perfectamente "puro" (todas las muestras pertenecen a la misma clase/tienen valores muy similares - verificar `_is_pure`).
    * Se alcanza una `max_depth` predefinida.
    * El número de muestras en un nodo cae por debajo de `min_samples_split`.
    * Una división potencial resultaría en un nodo hijo teniendo menos de `min_samples_leaf` muestras.
    * Ninguna división adicional mejora la pureza.
    Estos criterios de parada (hiperparámetros establecidos durante `__init__`) son cruciales para prevenir que el árbol crezca demasiado complejo y sufra **sobreajuste (overfitting)** (memorizar los datos de entrenamiento en lugar de aprender patrones generales).

**Haciendo Predicciones (El método `predict`):**

¡Una vez que el árbol está construido, predecir es directo! Para un nuevo punto de datos:
1.  Empezar en el nodo `root`.
2.  Verificar la regla de decisión (característica y umbral) en el nodo actual.
3.  Seguir la rama correspondiente (izquierda si `valor_característica <= umbral`, derecha de otro modo).
4.  Repetir pasos 2 y 3 hasta llegar a un nodo hoja (método `_traverse_tree`).
5.  La predicción es el valor almacenado en ese nodo hoja (`node.value`). Este valor se determina durante el entrenamiento (`_leaf_value`):
    * Clasificación: La clase más común entre las muestras de entrenamiento que terminaron en esta hoja.
    * Regresión: El valor promedio de las muestras de entrenamiento que terminaron en esta hoja.

¡Genial, verdad? Un solo árbol es intuitivo, pero a veces pueden ser un poco inestables y propensos al sobreajuste. ¿Qué tal si pudiéramos combinar *muchos* árboles?

## Bosques Aleatorios: La Sabiduría de Muchos Árboles

<div align="center">
  <img src="https://github.com/user-attachments/assets/6a652774-4fc3-4ed1-89d4-e9eaf1410e2a" width="600">
</div>

Un solo Árbol de Decisión puede ser sensible a los datos específicos en los que se entrena. Un conjunto de datos ligeramente diferente podría producir una estructura de árbol muy diferente. Los **Bosques Aleatorios (Random Forests)** abordan esto construyendo un *ensamble (ensemble)* (un "bosque") de muchos Árboles de Decisión y combinando sus predicciones. ¡Es como preguntar a muchos expertos diferentes (árboles) y seguir el consenso!

**Los Secretos "Aleatorios" (clase `RandomForest` en `random_forest.py`):**

Los Bosques Aleatorios introducen aleatoriedad inteligente durante el entrenamiento (`fit`) de árboles individuales para hacerlos diversos:

1.  **Bagging (Bootstrap Aggregating):** Cada árbol en el bosque se entrena en un conjunto de datos ligeramente diferente. Esto se hace mediante **bootstrapping**: crear una muestra aleatoria de los datos de entrenamiento originales *con reemplazo*. Esto significa que algunos puntos de datos podrían aparecer múltiples veces en el conjunto de entrenamiento de un árbol, mientras otros podrían quedar completamente fuera. (Controlado por el parámetro `bootstrap` e implementado en `_bootstrap_sample`). ¿Por qué? Asegura que cada árbol vea una perspectiva ligeramente diferente de los datos.
2.  **Subconjuntos Aleatorios de Características:** Al encontrar la mejor división en cada nodo dentro de *cada* árbol, el algoritmo no considera *todas* las características. En su lugar, solo evalúa un **subconjunto aleatorio** de las características (parámetro `max_features`). (Ver `_get_max_features` y la lógica `_find_best_split` modificada inyectada durante `RandomForest.fit`). ¿Por qué? Esto previene que unas pocas características muy fuertes dominen *todos* los árboles, forzando a otras características a ser consideradas y llevando a estructuras de árbol más diversas.

**Construyendo el Bosque (`fit`):**

El método `RandomForest.fit` esencialmente hace esto:
* Hacer un bucle `n_trees` veces:
    * Crear una muestra bootstrap de los datos (si `bootstrap=True`).
    * Instanciar un `DecisionTree`.
    * Inyectar la lógica de "subconjunto aleatorio de características" en el mecanismo de división del árbol.
    * Entrenar el `DecisionTree` en los datos muestreados con la división modificada.
    * Almacenar el árbol entrenado en `self.trees`.

**Haciendo Predicciones (`predict`):**

Para hacer una predicción para un nuevo punto de datos, el Bosque Aleatorio le pide a *cada árbol* en su ensamble (`self.trees`) que haga una predicción. Luego, las combina:
* **Clasificación:** Toma un **voto mayoritario**. La clase predicha por la mayoría de árboles gana.
* **Regresión:** Calcula el **promedio** de todas las predicciones de los árboles individuales.

Este proceso de agregación típicamente lleva a modelos que son mucho más robustos, menos propensos al sobreajuste, y generalizan mejor a datos nuevos no vistos comparado con un solo Árbol de Decisión.

## Ejemplo de Uso

Veamos cómo podrías usar un `RandomForest` (asumiendo tarea de clasificación):

```python
from smolml.models.tree import RandomForest, DecisionTree
from smolml.core.ml_array import MLArray

# Datos de Muestra (ej., 4 características, 5 muestras para clasificación)
X_data = [[5.1, 3.5, 1.4, 0.2],
          [4.9, 3.0, 1.4, 0.2],
          [6.7, 3.1, 4.4, 1.4],
          [6.0, 2.9, 4.5, 1.5],
          [5.8, 2.7, 5.1, 1.9]]
# Clases objetivo (ej., 0, 1, 2)
y_data = [0, 0, 1, 1, 2]

# Convertir a MLArray (aunque los métodos fit/predict manejan la conversión)
X = MLArray(X_data)
y = MLArray(y_data)

# --- Usando un Árbol de Decisión ---
print("--- Entrenando Árbol de Decisión ---")
dt = DecisionTree(max_depth=3, task="classification")
dt.fit(X, y)
print(dt) # Muestra estructura y estadísticas
dt_pred = dt.predict(X)
print(f"Predicciones DT en datos de entrenamiento: {dt_pred.to_list()}")

# --- Usando un Bosque Aleatorio ---
print("\n--- Entrenando Bosque Aleatorio ---")
# Construir un bosque de 10 árboles
rf = RandomForest(n_trees=10, max_depth=3, task="classification")
rf.fit(X, y)
print(rf) # Muestra estadísticas del bosque
rf_pred = rf.predict(X)
print(f"Predicciones RF en datos de entrenamiento: {rf_pred.to_list()}")

# Predecir en nuevos datos
X_new = MLArray([[6.0, 3.0, 4.8, 1.8], [5.0, 3.4, 1.6, 0.4]])
rf_new_pred = rf.predict(X_new)
print(f"\nPredicciones RF en datos nuevos {X_new.to_list()}: {rf_new_pred.to_list()}")
```

## De raíces a hojas, de datos a predicciones

Los Árboles de Decisión ofrecen una manera interpretable, similar a diagramas de flujo, para modelar datos, mientras que los Bosques Aleatorios aprovechan el poder del aprendizaje en ensamble (combinando muchos árboles diversos) para crear modelos altamente precisos y robustos tanto para clasificación como regresión. ¡Representan un paradigma diferente de la optimización basada en gradientes, pero son una piedra angular del machine learning práctico!