# SmolML - K-Means: ¡Encontrando Grupos en tus Datos!

Hasta ahora, hemos visto modelos que aprenden de datos etiquetados (**aprendizaje supervisado (supervised learning)**): predecir precios de casas (*regresión*) o clasificar imágenes (*clasificación*). Pero ¿qué pasa si solo tienes un gran montón de datos y no hay etiquetas? ¿Cómo puedes encontrar estructuras o grupos interesantes dentro de ellos? ¡Bienvenido al **Aprendizaje No Supervisado (Unsupervised Learning)**, y una de sus herramientas más populares: **Clustering K-Means**!

Imagina que tienes puntos de datos esparcidos en un gráfico. K-Means trata de agrupar automáticamente estos puntos en un número especificado (`k`) de clusters, sin conocimiento previo de qué deberían ser esos grupos. Es como tratar de encontrar agrupaciones naturales de clientes basándose en su comportamiento de compra, o agrupar estrellas basándose en su brillo y temperatura.

¡Esta parte de SmolML implementa el algoritmo K-Means usando nuestra confiable clase `MLArray`. Veamos cómo funciona!

## El Algoritmo K-Means: ¡Una Danza de Clusters!

<div align="center">
  <img src="https://github.com/user-attachments/assets/1d85e199-e5d1-4ff0-a70b-e1d8ab970e13" width="600">
</div>

K-Means busca particionar tus datos en `k` clusters distintos y no superpuestos. Hace esto encontrando `k` puntos centrales, llamados **centroides (centroids)**, y asignando cada punto de datos al centroide más cercano. La idea central es un proceso iterativo, una especie de "danza" entre asignar puntos y actualizar los centros:

1.  **Inicialización - Elegir Puntos de Inicio (`_initialize_centroids`):**
    * Primero, necesitamos decidir cuántos clusters (`k`, o `n_clusters` en nuestro código) queremos encontrar. Esto es algo que *tú* le dices al algoritmo.
    * Luego, necesitamos conjeturas iniciales para la ubicación de los `k` centroides de cluster. Una manera común (y la que usamos aquí) es simplemente elegir `k` puntos de datos aleatorios de tu conjunto de datos y llamarlos los centroides iniciales. Piensa en esto como dejar caer aleatoriamente `k` alfileres en tu mapa de datos.
    * **Conexión de Código:** El método `_initialize_centroids` maneja esto, muestreando aleatoriamente `n_clusters` puntos de la entrada `X_train` para establecer los `self.centroids` iniciales. También inicializa `self.centroid_history` para rastrear cómo se mueven estos centros.

2.  **Paso de Asignación - Encontrar tu Centro Más Cercano (`_assign_clusters`):**
    * Ahora, para *cada* punto de datos, calcula su distancia a *cada* uno de los `k` centroides. La medida de distancia más común es la buena **distancia Euclidiana (Euclidean distance)** (la distancia en línea recta).
    * Asigna cada punto de datos al cluster cuyo centroide esté más cerca de él. Cada punto ahora pertenece a uno de los `k` clusters.
    * **Conexión de Código:** El método `_compute_distances` calcula todas estas distancias eficientemente. Usa inteligentemente las operaciones de broadcasting y vectorizadas de `MLArray` (`reshape`, sustracción, multiplicación elemento por elemento, `sum`, `sqrt`) para evitar bucles lentos de Python. El método `_assign_clusters` luego itera a través de la matriz de distancias resultante para cada punto, encuentra el índice (`cluster_idx`) de la distancia mínima, y almacena estas asignaciones en `self.labels_`.

3.  **Paso de Actualización - Mover el Centro (`_update_centroids`):**
    * Los centroides iniciales eran solo conjeturas aleatorias. Ahora que tenemos puntos asignados a clusters, podemos calcular ubicaciones de centroides *mejores*.
    * Para cada cluster, encontrar el nuevo centroide calculando la **media (mean)** (posición promedio) de todos los puntos de datos asignados a ese cluster en el paso anterior. Imagina encontrar el "centro de gravedad" para todos los puntos en un cluster – esa es la nueva posición del centroide.
    * **Conexión de Código:** El método `_update_centroids` hace esto. Agrupa puntos basándose en `self.labels_`, calcula la media para cada dimensión dentro de cada grupo, y actualiza `self.centroids`. También maneja casos donde un cluster podría quedar vacío (en cuyo caso el centroide se queda quieto).

4.  **¡Repetir!**
    * Ahora que los centroides se han movido, ¡las distancias han cambiado! Volver al **Paso de Asignación (2)** y reasignar todos los puntos de datos a su *nuevo* centroide más cercano.
    * Luego, volver al **Paso de Actualización (3)** y recalcular las posiciones de los centroides basándose en las nuevas asignaciones.
    * Seguir repitiendo esta danza de asignar-y-actualizar.

**¿Cuándo Para la Danza? (Convergencia)**

El algoritmo sigue iterando hasta que sucede una de estas cosas:
* **Los Centroides se Asientan:** Los centroides dejan de moverse significativamente entre iteraciones. La distancia total que movieron los centroides es menor que un umbral pequeño (`tol`). (Verificado dentro de `_update_centroids`).
* **Se Alcanzan las Iteraciones Máximas:** Llegamos al número máximo predefinido de iteraciones (`max_iters`) para prevenir que el algoritmo corra para siempre si no converge rápidamente.

**El Resultado:**

Una vez que el algoritmo se detiene (converge), tienes:
* `self.centroids`: Las posiciones finales de los `k` centros de cluster.
* `self.labels_`: Un array indicando a qué cluster (0 a k-1) pertenece cada uno de tus puntos de datos de entrada.

## Implementación en SmolML (clase `KMeans`)

La clase `KMeans` en `kmeans.py` envuelve todo este proceso:
* **`__init__(self, n_clusters, max_iters, tol)`:** Inicializas el modelo diciéndole cuántos clusters encontrar (`n_clusters`), las iteraciones máximas (`max_iters`), y la tolerancia de convergencia (`tol`).
* **`fit(self, X_train)`:** Este es el método principal de entrenamiento. Toma tus datos (`X_train`, esperado como `MLArray` o lista-de-listas) y ejecuta el bucle iterativo de asignar-y-actualizar descrito arriba, llamando los métodos internos (`_initialize_centroids`, `_compute_distances`, `_assign_clusters`, `_update_centroids`) hasta convergencia. Almacena los centroides finales y etiquetas internamente.
* **`predict(self, X)`:** Después del ajuste, puedes usar este método para asignar *nuevos* puntos de datos (`X`) al centroide aprendido más cercano (`self.centroids`).
* **`fit_predict(self, X_train)`:** Un atajo conveniente que llama `fit` y luego inmediatamente `predict` en los mismos datos, retornando las etiquetas de cluster para los datos de entrenamiento.

## Ejemplo de Uso

Encontremos 3 clusters en algunos datos 2D simples:

```python
from smolml.cluster import KMeans
from smolml.core.ml_array import MLArray
import random

# Generar algunos datos sintéticos 2D alrededor de 3 centros
def generate_data(n_samples, centers):
    data = []
    for _ in range(n_samples):
        center = random.choice(centers)
        # Agregar algo de ruido aleatorio alrededor del centro
        point = [center[0] + random.gauss(0, 0.5),
                 center[1] + random.gauss(0, 0.5)]
        data.append(point)
    return data

centers = [[2, 2], [8, 3], [5, 8]]
X_data = generate_data(150, centers)

# Convertir a MLArray
X = MLArray(X_data)

# Inicializar y ajustar K-Means
k = 3 # Queremos encontrar 3 clusters
kmeans = KMeans(n_clusters=k, max_iters=100, tol=1e-4)

print("Ajustando K-Means...")
kmeans.fit(X)

# Obtener los resultados
final_centroids = kmeans.centroids
cluster_labels = kmeans.labels_

print("\n¡Ajuste de K-Means completo!")
print(f"Posiciones finales de Centroides:\n{final_centroids}")
# print(f"Etiquetas de cluster para los primeros 10 puntos: {cluster_labels.to_list()[:10]}")
print(f"Número de puntos en cada cluster:")
labels_list = cluster_labels.to_list()
for i in range(k):
    print(f"  Cluster {i}: {labels_list.count(i)} puntos")

# ¡Ahora podrías usar estas etiquetas o centroides para análisis adicional o visualización!
# Por ejemplo, predecir el cluster para un nuevo punto:
new_point = MLArray([[6, 6]])
predicted_cluster = kmeans.predict(new_point)
print(f"\nNuevo punto {new_point.to_list()} asignado al cluster: {predicted_cluster.to_list()[0]}")
```

> (Nota: Porque K-Means empieza con centroides iniciales aleatorios, podrías obtener resultados de clustering ligeramente diferentes cada vez que lo ejecutes. Ejecutarlo múltiples veces y elegir el mejor resultado basándose en alguna métrica es una práctica común, aunque no está implementado aquí.)

## Final de la danza

K-Means es un algoritmo no supervisado fundamental, fantástico para explorar datos sin etiquetas y descubrir agrupaciones potenciales. Es intuitivo, relativamente simple de implementar (¡especialmente con herramientas como nuestro `MLArray` para matemáticas eficientes!), y frecuentemente proporciona insights valiosos sobre la estructura oculta de tus datos. ¡Es un gran primer paso hacia el mundo del aprendizaje no supervisado!