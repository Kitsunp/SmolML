from smolml.core.ml_array import MLArray
from smolml.core.value import Value
import random

"""
///////////////
/// KMEANS ///
///////////////

Algoritmo de clustering no supervisado que particiona n muestras en k clusters.
Cada cluster está representado por la media de sus puntos (centroide).
La implementación se enfoca en usar MLArray para cálculos y manejo.
"""
class KMeans:
    """
    Implementación del algoritmo de clustering K-means usando MLArray.
    Particiona datos en k clusters actualizando iterativamente centros de cluster
    y reasignando puntos al centro más cercano. Usa distancia Euclidiana
    para medición de similitud y medias para actualizaciones de centroides.
    """
    def __init__(self, n_clusters, max_iters, tol) -> None:
        """
        Inicializa KMeans con el número de clusters, iteraciones máximas,
        y tolerancia de convergencia. Configura marcadores de posición vacíos para centroides
        y asignaciones de clusters.
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels_ = None
        self.centroid_history = []

    def _initialize_centroids(self, X_train):
        """
        Selecciona aleatoriamente k puntos de los datos de entrada para servir como
        centroides iniciales. Usa muestreo aleatorio sin reemplazo para asegurar
        posiciones iniciales distintas.
        """
        centroids = random.sample(X_train.to_list(), self.n_clusters)
        self.centroids = MLArray(centroids)
        self.centroid_history = [self.centroids.to_list()]
        return self.centroids
    
    def _compute_distances(self, X_train):
        """
        Calcula las distancias Euclidianas entre todos los puntos de datos y todos
        los centroides. Usa broadcasting de MLArray para cálculo eficiente
        y evita bucles explícitos donde sea posible.
        """
        diff = X_train.reshape(-1, 1, X_train.shape[1]) - self.centroids
        squared_diff = diff * diff
        squared_distances = squared_diff.sum(axis=2)
        distances = squared_distances.sqrt()
        return distances
    
    def _assign_clusters(self, distance_matrix):
        """
        Asigna cada punto de datos a su centroide más cercano basándose en la
        matriz de distancias calculada. Convierte arrays a listas para encontrar
        mínimos eficientemente y maneja la conversión de vuelta a formato MLArray.
        """
        distances = distance_matrix.to_list()
        labels = []
        
        for sample_distances in distances:
            min_distance = float('inf')
            min_index = 0
            
            for cluster_idx, distance in enumerate(sample_distances):
                if distance < min_distance:
                    min_distance = distance
                    min_index = cluster_idx
                    
            labels.append(min_index)
        
        self.labels_ = MLArray(labels)
        return self.labels_
    
    def _update_centroids(self, X_train):
        """
        Actualiza posiciones de centroides calculando la media de todos los puntos
        asignados a cada cluster. Maneja clusters vacíos manteniendo
        sus posiciones previas. Verifica convergencia midiendo el
        movimiento total de todos los centroides.
        """
        X_data = X_train.to_list()
        labels = self.labels_.to_list()
        new_centroids = []
        
        for cluster_idx in range(self.n_clusters):
            cluster_points = []
            for point_idx, label in enumerate(labels):
                if label == cluster_idx:
                    cluster_points.append(X_data[point_idx])
            
            if cluster_points:
                centroid = []
                n_features = len(cluster_points[0])
                for feature_idx in range(n_features):
                    feature_sum = sum(point[feature_idx] for point in cluster_points)
                    feature_mean = feature_sum / len(cluster_points)
                    centroid.append(feature_mean)
                new_centroids.append(centroid)
            else:
                new_centroids.append(self.centroids.to_list()[cluster_idx])
        
        old_centroids = self.centroids
        self.centroids = MLArray(new_centroids)
        self.centroid_history.append(new_centroids)  # Registrar nuevos centroides
        
        if old_centroids is not None:
            diff = self.centroids - old_centroids
            movement = (diff * diff).sum().sqrt()
            return movement.data < self.tol
        
        return False

    def fit(self, X_train):
        """
        Loop de entrenamiento principal del algoritmo KMeans. Inicializa centroides
        y los refina iterativamente hasta convergencia o hasta que se alcancen las iteraciones máximas.
        Retorna self para encadenamiento de métodos.
        """
        self.centroids = self._initialize_centroids(X_train)
        
        for _ in range(self.max_iters):
            distances = self._compute_distances(X_train)
            self.labels_ = self._assign_clusters(distances)
            has_converged = self._update_centroids(X_train)
            
            if has_converged:
                break
        
        return self
    
    def predict(self, X):
        """
        Predice asignaciones de clusters para nuevos puntos de datos usando los
        centroides entrenados. Lanza error si se llama antes de ajustar el modelo.
        """
        if self.centroids is None:
            raise ValueError("El modelo aún no ha sido ajustado.")
        
        distances = self._compute_distances(X)
        return self._assign_clusters(distances)
    
    def fit_predict(self, X_train):
        """
        Método de conveniencia que realiza ajuste y predicción en un paso.
        Equivalente a llamar fit() seguido de predict() en los mismos datos.
        """
        return self.fit(X_train).predict(X_train)