import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans as SKLearnKMeans
import matplotlib.pyplot as plt
from smolml.core.ml_array import MLArray
from smolml.models.unsupervised.kmeans import KMeans

class TestKMeansVsSklearn(unittest.TestCase):
    """
    Comparar implementación K-Means personalizada contra scikit-learn
    usando datos sintéticos agrupados
    """
    
    def setUp(self):
        """
        Configurar conjunto de datos y modelos
        """
        # Establecer semilla aleatoria para reproducibilidad
        np.random.seed(42)
        
        # Generar datos agrupados sintéticos
        n_samples = 300
        self.n_clusters = 3
        X, y = make_blobs(n_samples=n_samples, 
                         centers=self.n_clusters,
                         cluster_std=1.0,
                         random_state=42)
        
        # Escalar los datos
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        
        # Almacenar los datos
        self.X = X
        self.y = y
        
        # Convertir datos para implementación personalizada
        self.X_ml = MLArray([[float(x) for x in row] for row in self.X])
        
        # Inicializar modelos
        self.custom_kmeans = KMeans(n_clusters=self.n_clusters, 
                                  max_iters=100, 
                                  tol=1e-4)
        self.sklearn_kmeans = SKLearnKMeans(n_clusters=self.n_clusters,
                                          max_iter=100,
                                          tol=1e-4,
                                          random_state=42)

    def _plot_clusters(self, custom_labels, sklearn_labels):
        """
        Graficar resultados de clustering de ambas implementaciones
        """
        plt.figure(figsize=(12, 5))
        
        # Graficar resultados de implementación personalizada
        plt.subplot(1, 2, 1)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=custom_labels.to_list(), 
                   cmap='viridis', alpha=0.6)
        custom_centroids = self.custom_kmeans.centroids.to_list()
        plt.scatter(np.array(custom_centroids)[:, 0], 
                   np.array(custom_centroids)[:, 1], 
                   c='red', marker='x', s=200, linewidth=3, 
                   label='Centroides')
        plt.title('Clustering K-Means Personalizado')
        plt.legend()
        
        # Graficar resultados scikit-learn
        plt.subplot(1, 2, 2)
        plt.scatter(self.X[:, 0], self.X[:, 1], c=sklearn_labels, 
                   cmap='viridis', alpha=0.6)
        plt.scatter(self.sklearn_kmeans.cluster_centers_[:, 0],
                   self.sklearn_kmeans.cluster_centers_[:, 1],
                   c='red', marker='x', s=200, linewidth=3,
                   label='Centroides')
        plt.title('Clustering K-Means Scikit-learn')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('comparacion_kmeans.png')
        plt.close()

    def _compute_inertia(self, X, labels, centroids):
        """
        Calcular inercia (suma de cuadrados intra-cluster)
        """
        inertia = 0
        X_list = X.to_list() if isinstance(X, MLArray) else X
        centroids_list = centroids.to_list() if isinstance(centroids, MLArray) else centroids
        labels_list = labels.to_list() if isinstance(labels, MLArray) else labels
        
        for i, point in enumerate(X_list):
            centroid = centroids_list[labels_list[i]]
            diff = np.array(point) - np.array(centroid)
            inertia += np.sum(diff ** 2)
            
        return inertia

    def _plot_training_progress(self):
        """
        Visualiza el progreso de entrenamiento mostrando cómo se movieron los centroides durante el entrenamiento
        """
        n_plots = min(5, len(self.custom_kmeans.centroid_history))
        fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))
        
        # Si solo tenemos un subplot, envolverlo en una lista
        if n_plots == 1:
            axes = [axes]
        
        # Seleccionar iteraciones para graficar
        if len(self.custom_kmeans.centroid_history) <= n_plots:
            plot_iterations = range(len(self.custom_kmeans.centroid_history))
        else:
            # Seleccionar iteraciones espaciadas uniformemente incluyendo primera y última
            plot_iterations = np.linspace(0, len(self.custom_kmeans.centroid_history)-1, n_plots, dtype=int)
        
        # Convertir datos a numpy para graficar más fácil
        X_array = np.array(self.X_ml.to_list())
        
        # Graficar cada iteración seleccionada
        for idx, iter_idx in enumerate(plot_iterations):
            ax = axes[idx]
            centroids = np.array(self.custom_kmeans.centroid_history[iter_idx])
            
            # Graficar puntos de datos
            if iter_idx == len(self.custom_kmeans.centroid_history) - 1:
                # Para la iteración final, colorear puntos por cluster
                labels = np.array(self.custom_kmeans.labels_.to_list())
                scatter = ax.scatter(X_array[:, 0], X_array[:, 1], c=labels, 
                                cmap='viridis', alpha=0.6, s=50)
            else:
                # Para iteraciones anteriores, mostrar todos los puntos en gris
                ax.scatter(X_array[:, 0], X_array[:, 1], c='grey', 
                        alpha=0.3, s=50)
            
            # Graficar centroides
            ax.scatter(centroids[:, 0], centroids[:, 1], c='red', 
                    marker='x', s=200, linewidth=3, label='Centroides')
            
            # Agregar número de iteración
            ax.set_title(f'Iteración {iter_idx}')
            
            # Si es el primer subplot, agregar leyenda
            if idx == 0:
                ax.legend()
        
        plt.tight_layout()
        plt.savefig('progreso_entrenamiento_kmeans.png')
        plt.close()

    def test_compare_clustering(self):
        """
        Entrenar y comparar ambas implementaciones
        """
        print("\nAjustando K-Means personalizado...")
        custom_labels = self.custom_kmeans.fit_predict(self.X_ml)
        
        # Graficar progreso de entrenamiento
        self._plot_training_progress()
        
        print("Ajustando K-Means scikit-learn...")
        sklearn_labels = self.sklearn_kmeans.fit_predict(self.X)
        
        # Graficar resultados de clustering
        self._plot_clusters(custom_labels, sklearn_labels)

if __name__ == '__main__':
    unittest.main()