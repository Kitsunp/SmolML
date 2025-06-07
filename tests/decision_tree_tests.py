import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from smolml.core.ml_array import MLArray
from smolml.models.tree.decision_tree import DecisionTree

class TestDecisionTree(unittest.TestCase):
    """
    Probar implementación de árbol de decisión usando conjuntos de datos sklearn
    """
    
    def setUp(self):
        """
        Configurar parámetros comunes y cargar conjuntos de datos
        """
        np.random.seed(42)
        
        # Cargar y preparar conjunto de datos iris para clasificación
        iris = load_iris()
        X_iris, y_iris = iris.data, iris.target
        
        # Usar solo dos características para visualización
        self.X_iris = X_iris[:, [0, 1]]  # longitud y ancho del sépalo
        self.y_iris = y_iris
        self.feature_names_iris = [iris.feature_names[0], iris.feature_names[1]]
        self.class_names_iris = iris.target_names
        
        # Dividir datos iris
        self.X_iris_train, self.X_iris_test, self.y_iris_train, self.y_iris_test = train_test_split(
            self.X_iris, self.y_iris, test_size=0.2, random_state=42
        )
        
        # Cargar y preparar conjunto de datos diabetes para regresión
        diabetes = load_diabetes()
        X_diabetes, y_diabetes = diabetes.data, diabetes.target
        
        # Escalar los datos
        scaler = StandardScaler()
        X_diabetes = scaler.fit_transform(X_diabetes)
        y_diabetes = (y_diabetes - y_diabetes.mean()) / y_diabetes.std()
        
        # Usar solo dos características para visualización
        self.X_diabetes = X_diabetes[:, [0, 2]]  # edad e imc
        self.y_diabetes = y_diabetes
        self.feature_names_diabetes = [diabetes.feature_names[0], diabetes.feature_names[2]]
        
        # Dividir datos diabetes
        self.X_diabetes_train, self.X_diabetes_test, self.y_diabetes_train, self.y_diabetes_test = train_test_split(
            self.X_diabetes, self.y_diabetes, test_size=0.2, random_state=42
        )

    def plot_decision_boundary(self, X, y, tree, title, feature_names, class_names=None):
        """
        Graficar frontera de decisión para clasificación o predicciones codificadas por color para regresión
        """
        plt.figure(figsize=(10, 8))
        
        # Crear malla
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))
        
        # Convertir puntos de malla a formato lista para MLArray
        mesh_points = [[float(x), float(y)] for x, y in zip(xx.ravel(), yy.ravel())]
        
        # Hacer predicciones
        Z = tree.predict(MLArray(mesh_points)).to_list()
        Z = np.array(Z).reshape(xx.shape)
        
        if class_names is not None:  # Clasificación
            # Graficar frontera de decisión
            plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
            scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='black')
            plt.colorbar(scatter)
            
            # Agregar leyenda
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=plt.cm.viridis(i/2.), 
                                        label=class_names[i], markersize=10)
                             for i in range(3)]
            plt.legend(handles=legend_elements)
        else:  # Regresión
            # Graficar predicciones
            plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
            scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
            plt.colorbar(scatter)
        
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title(title)
        
        # Guardar gráfico
        plot_type = 'clasificacion' if class_names is not None else 'regresion'
        plt.savefig(f'arbol_decision_{plot_type}.png')
        plt.close()

    def test_classification(self):
        """
        Probar clasificación de árbol de decisión en conjunto de datos iris
        """
        print("\nProbando Clasificación en Conjunto de Datos Iris...")
        
        # Convertir arrays numpy a listas para MLArray
        X_train_list = [[float(x) for x in row] for row in self.X_iris_train]
        y_train_list = [float(y) for y in self.y_iris_train]
        X_test_list = [[float(x) for x in row] for row in self.X_iris_test]
        
        # Crear y entrenar árbol
        clf = DecisionTree(max_depth=10, min_samples_split=5, task="classification")
        clf.fit(MLArray(X_train_list), MLArray(y_train_list))
        print(clf)
        
        # Hacer predicciones
        y_pred = clf.predict(MLArray(X_test_list))
        y_pred = np.array(y_pred.to_list())
        
        # Calcular precisión
        accuracy = np.mean(y_pred == self.y_iris_test)
        print(f"Precisión de Clasificación: {accuracy:.3f}")
        
        # Preparar datos para graficar
        X_plot_list = [[float(x) for x in row] for row in self.X_iris]
        y_plot_list = [float(y) for y in self.y_iris]
        
        # Graficar frontera de decisión
        self.plot_decision_boundary(
            self.X_iris, self.y_iris, clf,
            "Frontera de Decisión de Clasificación Iris",
            self.feature_names_iris, self.class_names_iris
        )
        
        # Aserciones
        self.assertGreater(accuracy, 0.7, "La precisión de clasificación debería ser > 70%")

    def test_regression(self):
        """
        Probar regresión de árbol de decisión en conjunto de datos diabetes
        """
        print("\nProbando Regresión en Conjunto de Datos Diabetes...")
        
        # Convertir arrays numpy a listas para MLArray
        X_train_list = [[float(x) for x in row] for row in self.X_diabetes_train]
        y_train_list = [float(y) for y in self.y_diabetes_train]
        X_test_list = [[float(x) for x in row] for row in self.X_diabetes_test]
        
        # Crear y entrenar árbol
        reg = DecisionTree(max_depth=5, min_samples_split=5, task="regression")
        reg.fit(MLArray(X_train_list), MLArray(y_train_list))
        print(reg)
        
        # Hacer predicciones
        y_pred = reg.predict(MLArray(X_test_list))
        y_pred = np.array(y_pred.to_list())
        
        # Calcular MSE
        mse = np.mean((y_pred - self.y_diabetes_test) ** 2)
        print(f"MSE de Regresión: {mse:.3f}")
        
        # Preparar datos para graficar
        X_plot_list = [[float(x) for x in row] for row in self.X_diabetes]
        y_plot_list = [float(y) for y in self.y_diabetes]
        
        # Graficar predicciones
        self.plot_decision_boundary(
            self.X_diabetes, self.y_diabetes, reg,
            "Predicciones de Regresión Diabetes",
            self.feature_names_diabetes
        )
        
        # Aserciones
        self.assertLess(mse, 1.0, "MSE debería ser < 1.0 para datos escalados")

    def test_edge_cases(self):
        """
        Probar casos extremos y validación de parámetros
        """
        # Probar con muestras mínimas de hoja
        tree = DecisionTree(max_depth=2, min_samples_leaf=1)
        tree.fit(MLArray([[1], [2]]), MLArray([1, 2]))
        self.assertIsNotNone(tree.root)
        
        # Probar con clase única
        tree = DecisionTree()
        tree.fit(MLArray([[1], [2]]), MLArray([1, 1]))
        self.assertEqual(tree.predict(MLArray([[1.5]])).to_list()[0], 1)

if __name__ == '__main__':
    unittest.main()