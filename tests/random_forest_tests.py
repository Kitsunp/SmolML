import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from smolml.core.ml_array import MLArray
from smolml.models.tree.random_forest import RandomForest

class TestRandomForest(unittest.TestCase):
    def setUp(self):
        """
        Configurar un subconjunto pequeño del conjunto de datos Iris para pruebas
        """
        # Cargar conjunto de datos iris
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Tomar un subconjunto pequeño (150 muestras) manteniendo distribución de clases
        indices = []
        for class_idx in range(3):
            class_indices = [i for i, label in enumerate(y) if label == class_idx]
            indices.extend(class_indices[:50])
        
        self.X = X[indices]
        self.y = y[indices]
        self.feature_names = iris.feature_names
        self.class_names = iris.target_names
        
        # Dividir en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.33, random_state=42
        )
        
        # Convertir a listas para MLArray
        self.X_train = MLArray([[float(x) for x in row] for row in X_train])
        self.y_train = MLArray([float(y) for y in y_train])
        self.X_test = MLArray([[float(x) for x in row] for row in X_test])
        self.y_test = MLArray([float(y) for y in y_test])

    def test_iris_classification(self):
        """
        Probar clasificación de bosque aleatorio en conjunto de datos Iris
        """
        print("\nProbando Bosque Aleatorio en Conjunto de Datos Iris...")
        print(f"Muestras de entrenamiento: {len(self.X_train.data)}")
        print(f"Muestras de prueba: {len(self.X_test.data)}")
        print(f"Características usadas: {self.feature_names}")
        
        configs = [
            {"n_trees": 5, "max_depth": 3},
            {"n_trees": 10, "max_depth": 5},
            {"n_trees": 50, "max_depth": 15}
        ]
        
        for config in configs:
            print(f"\nProbando con {config['n_trees']} árboles, max_depth={config['max_depth']}")
            
            rf = RandomForest(
                n_trees=config['n_trees'],
                max_depth=config['max_depth'],
                min_samples_split=2,
                task="classification"
            )
            
            # Entrenar bosque
            rf.fit(self.X_train, self.y_train)

            print(rf)
            
            # Hacer predicciones
            y_pred = rf.predict(self.X_test)
            
            # Calcular precisión
            correct = sum(1 for pred, true in zip(y_pred.data, self.y_test.data) 
                        if pred == true)
            accuracy = correct / len(self.y_test.data)
            
            print(f"Precisión de Clasificación: {accuracy:.3f}")
            
            # Imprimir predicciones detalladas vs valores verdaderos
            print("\nPredicciones vs Valores Verdaderos:")
            for pred, true in zip(y_pred.data, self.y_test.data):
                pred_val = pred.data if hasattr(pred, 'data') else pred
                true_val = true.data if hasattr(true, 'data') else true
                print(f"Predicho: {self.class_names[int(pred_val)]}, "
                      f"Verdadero: {self.class_names[int(true_val)]}")
            
            self.assertGreaterEqual(accuracy, 0.6,
                f"Precisión de clasificación ({accuracy:.3f}) muy baja para {config['n_trees']} árboles")

if __name__ == '__main__':
    unittest.main()