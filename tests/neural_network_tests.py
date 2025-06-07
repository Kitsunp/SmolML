import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from smolml.core.ml_array import MLArray
from smolml.models.nn.neural_network import NeuralNetwork
from smolml.models.nn.layer import DenseLayer
import smolml.utils.activation as activation
import smolml.utils.losses as losses
import smolml.utils.optimizers as optimizers

class TestNeuralNetworkVsTensorflow(unittest.TestCase):
    """
    Comparar implementación de red neuronal personalizada contra TensorFlow
    usando el conjunto de datos make_moons
    """
    
    def setUp(self):
        """
        Configurar conjunto de datos y modelos
        """
        # Establecer semillas aleatorias para reproducibilidad
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Generar conjunto de datos luna
        X, y = make_moons(n_samples=100, noise=0.1)
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        
        # Dividir datos
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Convertir datos para implementación personalizada
        self.X_train_ml = MLArray([[float(x) for x in row] for row in self.X_train])
        self.y_train_ml = MLArray([[float(y)] for y in self.y_train])
        self.X_test_ml = MLArray([[float(x) for x in row] for row in self.X_test])
        self.y_test_ml = MLArray([[float(y)] for y in self.y_test])
        
        # Parámetros del modelo
        self.input_size = 2
        self.hidden_size = 32
        self.output_size = 1
        self.epochs = 100
        self.learning_rate = 0.05
        
        # Inicializar modelos
        self.custom_model = self._create_custom_model()
        self.tf_model = self._create_tf_model()

    def _create_custom_model(self):
        """
        Crear red neuronal personalizada con la misma arquitectura
        """
        return NeuralNetwork([
            DenseLayer(self.input_size, self.hidden_size, activation.relu),
            DenseLayer(self.hidden_size, self.output_size, activation.sigmoid)
        ], losses.binary_cross_entropy, optimizer=optimizers.SGDMomentum(learning_rate=0.1))

    def _create_tf_model(self):
        """
        Crear modelo TensorFlow equivalente
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_size, activation='relu', input_shape=(self.input_size,)),
            tf.keras.layers.Dense(self.output_size, activation='sigmoid')
        ])
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        return model

    def _plot_decision_boundary(self, model, is_tf=False):
        """
        Graficar frontera de decisión para cualquier modelo
        """
        x_min, x_max = self.X_test[:, 0].min() - 0.5, self.X_test[:, 0].max() + 0.5
        y_min, y_max = self.X_test[:, 1].min() - 0.5, self.X_test[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        # Obtener predicciones para puntos de malla
        if is_tf:
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        else:
            X_mesh = MLArray([[float(x), float(y)] for x, y in zip(xx.ravel(), yy.ravel())])
            Z = model.forward(X_mesh).to_list()
        Z = np.array(Z).reshape(xx.shape)
        
        return xx, yy, Z

    def test_compare_models(self):
        """
        Entrenar y comparar ambos modelos
        """
        print("IMPORTANTE: La implementación SmolML de RNs es muy ineficiente debido a estar escrita en Python. Ejecutar esto en una máquina de bajo hardware podría tomar mucho tiempo.")
        # Entrenar modelo personalizado
        print("\nEntrenando modelo personalizado...")
        custom_history = []
        for epoch in range(self.epochs):
            y_pred = self.custom_model.forward(self.X_train_ml)
            loss = self.custom_model.loss_function(y_pred, self.y_train_ml)
            loss.backward()
            
            for idx, layer in enumerate(self.custom_model.layers):
                layer.update(self.custom_model.optimizer, idx)
            
            # Reiniciar grafo computacional
            self.X_train_ml = self.X_train_ml.restart()
            self.y_train_ml = self.y_train_ml.restart()
            for layer in self.custom_model.layers:
                layer.weights = layer.weights.restart()
                layer.biases = layer.biases.restart()
            
            custom_history.append(float(loss.data.data))
            print(f"Época {epoch + 1}/{self.epochs}, Pérdida: {loss.data.data}")
        
        # Entrenar modelo TensorFlow
        print("\nEntrenando modelo TensorFlow...")
        tf_history = self.tf_model.fit(
            self.X_train, self.y_train,
            epochs=self.epochs,
            batch_size=len(self.X_train),
            verbose=1
        )
        
        # Graficar curvas de entrenamiento
        plt.figure(figsize=(12, 4))

        print("\n Graficando pérdida de entrenamiento...")
        # Gráfico 1: Pérdida de Entrenamiento
        plt.subplot(1, 2, 1)
        plt.plot(range(self.epochs), custom_history, label='RN Personalizada')
        plt.plot(range(self.epochs), tf_history.history['loss'], label='TensorFlow')
        plt.title('Pérdida de Entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()
        
        # Gráfico 2: Fronteras de Decisión
        plt.subplot(1, 2, 2)
        
        print("\n Graficando fronteras de decisión...")
        # Graficar fronteras de decisión
        xx, yy, Z_custom = self._plot_decision_boundary(self.custom_model)
        plt.contourf(xx, yy, Z_custom > 0.5, alpha=0.4)
        plt.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, alpha=0.8)
        plt.title('Fronteras de Decisión (RN Personalizada)')
        
        plt.tight_layout()
        plt.savefig('comparacion_red_neuronal.png')
        plt.close()
        
        # Calcular e imprimir precisiones
        custom_pred = np.array(self.custom_model.forward(self.X_test_ml).to_list()) > 0.5
        tf_pred = self.tf_model.predict(self.X_test) > 0.5
        
        custom_accuracy = np.mean(custom_pred.flatten() == self.y_test)
        tf_accuracy = np.mean(tf_pred.flatten() == self.y_test)
        
        print("\nPrecisiones de Prueba:")
        print(f"RN Personalizada: {custom_accuracy:.4f}")
        print(f"TensorFlow: {tf_accuracy:.4f}")
        
        # Asegurar que los modelos alcancen precisión razonable
        self.assertGreater(custom_accuracy, 0.8, "La precisión del modelo personalizado debería ser > 80%")
        self.assertGreater(tf_accuracy, 0.8, "La precisión del modelo TensorFlow debería ser > 80%")

if __name__ == '__main__':
    unittest.main()