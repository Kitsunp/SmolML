import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from smolml.core.ml_array import MLArray, randn, ones
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import smolml.utils.initializers as initializers
import smolml.utils.optimizers as optimizers
import smolml.utils.losses as losses
from smolml.models.regression import LinearRegression, PolynomialRegression

class TestRegressionVisualization(unittest.TestCase):
    """
    Probar y visualizar implementaciones de regresión lineal y polinomial
    con deslizador interactivo de épocas
    """
    
    def setUp(self):
        """
        Configurar parámetros comunes y estilo para pruebas
        """
        np.random.seed(42)
        
        # Parámetros de entrenamiento
        self.iterations = 100
        self.epochs_to_store = [0, 5, 10, 25, 50, 99]
        
        # Inicializar optimizador
        self.optimizer = optimizers.SGD(learning_rate=0.1)
        
        # Establecer estilo de gráfico
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['axes.facecolor'] = '#f0f0f0'
        plt.rcParams['axes.edgecolor'] = '#333333'
        plt.rcParams['axes.labelcolor'] = '#333333'
        plt.rcParams['xtick.color'] = '#333333'
        plt.rcParams['ytick.color'] = '#333333'
        plt.rcParams['grid.color'] = '#ffffff'
        plt.rcParams['grid.linestyle'] = '-'
        plt.rcParams['grid.linewidth'] = 1

    def generate_linear_data(self, size=25):
        """Generar datos con relación lineal más ruido"""
        X = randn(size, 1)
        y = X * 2 + 1 + randn(size, 1) * 0.1
        return X, y

    def generate_nonlinear_data(self, size=25):
        """Generar datos con relación polinomial más ruido"""
        X = randn(size, 1)
        y = X * 2 + X * X * 3 + 1 + randn(size, 1) * 0.1
        return X, y

    def train_and_visualize(self, model, X, y, title):
        """Entrenar modelo y crear visualización interactiva"""
        # Almacenar historial de predicciones
        predictions_history = []
        losses_history = []
        
        # Predicción inicial para almacenamiento
        y_pred = model.predict(X)
        predictions_history.append(y_pred.to_list())
        
        # Loop de entrenamiento usando método fit del modelo
        losses = model.fit(X, y, iterations=self.iterations, verbose=True, print_every=10)
        
        # Almacenar predicciones en épocas especificadas
        X_eval = X.restart()  # Crear copia fresca para evaluación
        for epoch in self.epochs_to_store[1:]:  # Saltar 0 ya que ya lo almacenamos
            y_pred = model.predict(X_eval)
            predictions_history.append(y_pred.to_list())
        
        # Convertir a numpy para graficar
        X_np = np.array(X.to_list())
        y_np = np.array(y.to_list())
        
        # Ordenar para graficar curva suave
        sort_idx = np.argsort(X_np.flatten())
        X_np = X_np[sort_idx]
        y_np = y_np[sort_idx]
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.25)
        
        scatter = ax.scatter(X_np, y_np, c='#1E88E5', alpha=0.6, label='Datos')
        predictions_sorted = [np.array(pred)[sort_idx] for pred in predictions_history]
        line, = ax.plot(X_np, predictions_sorted[0], color='#D81B60', lw=2, label='Predicción')
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True)
        
        # Agregar deslizador
        slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='#d3d3d3')
        slider = Slider(slider_ax, 'Época', 0, len(self.epochs_to_store) - 1,
                       valinit=0, valstep=1, color='#FFC107')
        
        def update(val):
            epoch_index = int(slider.val)
            line.set_ydata(predictions_sorted[epoch_index])
            fig.canvas.draw_idle()
        
        slider.on_changed(update)
        
        # Agregar texto de época
        epoch_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                           fontsize=12, fontweight='bold')
        
        def update_epoch_text(val):
            epoch_index = int(slider.val)
            epoch_text.set_text(f'Época: {self.epochs_to_store[epoch_index]}')
        
        slider.on_changed(update_epoch_text)
        update_epoch_text(0)  # Inicializar texto
        
        plt.show()
        
        # Imprimir parámetros finales
        print("\nParámetros Finales:")
        print("Pesos:", model.weights.data)
        print("Sesgo:", model.bias.data)
        
        return predictions_history, losses[-1]

    def test_linear_regression(self):
        """Probar regresión lineal con visualización"""
        print("\nProbando Regresión Lineal...")
        X, y = self.generate_linear_data()
        
        model = LinearRegression(
            input_size=1,
            loss_function=losses.mse_loss,
            optimizer=self.optimizer,
            initializer=initializers.XavierUniform()
        )

        print(model)
        predictions, final_loss = self.train_and_visualize(
            model, X, y, 'Regresión Lineal: Datos vs Predicciones'
        )
        
        # Aserciones básicas
        self.assertIsNotNone(predictions)
        self.assertGreater(len(predictions), 0)
        self.assertLess(final_loss, 1.0)  # Asumiendo convergencia

    def test_polynomial_regression(self):
        """Probar regresión polinomial con visualización"""
        print("\nProbando Regresión Polinomial...")
        X, y = self.generate_nonlinear_data()
        
        model = PolynomialRegression(
            input_size=1,
            degree=2,
            loss_function=losses.mse_loss,
            optimizer=self.optimizer,
            initializer=initializers.XavierUniform()
        )

        print(model)
        predictions, final_loss = self.train_and_visualize(
            model, X, y, 'Regresión Polinomial: Datos vs Predicciones'
        )
        
        # Aserciones básicas
        self.assertIsNotNone(predictions)
        self.assertGreater(len(predictions), 0)
        self.assertLess(final_loss, 1.0)  # Asumiendo convergencia

if __name__ == '__main__':
    unittest.main()