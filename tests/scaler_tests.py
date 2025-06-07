import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler as SKStandardScaler
from sklearn.preprocessing import MinMaxScaler as SKMinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from smolml.core.ml_array import MLArray
from smolml.preprocessing.scalers import StandardScaler, MinMaxScaler

class TestScalers(unittest.TestCase):
    """
    Probar implementaciones de escaladores personalizados contra scikit-learn
    """
    
    def setUp(self):
        """
        Configurar datos de prueba e inicializar escaladores
        """
        # Datos de prueba simples
        self.simple_data = [
            [1, 4], 
            [100, 2], 
            [-20, 2],
            [0, 8],
            [50, -4]
        ]
        
        # Datos de casos extremos
        self.edge_data = [
            [0, 0],  # ceros
            [1e6, 1e-6],  # valores muy grandes/pequeños
            [-1e6, -1e-6],  # valores grandes/pequeños negativos
            [1, 1],  # mismos valores
            [-1, -1]  # mismos valores negativos
        ]
        
        # Convertir a formatos apropiados
        self.simple_ml = MLArray(self.simple_data)
        self.simple_np = np.array(self.simple_data)
        self.edge_ml = MLArray(self.edge_data)
        self.edge_np = np.array(self.edge_data)
        
        # Inicializar todos los escaladores
        self.standard_ml = StandardScaler()
        self.standard_sk = SKStandardScaler()
        self.minmax_ml = MinMaxScaler()
        self.minmax_sk = SKMinMaxScaler()

    def plot_comparison(self, original_data, ml_scaled, sk_scaled, title, filename):
        """
        Crear visualización comparando datos originales y escalados
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Datos originales
        ax1.scatter(original_data[:, 0], original_data[:, 1], c='blue', alpha=0.6)
        ax1.set_title('Datos Originales')
        ax1.grid(True)
        
        # Implementación personalizada
        ax2.scatter(ml_scaled[:, 0], ml_scaled[:, 1], c='red', alpha=0.6)
        ax2.set_title('Escalador Personalizado')
        ax2.grid(True)
        
        # Implementación scikit-learn
        ax3.scatter(sk_scaled[:, 0], sk_scaled[:, 1], c='green', alpha=0.6)
        ax3.set_title('Escalador Scikit-learn')
        ax3.grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def test_standard_scaler_simple(self):
        """Probar StandardScaler con datos simples"""
        print("\nProbando StandardScaler con datos simples...")
        
        # Ajustar y transformar usando ambas implementaciones
        ml_scaled = self.standard_ml.fit_transform(self.simple_ml)
        sk_scaled = self.standard_sk.fit_transform(self.simple_np)
        
        # Convertir MLArray a numpy para comparación
        ml_scaled_np = np.array(ml_scaled.to_list())
        
        # Graficar comparación
        self.plot_comparison(
            self.simple_np, 
            ml_scaled_np, 
            sk_scaled,
            'Comparación StandardScaler - Datos Simples',
            'standard_scaler_simple.png'
        )
        
        # Pruebas estadísticas básicas
        print("\nEstadísticas del Escalador Estándar (Datos Simples):")
        print("Implementación Personalizada:")
        print(f"Media: {ml_scaled_np.mean(axis=0)}")
        print(f"Std: {ml_scaled_np.std(axis=0)}")
        print("\nImplementación Scikit-learn:")
        print(f"Media: {sk_scaled.mean(axis=0)}")
        print(f"Std: {sk_scaled.std(axis=0)}")
        
        # Aserciones
        np.testing.assert_array_almost_equal(
            ml_scaled_np.mean(axis=0), 
            np.zeros_like(ml_scaled_np.mean(axis=0)), 
            decimal=10
        )
        np.testing.assert_array_almost_equal(
            ml_scaled_np.std(axis=0), 
            np.ones_like(ml_scaled_np.std(axis=0)), 
            decimal=10
        )
        np.testing.assert_array_almost_equal(ml_scaled_np, sk_scaled, decimal=10)

    def test_standard_scaler_edge(self):
        """Probar StandardScaler con casos extremos"""
        print("\nProbando StandardScaler con casos extremos...")
        
        # Ajustar y transformar usando ambas implementaciones
        ml_scaled = self.standard_ml.fit_transform(self.edge_ml)
        sk_scaled = self.standard_sk.fit_transform(self.edge_np)
        
        # Convertir MLArray a numpy para comparación
        ml_scaled_np = np.array(ml_scaled.to_list())
        
        # Graficar comparación
        self.plot_comparison(
            self.edge_np, 
            ml_scaled_np, 
            sk_scaled,
            'Comparación StandardScaler - Casos Extremos',
            'standard_scaler_edge.png'
        )
        
        # Aserciones para casos extremos
        np.testing.assert_array_almost_equal(ml_scaled_np, sk_scaled, decimal=10)

    def test_minmax_scaler_simple(self):
        """Probar MinMaxScaler con datos simples"""
        print("\nProbando MinMaxScaler con datos simples...")
        
        # Ajustar y transformar usando ambas implementaciones
        ml_scaled = self.minmax_ml.fit_transform(self.simple_ml)
        sk_scaled = self.minmax_sk.fit_transform(self.simple_np)
        
        # Convertir MLArray a numpy para comparación
        ml_scaled_np = np.array(ml_scaled.to_list())
        
        # Graficar comparación
        self.plot_comparison(
            self.simple_np, 
            ml_scaled_np, 
            sk_scaled,
            'Comparación MinMaxScaler - Datos Simples',
            'minmax_scaler_simple.png'
        )
        
        # Pruebas de rango básicas
        print("\nEstadísticas del Escalador MinMax (Datos Simples):")
        print("Implementación Personalizada:")
        print(f"Min: {ml_scaled_np.min(axis=0)}")
        print(f"Max: {ml_scaled_np.max(axis=0)}")
        print("\nImplementación Scikit-learn:")
        print(f"Min: {sk_scaled.min(axis=0)}")
        print(f"Max: {sk_scaled.max(axis=0)}")
        
        # Aserciones
        np.testing.assert_array_almost_equal(
            ml_scaled_np.min(axis=0), 
            np.zeros_like(ml_scaled_np.min(axis=0)), 
            decimal=10
        )
        np.testing.assert_array_almost_equal(
            ml_scaled_np.max(axis=0), 
            np.ones_like(ml_scaled_np.max(axis=0)), 
            decimal=10
        )
        np.testing.assert_array_almost_equal(ml_scaled_np, sk_scaled, decimal=10)

    def test_minmax_scaler_edge(self):
        """Probar MinMaxScaler con casos extremos"""
        print("\nProbando MinMaxScaler con casos extremos...")
        
        # Ajustar y transformar usando ambas implementaciones
        ml_scaled = self.minmax_ml.fit_transform(self.edge_ml)
        sk_scaled = self.minmax_sk.fit_transform(self.edge_np)
        
        # Convertir MLArray a numpy para comparación
        ml_scaled_np = np.array(ml_scaled.to_list())
        
        # Graficar comparación
        self.plot_comparison(
            self.edge_np, 
            ml_scaled_np, 
            sk_scaled,
            'Comparación MinMaxScaler - Casos Extremos',
            'minmax_scaler_edge.png'
        )
        
        # Aserciones para casos extremos
        np.testing.assert_array_almost_equal(ml_scaled_np, sk_scaled, decimal=10)

    def test_single_value(self):
        """Probar escaladores con valor único"""
        single_data_ml = MLArray([[1.0, 2.0]])
        single_data_np = np.array([[1.0, 2.0]])
        
        # Probar StandardScaler
        std_ml = self.standard_ml.fit_transform(single_data_ml)
        std_sk = self.standard_sk.fit_transform(single_data_np)
        
        # Probar MinMaxScaler
        minmax_ml = self.minmax_ml.fit_transform(single_data_ml)
        minmax_sk = self.minmax_sk.fit_transform(single_data_np)
        
        # Convertir a numpy para comparación
        std_ml_np = np.array(std_ml.to_list())
        minmax_ml_np = np.array(minmax_ml.to_list())
        
        # Estos deberían ser 0 o NaN para ambas implementaciones
        np.testing.assert_array_almost_equal(std_ml_np, std_sk, decimal=10)
        np.testing.assert_array_almost_equal(minmax_ml_np, minmax_sk, decimal=10)

if __name__ == '__main__':
    unittest.main()