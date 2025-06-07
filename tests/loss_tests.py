import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from smolml.core.ml_array import MLArray
import smolml.utils.losses as losses

class TestLossFunctions(unittest.TestCase):
    """
    Probar funciones de pérdida personalizadas contra implementaciones TensorFlow
    """
    
    def setUp(self):
        """
        Configurar datos de prueba y configurar TensorFlow
        """
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Datos de regresión (para MSE, MAE, Huber)
        self.y_true_reg = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        self.y_pred_reg = [[1.2], [1.8], [3.3], [3.9], [5.2]]
        
        # Datos de clasificación binaria (para BCE)
        self.y_true_bin = [[0], [1], [1], [0], [1]]
        self.y_pred_bin = [[0.2], [0.8], [0.7], [0.1], [0.9]]
        
        # Datos de clasificación multi-clase (para CCE)
        self.y_true_cat = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ]
        self.y_pred_cat = [
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.2, 0.6],
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2]
        ]
        
        # Convertir a formato MLArray
        self.ml_true_reg = MLArray(self.y_true_reg)
        self.ml_pred_reg = MLArray(self.y_pred_reg)
        self.ml_true_bin = MLArray(self.y_true_bin)
        self.ml_pred_bin = MLArray(self.y_pred_bin)
        self.ml_true_cat = MLArray(self.y_true_cat)
        self.ml_pred_cat = MLArray(self.y_pred_cat)
        
        # Convertir a formato TensorFlow
        self.tf_true_reg = tf.constant(self.y_true_reg, dtype=tf.float32)
        self.tf_pred_reg = tf.constant(self.y_pred_reg, dtype=tf.float32)
        self.tf_true_bin = tf.constant(self.y_true_bin, dtype=tf.float32)
        self.tf_pred_bin = tf.constant(self.y_pred_bin, dtype=tf.float32)
        self.tf_true_cat = tf.constant(self.y_true_cat, dtype=tf.float32)
        self.tf_pred_cat = tf.constant(self.y_pred_cat, dtype=tf.float32)

    def test_losses(self):
        """
        Probar todas las funciones de pérdida y crear visualización
        """
        # Crear figura
        fig = plt.figure(figsize=(20, 15))
        plt.suptitle('Comparaciones de Funciones de Pérdida', fontsize=16, y=0.95)
        
        # Calcular todas las pérdidas
        # 1. Pérdida MSE
        custom_mse = float(losses.mse_loss(self.ml_pred_reg, self.ml_true_reg).data.data)
        mse = tf.keras.losses.MeanSquaredError()
        tf_mse = float(mse(self.tf_true_reg, self.tf_pred_reg).numpy())
        print(f"\nMSE - Personalizado: {custom_mse:.6f}, TF: {tf_mse:.6f}")
        np.testing.assert_allclose(custom_mse, tf_mse, rtol=1e-5)
        
        # 2. Pérdida MAE
        custom_mae = float(losses.mae_loss(self.ml_pred_reg, self.ml_true_reg).data.data)
        mae = tf.keras.losses.MeanAbsoluteError()
        tf_mae = float(mae(self.tf_true_reg, self.tf_pred_reg).numpy())
        print(f"MAE - Personalizado: {custom_mae:.6f}, TF: {tf_mae:.6f}")
        np.testing.assert_allclose(custom_mae, tf_mae, rtol=1e-5)
        
        # 3. Entropía Cruzada Binaria
        custom_bce = float(losses.binary_cross_entropy(self.ml_pred_bin, self.ml_true_bin).data.data)
        bce = tf.keras.losses.BinaryCrossentropy()
        tf_bce = float(bce(self.tf_true_bin, self.tf_pred_bin).numpy())
        print(f"BCE - Personalizado: {custom_bce:.6f}, TF: {tf_bce:.6f}")
        np.testing.assert_allclose(custom_bce, tf_bce, rtol=1e-5)
        
        # 4. Entropía Cruzada Categórica
        custom_cce = float(losses.categorical_cross_entropy(self.ml_pred_cat, self.ml_true_cat).data.data)
        cce = tf.keras.losses.CategoricalCrossentropy()
        tf_cce = float(cce(self.tf_true_cat, self.tf_pred_cat).numpy())
        print(f"CCE - Personalizado: {custom_cce:.6f}, TF: {tf_cce:.6f}")
        np.testing.assert_allclose(custom_cce, tf_cce, rtol=1e-5)
        
        # 5. Pérdida Huber
        delta = 1.0
        custom_huber = float(losses.huber_loss(self.ml_pred_reg, self.ml_true_reg, delta).data.data)
        huber = tf.keras.losses.Huber(delta=delta)
        tf_huber = float(huber(self.tf_true_reg, self.tf_pred_reg).numpy())
        print(f"Huber - Personalizado: {custom_huber:.6f}, TF: {tf_huber:.6f}")
        np.testing.assert_allclose(custom_huber, tf_huber, rtol=1e-5)

        # Graficar
        # 1. Predicciones de regresión (superior izquierda)
        ax1 = plt.subplot(3, 2, 1)
        plt.scatter([y[0] for y in self.y_true_reg], [y[0] for y in self.y_pred_reg],
                   c='blue', alpha=0.6, label='Predicciones')
        plt.plot([min(self.y_true_reg), max(self.y_true_reg)],
                [min(self.y_true_reg), max(self.y_true_reg)],
                'r--', label='Predicción Perfecta')
        plt.xlabel('Valores Verdaderos')
        plt.ylabel('Valores Predichos')
        plt.title('Predicciones de Regresión vs Verdad')
        plt.legend()

        # 2. Comparación de Pérdidas de Regresión (superior derecha)
        ax2 = plt.subplot(3, 2, 2)
        width = 0.35
        x = np.arange(3)
        plt.bar(x - width/2, [custom_mse, custom_mae, custom_huber],
                width, label='Personalizado', color='blue', alpha=0.6)
        plt.bar(x + width/2, [tf_mse, tf_mae, tf_huber],
                width, label='TensorFlow', color='green', alpha=0.6)
        plt.xticks(x, ['MSE', 'MAE', 'Huber'])
        plt.title('Valores de Pérdida de Regresión')
        plt.legend()

        # 3. Entropía Cruzada Binaria (medio izquierda)
        ax3 = plt.subplot(3, 2, 3)
        x = np.arange(len(self.y_true_bin))
        width = 0.35
        plt.bar(x - width/2, [p[0] for p in self.y_pred_bin],
                width, label='Predicho', color='blue', alpha=0.6)
        plt.bar(x + width/2, [t[0] for t in self.y_true_bin],
                width, label='Verdadero', color='green', alpha=0.6)
        plt.xlabel('Muestra')
        plt.ylabel('Probabilidad')
        plt.title(f'Entropía Cruzada Binaria\nPersonalizado: {custom_bce:.4f}, TF: {tf_bce:.4f}')
        plt.legend()

        # 4. Entropía Cruzada Categórica (medio derecha)
        ax4 = plt.subplot(3, 2, 4)
        num_samples = len(self.y_true_cat)
        x = np.arange(num_samples)
        width = 0.35

        # Valores verdaderos a la izquierda, predichos a la derecha para cada muestra
        for i in range(num_samples):
            plt.bar(x[i] - width/2, self.y_true_cat[i],
                   width, label='Verdadero' if i == 0 else "", bottom=np.sum(self.y_true_cat[i][:i]),
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.6)
            plt.bar(x[i] + width/2, self.y_pred_cat[i],
                   width, label='Predicho' if i == 0 else "", bottom=np.sum(self.y_pred_cat[i][:i]),
                   color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.3)

        plt.xlabel('Muestra')
        plt.ylabel('Distribución de Clases')
        plt.title(f'Entropía Cruzada Categórica\nPersonalizado: {custom_cce:.4f}, TF: {tf_cce:.4f}')
        plt.legend()

        # 5. Comparación de Todas las Pérdidas (inferior)
        ax5 = plt.subplot(3, 2, (5, 6))
        losses_custom = [custom_mse, custom_mae, custom_huber, custom_bce, custom_cce]
        losses_tf = [tf_mse, tf_mae, tf_huber, tf_bce, tf_cce]
        x = np.arange(5)
        plt.bar(x - width/2, losses_custom, width, label='Personalizado', color='blue', alpha=0.6)
        plt.bar(x + width/2, losses_tf, width, label='TensorFlow', color='green', alpha=0.6)
        plt.xticks(x, ['MSE', 'MAE', 'Huber', 'BCE', 'CCE'])
        plt.title('Comparación de Todas las Funciones de Pérdida')
        plt.legend()

        plt.tight_layout()
        plt.savefig('comparacion_funciones_perdida.png', bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == '__main__':
    unittest.main()