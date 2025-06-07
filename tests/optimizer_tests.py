import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from smolml.core.ml_array import MLArray, zeros
import smolml.utils.losses as losses
import smolml.utils.activation as activation
import smolml.utils.initializers as initializers
import smolml.utils.optimizers as optimizers
import random
import numpy as np
from smolml.models.nn.neural_network import NeuralNetwork
from smolml.models.nn.layer import DenseLayer

def create_network(optimizer):
    """Función auxiliar para crear una red con optimizador especificado"""
    input_size = 2
    hidden_size = 32
    output_size = 1
    
    return NeuralNetwork([
        DenseLayer(input_size, hidden_size, activation.relu),
        DenseLayer(hidden_size, output_size, activation.tanh)
    ], losses.mse_loss, optimizer)

def train_and_get_losses(network, X, y, epochs=100):
    """Entrenar red y retornar lista de pérdidas"""
    losses = []
    
    for epoch in range(epochs):
        # Paso hacia adelante
        y_pred = network.forward(X)
        loss = network.loss_function(y_pred, y)
        
        # Almacenar pérdida
        losses.append(loss.data.data)
        print(f"Época: {epoch+1}/{epochs} | Pérdida: {loss.data.data}")
        
        # Paso hacia atrás y actualización
        loss.backward()
        for idx, layer in enumerate(network.layers):
            layer.update(network.optimizer, idx)
        
        # Reiniciar gradientes
        X.restart()
        y.restart()
        for layer in network.layers:
            layer.weights.restart()
            layer.biases.restart()
    
    return losses

def compare_optimizers():
    # Establecer semilla aleatoria para reproducibilidad
    random.seed(42)
    np.random.seed(42)

    # Crear datos
    X = MLArray([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = MLArray([[0], [1], [1], [0]])

    # Crear redes con diferentes optimizadores
    network_sgd = create_network(optimizers.SGD(learning_rate=0.1))
    network_momentum = create_network(optimizers.SGDMomentum(learning_rate=0.1, momentum_coefficient=0.9))
    network_adagrad = create_network(optimizers.AdaGrad(learning_rate=0.1))
    network_adam = create_network(optimizers.Adam(learning_rate=0.01)) 

    # Entrenar redes
    losses_sgd = train_and_get_losses(network_sgd, X, y)
    losses_momentum = train_and_get_losses(network_momentum, X, y)
    losses_adagrad = train_and_get_losses(network_adagrad, X, y)
    losses_adam = train_and_get_losses(network_adam, X, y)

    # Graficar resultados
    plt.figure(figsize=(10, 6))
    plt.plot(losses_sgd, label='SGD')
    plt.plot(losses_momentum, label='SGD con Momentum')
    plt.plot(losses_adagrad, label='AdaGrad')
    plt.plot(losses_adam, label='Adam', linestyle='--')  # Estilo de línea diferente para distinguir
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Comparación de Optimizadores')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Escala log ayuda a visualizar diferencias de convergencia
    plt.show()

    # Imprimir pérdidas finales
    print(f"Pérdida final SGD: {losses_sgd[-1]:.6f}")
    print(f"Pérdida final SGD con Momentum: {losses_momentum[-1]:.6f}")
    print(f"Pérdida final AdaGrad: {losses_adagrad[-1]:.6f}")
    print(f"Pérdida final Adam: {losses_adam[-1]:.6f}")

if __name__ == '__main__':
    compare_optimizers()