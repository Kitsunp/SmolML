from smolml.core.ml_array import MLArray
from smolml.models.nn import DenseLayer
import smolml.utils.memory as memory
import smolml.utils.losses as losses
import smolml.utils.activation as activation
import smolml.utils.optimizers as optimizers

"""
//////////////////////
/// RED NEURONAL ///
//////////////////////
"""

class NeuralNetwork:
    """
    Implementación de una red neuronal feedforward con capas y función de pérdida personalizables.
    Soporta entrenamiento a través de retropropagación y descenso de gradiente.
    """
    def __init__(self, layers: list, loss_function: callable, optimizer: optimizers.Optimizer = optimizers.SGD()) -> None:
        """
        Inicializa la red con una lista de capas y una función de pérdida para entrenamiento.
        """
        self.layers = layers
        self.loss_function = loss_function
        self.optimizer = optimizer if optimizer is not None else optimizers.SGD()

    def forward(self, input_data):
        """
        Realiza paso hacia adelante aplicando secuencialmente la transformación de cada capa.
        """
        if not isinstance(input_data, MLArray):
            raise TypeError(f"Los datos de entrada deben ser MLArray, no {type(input_data)}")
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def train(self, X, y, epochs, verbose=True, print_every=1):
        """
        Entrena la red usando descenso de gradiente por el número especificado de épocas.
        Imprime pérdida cada 100 épocas para monitorear progreso del entrenamiento.
        """
        X, y = MLArray.ensure_array(X, y)
        losses = []
        for epoch in range(epochs):
            # Paso hacia adelante a través de la red
            y_pred = self.forward(X)
            
            # Calcular pérdida entre predicciones y objetivos
            loss = self.loss_function(y_pred, y)
            losses.append(loss.data.data)
            
            # Paso hacia atrás para calcular gradientes
            loss.backward()
            
            # Actualizar parámetros en cada capa
            for idx, layer in enumerate(self.layers):
                layer.update(self.optimizer, idx)
            
            # Reiniciar gradientes para la siguiente iteración
            X.restart()
            y.restart()
            for layer in self.layers:
                layer.weights.restart()
                layer.biases.restart()
                
            if verbose:
                # Imprimir progreso del entrenamiento
                if (epoch+1) % print_every == 0:
                    print(f"Época {epoch + 1}/{epochs}, Pérdida: {loss.data}")

        return losses
        
    def __repr__(self):
        """
        Retorna una representación en cadena de la arquitectura de red neuronal.
        Muestra información de capas, función de pérdida, detalles del optimizador y uso detallado de memoria.
        """
        # Obtener ancho de terminal para formateo
        try:
            import os
            terminal_width = os.get_terminal_size().columns
        except Exception:
            terminal_width = 80

        # Crear encabezado
        header = "Arquitectura de Red Neuronal"
        separator = "=" * terminal_width
        
        # Obtener información de tamaño
        size_info = memory.calculate_neural_network_size(self)
        
        # Formatear información de capas
        layers_info = []
        for i, (layer, layer_size) in enumerate(zip(self.layers, size_info['layers'])):
            if isinstance(layer, DenseLayer):
                input_size = layer.weights.shape[0]
                output_size = layer.weights.shape[1]
                activation_name = layer.activation_function.__name__
                layer_info = [
                    f"Capa {i+1}: Densa("
                    f"entrada={input_size}, "
                    f"salida={output_size}, "
                    f"activación={activation_name})"
                ]
                
                # Información de parámetros
                params = input_size * output_size + output_size  # pesos + sesgos
                layer_info.append(
                    f"    Parámetros: {params:,} "
                    f"({input_size}×{output_size} pesos + {output_size} sesgos)"
                )
                
                # Información de memoria
                layer_info.append(
                    f"    Memoria: {memory.format_size(layer_size['total'])} "
                    f"(pesos: {memory.format_size(layer_size['weights_size'])}, "
                    f"sesgos: {memory.format_size(layer_size['biases_size'])})"
                )
                
                layers_info.append("\n".join(layer_info))

        # Calcular parámetros totales
        total_params = sum(
            layer.weights.size() + layer.biases.size()
            for layer in self.layers
        )

        # Formatear información del optimizador
        optimizer_info = [
            f"Optimizador: {self.optimizer.__class__.__name__}("
            f"learning_rate={self.optimizer.learning_rate})"
        ]
        
        # Agregar información de estado del optimizador si existe
        if size_info['optimizer']['state']:
            state_sizes = [
                f"    {key}: {memory.format_size(value)}"
                for key, value in size_info['optimizer']['state'].items()
            ]
            optimizer_info.extend(state_sizes)
        
        # Formatear información de función de pérdida
        loss_info = f"Función de Pérdida: {self.loss_function.__name__}"

        # Desglose detallado de memoria
        memory_info = ["Uso de Memoria:"]
        
        # Memoria de capas
        for i, layer_size in enumerate(size_info['layers']):
            memory_info.append(
                f"  Capa {i+1}: {memory.format_size(layer_size['total'])} "
                f"(pesos: {memory.format_size(layer_size['weights_size'])}, "
                f"sesgos: {memory.format_size(layer_size['biases_size'])})"
            )
        
        # Memoria del optimizador
        if size_info['optimizer']['state']:
            opt_size = sum(size_info['optimizer']['state'].values())
            memory_info.append(f"  Estado del Optimizador: {memory.format_size(opt_size)}")
        
        memory_info.append(f"  Objetos Base: {memory.format_size(size_info['optimizer']['size'])}")
        memory_info.append(f"Memoria Total: {memory.format_size(size_info['total'])}")

        # Combinar todas las partes
        return (
            f"\n{header}\n{separator}\n\n"
            f"Arquitectura:\n"
            + "\n".join(f"  {layer}" for layer in layers_info)
            + f"\n\n"
            + "\n".join(optimizer_info)
            + f"\n{loss_info}\n\n"
            f"Parámetros Totales: {total_params:,}\n\n"
            + "\n".join(memory_info)
            + f"\n{separator}\n"
        )

def example_neural_network():
    # Ejemplo de uso
    input_size = 2
    hidden_size = 32
    output_size = 1

    # Crear la red neuronal
    nn = NeuralNetwork([
        DenseLayer(input_size, hidden_size, activation.relu),
        DenseLayer(hidden_size, output_size, activation.tanh)
    ], losses.mse_loss, optimizers.AdaGrad(learning_rate=0.1))

    # Generar algunos datos dummy
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [[0], [1], [1], [0]]

    print(nn)
    # Entrenar la red
    nn.train(X, y, epochs=100)

    y_pred = nn.forward(MLArray(X))
    print(y_pred)

if __name__ == '__main__':
    example_neural_network()