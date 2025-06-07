from smolml.core.ml_array import zeros
import smolml.utils.activation as activation
import smolml.utils.initializers as initializers

"""
//////////////
/// CAPAS ///
//////////////
"""

class DenseLayer:
    """
    Crea una capa Densa (lineal) estándar para una Red Neuronal.
    Esta capa realiza la operación: salida = activación(entrada @ pesos + sesgos)
    """
    def __init__(self, input_size: int, output_size: int, activation_function: callable = activation.linear, 
                 weight_initializer: initializers.WeightInitializer = initializers.XavierUniform) -> None:
        """
        Inicializa parámetros de capa: pesos usando el inicializador especificado y sesgos con ceros.
        La activación por defecto es lineal y la inicialización de pesos por defecto es Xavier Uniforme.
        """
        self.weights = weight_initializer.initialize(input_size, output_size)
        self.biases = zeros(1, output_size)  # Inicializar sesgos con ceros
        self.activation_function = activation_function

    def forward(self, input_data):
        """
        Realiza el paso hacia adelante: aplica transformación lineal seguida de función de activación.
        """
        z = input_data @ self.weights + self.biases  # Transformación lineal
        return self.activation_function(z)  # Aplicar función de activación

    def update(self, optimizer, layer_idx):
        """Actualizar parámetros usando el optimizador proporcionado"""
        self.weights, self.biases = optimizer.update(self, layer_idx, param_names=("weights", "biases"))