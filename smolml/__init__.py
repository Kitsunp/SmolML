# smolml/__init__.py

# Importar componentes centrales (core components)
from .core import MLArray, Value

# Importar modelos (models)
from .models.nn import DenseLayer, NeuralNetwork
from .models.regression import LinearRegression, PolynomialRegression
from .models.tree import DecisionTree, RandomForest
from .models.unsupervised import KMeans

# Importar preprocesamiento (preprocessing)
from .preprocessing import StandardScaler, MinMaxScaler

# Importar utilidades (utilities)
from .utils.activation import (
    relu, leaky_relu, elu, sigmoid, softmax, tanh, linear
)
from .utils.initializers import (
    WeightInitializer, XavierUniform, XavierNormal, HeInitialization
)
from .utils.losses import (
    mse_loss, mae_loss, binary_cross_entropy, 
    categorical_cross_entropy, huber_loss
)
from .utils.optimizers import (
    Optimizer, SGD, SGDMomentum, AdaGrad, Adam
)
from .utils.memory import (format_size, calculate_value_size, calculate_mlarray_size, calculate_neural_network_size)

# Versión del paquete smolml
__version__ = '0.1.0'

__all__ = [
    # Núcleo (Core)
    'MLArray',
    'Value',
    
    # Modelos - Redes Neuronales (Models - Neural Networks)
    'DenseLayer',
    'NeuralNetwork',
    
    # Modelos - Regresión (Models - Regression)
    'LinearRegression',
    'PolynomialRegression',
    
    # Modelos - Basados en Árboles (Models - Tree-based)
    'DecisionTree',
    'RandomForest',

    # Modelos - No Supervisado (Models - Unsupervised)
    'KMeans',
    
    # Preprocesamiento (Preprocessing)
    'StandardScaler',
    'MinMaxScaler',
    
    # Utilidades - Funciones de Activación (Utils - Activation Functions)
    'relu',
    'leaky_relu',
    'elu',
    'sigmoid',
    'softmax',
    'tanh',
    'linear',
    
    # Utilidades - Inicializadores (Utils - Initializers)
    'WeightInitializer',
    'XavierUniform',
    'XavierNormal',
    'HeInitialization',
    
    # Utilidades - Funciones de Pérdida (Utils - Loss Functions)
    'mse_loss',
    'mae_loss',
    'binary_cross_entropy',
    'categorical_cross_entropy',
    'huber_loss',
    'log_cosh_loss',
    
    # Utilidades - Optimizadores (Utils - Optimizers)
    'Optimizer',
    'SGD',
    'SGDMomentum',
    'AdaGrad',
    'Adam',

    # Utilidades - Memoria (Utils - Memory)
    'format_size',
    'calculate_value_size',
    'calculate_mlarray_size',
    'calculate_neural_network_size'
]