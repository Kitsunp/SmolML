from smolml.core.ml_array import MLArray
import math
import random
from functools import reduce
from operator import mul

"""
////////////////////
/// INICIALIZADORES ///
////////////////////
"""

class WeightInitializer:
   """
   Clase base para estrategias de inicialización de pesos de redes neuronales.
   Proporciona utilidades comunes para crear arrays de pesos.
   """
   @staticmethod
   def initialize(*dims):
       """
       Método abstracto para inicializar pesos.
       Debe ser implementado por clases inicializadoras concretas.
       """
       raise NotImplementedError("Las subclases deben implementar este método")

   @staticmethod
   def _create_array(generator, dims):
       """
       Crea un MLArray con dimensiones dadas usando una función generadora.
       Aplana dimensiones y redimensiona array a forma deseada.
       """
       total_elements = reduce(mul, dims)
       flat_array = [generator() for _ in range(total_elements)]
       return MLArray(flat_array).reshape(*dims)

class XavierUniform(WeightInitializer):
   """
   Inicialización uniforme Xavier/Glorot.
   Genera pesos desde distribución uniforme con varianza basada en dimensiones de capa.
   """
   @staticmethod
   def initialize(*dims):
       """
       Inicializa pesos usando distribución uniforme escalada por dimensiones de entrada/salida.
       Buena para capas con activación tanh o sigmoid.
       """
       dims = XavierUniform._process_dims(dims)
       fan_in = dims[0] if len(dims) > 0 else 1
       fan_out = dims[-1] if len(dims) > 1 else fan_in
       limit = math.sqrt(6. / (fan_in + fan_out))
       return XavierUniform._create_array(lambda: random.uniform(-limit, limit), dims)

   @staticmethod
   def _process_dims(dims):
       """
       Procesa dimensiones de entrada para manejar tanto tuplas/listas como argumentos separados.
       """
       if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
           return dims[0]
       return dims

class XavierNormal(WeightInitializer):
   """
   Inicialización normal Xavier/Glorot.
   Genera pesos desde distribución normal con varianza basada en dimensiones de capa.
   """
   @staticmethod
   def initialize(*dims):
       """
       Inicializa pesos usando distribución normal escalada por dimensiones de entrada/salida.
       Buena para capas con activación tanh o sigmoid.
       """
       dims = XavierNormal._process_dims(dims)
       fan_in = dims[0] if len(dims) > 0 else 1
       fan_out = dims[-1] if len(dims) > 1 else fan_in
       std = math.sqrt(2. / (fan_in + fan_out))
       return XavierNormal._create_array(lambda: random.gauss(0, std), dims)

   @staticmethod
   def _process_dims(dims):
       """
       Procesa dimensiones de entrada para manejar tanto tuplas/listas como argumentos separados.
       """
       if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
           return dims[0]
       return dims

class HeInitialization(WeightInitializer):
   """
   Inicialización He/Kaiming.
   Genera pesos desde distribución normal con varianza basada en dimensión de entrada.
   """
   @staticmethod
   def initialize(*dims):
       """
       Inicializa pesos usando distribución normal escalada por dimensión de entrada.
       Óptima para capas con activación ReLU.
       """
       dims = HeInitialization._process_dims(dims)
       fan_in = dims[0] if len(dims) > 0 else 1
       std = math.sqrt(2. / fan_in)
       return HeInitialization._create_array(lambda: random.gauss(0, std), dims)

   @staticmethod
   def _process_dims(dims):
       """
       Procesa dimensiones de entrada para manejar tanto tuplas/listas como argumentos separados.
       """
       if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
           return dims[0]
       return dims