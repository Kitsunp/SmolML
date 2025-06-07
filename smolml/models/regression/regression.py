import sys
import smolml.utils.initializers as initializers
import smolml.utils.optimizers as optimizers
from smolml.core.ml_array import MLArray
import smolml.core.ml_array as ml_array
import smolml.utils.losses as losses
import smolml.utils.memory as memory

"""
//////////////////
/// REGRESIÓN ///
//////////////////
"""

class Regression:
    """
    Clase base para algoritmos de regresión implementando funcionalidad común.
    Proporciona framework para ajustar modelos usando optimización de descenso de gradiente.
    Los tipos específicos de regresión deben heredar de esta clase.
    """
    def __init__(self, input_size: int, loss_function: callable = losses.mse_loss, optimizer: optimizers.Optimizer = optimizers.SGD, initializer: initializers.WeightInitializer = initializers.XavierUniform):
        """
        Inicializa modelo de regresión base con parámetros comunes.
        """
        self.input_size = input_size
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.initializer = initializer
        self.weights = self.initializer.initialize((self.input_size,))
        self.bias = ml_array.ones((1))

    def fit(self, X, y, iterations: int = 100, verbose: bool = True, print_every: int = 1):
        """
        Entrena el modelo de regresión usando descenso de gradiente.
        """
        X, y = MLArray.ensure_array(X, y)
        losses = []
        for i in range(iterations):
            # Hacer predicción 
            y_pred = self.predict(X)
            # Calcular pérdida
            loss = self.loss_function(y, y_pred)
            losses.append(loss.data.data)
            # Paso hacia atrás
            loss.backward()

            # Actualizar parámetros
            self.weights, self.bias = self.optimizer.update(self, self.__class__.__name__, param_names=("weights", "bias"))

            # Reiniciar gradientes
            X, y = self.restart(X, y)

            if verbose:
                if (i+1) % print_every == 0:
                    print(f"Iteración {i + 1}/{iterations}, Pérdida: {loss.data}")

        return losses

    def restart(self, X, y):
        """
        Reinicia gradientes para todos los parámetros y datos para la siguiente iteración.
        """
        X = X.restart()
        y = y.restart()
        self.weights = self.weights.restart()
        self.bias = self.bias.restart()
        return X, y
    
    def predict(self, X):
        """
        Método abstracto para hacer predicciones.
        Debe ser implementado por clases específicas de regresión.
        """
        raise NotImplementedError("Regression es solo clase base para algoritmos de Regresión, usa una de las clases que heredan de ella.")
    
    def __repr__(self):
        """
        Retorna una representación en cadena del modelo de regresión.
        Incluye tipo de modelo, parámetros, detalles del optimizador y uso de memoria.
        """
        try:
            import os
            terminal_width = os.get_terminal_size().columns
        except Exception:
            terminal_width = 80

        # Crear encabezado
        header = f"Modelo {self.__class__.__name__}"
        separator = "=" * terminal_width
        
        # Obtener información de tamaño
        size_info = memory.calculate_regression_size(self)
        
        # Estructura del modelo
        structure_info = [
            f"Tamaño de Entrada: {self.input_size}",
            f"Función de Pérdida: {self.loss_function.__name__}",
            f"Optimizador: {self.optimizer.__class__.__name__}(learning_rate={self.optimizer.learning_rate})"
        ]
        
        # Conteo de parámetros
        total_params = self.weights.size() + self.bias.size()
        
        # Desglose de memoria
        memory_info = ["Uso de Memoria:"]
        memory_info.append(
            f"  Parámetros: {memory.format_size(size_info['parameters']['total'])} "
            f"(pesos: {memory.format_size(size_info['parameters']['weights_size'])}, "
            f"sesgo: {memory.format_size(size_info['parameters']['bias_size'])})"
        )
        
        # Agregar información de estado del optimizador si existe
        if size_info['optimizer']['state']:
            opt_size = sum(size_info['optimizer']['state'].values())
            memory_info.append(f"  Estado del Optimizador: {memory.format_size(opt_size)}")
            for key, value in size_info['optimizer']['state'].items():
                memory_info.append(f"    {key}: {memory.format_size(value)}")
        
        memory_info.append(f"  Objetos Base: {memory.format_size(size_info['optimizer']['size'])}")
        memory_info.append(f"Memoria Total: {memory.format_size(size_info['total'])}")
        
        # Combinar todas las partes
        return (
            f"\n{header}\n{separator}\n\n"
            + "\n".join(structure_info)
            + f"\n\nParámetros Totales: {total_params:,}\n\n"
            + "\n".join(memory_info)
            + f"\n{separator}\n"
        )

class LinearRegression(Regression):
   """
   Implementa regresión lineal usando optimización de descenso de gradiente.
   El modelo aprende a ajustar: y = X @ weights + bias
   """
   def __init__(self, input_size: int, loss_function: callable = losses.mse_loss, optimizer: optimizers.Optimizer = optimizers.SGD, initializer: initializers.WeightInitializer = initializers.XavierUniform):
       """
       Inicializa modelo de regresión con parámetros de entrenamiento.
       """
       super().__init__(input_size, loss_function, optimizer, initializer)

   def predict(self, X):
       """
       Hace predicciones usando ecuación del modelo lineal.
       """
       if not isinstance(X, MLArray):
            raise TypeError(f"Los datos de entrada deben ser MLArray, no {type(X)}")
       return X @ self.weights + self.bias

class PolynomialRegression(Regression):
   """
   Extiende regresión lineal para ajustar relaciones polinomiales.
   Transforma características en términos polinomiales antes del ajuste.
   """
   def __init__(self, input_size: int, degree: int, loss_function: callable = losses.mse_loss, optimizer: optimizers.Optimizer = optimizers.SGD, initializer: initializers.WeightInitializer = initializers.XavierUniform):
       """
       Inicializa modelo polinomial con grado y parámetros de entrenamiento.
       """
       # Inicializar con grado para número de pesos necesarios
       super().__init__(degree, loss_function, optimizer, initializer)
       self.degree = degree
       
   def transform_features(self, X):
       """
       Crea características polinomiales hasta el grado especificado.
       Para entrada X y grado 2, produce [X, X^2].
       """
       features = [X]
       current = X
       for d in range(2, self.degree + 1):
           # Usar multiplicación elemento por elemento para potencia
           current = current * X
           features.append(current)
           
       # Crear un nuevo MLArray para características concatenadas
       new_data = []
       for i in range(len(X.data)):
           row = []
           for feature in features:
               row.append(feature.data[i][0])  # Extraer valor de cada característica
           new_data.append(row)
           
       return MLArray(new_data)

   def predict(self, X):
       """
       Hace predicciones después de transformar características a forma polinomial.
       """
       if not isinstance(X, MLArray):
            raise TypeError(f"Los datos de entrada deben ser MLArray, no {type(X)}")
       X_poly = self.transform_features(X)
       return X_poly @ self.weights + self.bias

   def fit(self, X, y, iterations: int = 100, verbose: bool = True, print_every: int = 1):
       """
       Transforma características a forma polinomial antes del entrenamiento.
       """
       X_poly = self.transform_features(X)
       return super().fit(X_poly, y, iterations, verbose, print_every)
   
   def __repr__(self):
        """
        Repr mejorado para regresión polinomial incluyendo información de grado.
        """
        try:
            import os
            terminal_width = os.get_terminal_size().columns
        except Exception:
            terminal_width = 80

        # Crear encabezado
        header = "Modelo de Regresión Polinomial"
        separator = "=" * terminal_width
        
        # Obtener información de tamaño
        size_info = memory.calculate_regression_size(self)
        
        # Estructura del modelo
        structure_info = [
            f"Tamaño de Entrada Original: {self.input_size}",
            f"Grado Polinomial: {self.degree}",
            f"Función de Pérdida: {self.loss_function.__name__}",
            f"Optimizador: {self.optimizer.__class__.__name__}(learning_rate={self.optimizer.learning_rate})"
        ]
        
        # Conteo de parámetros
        total_params = self.weights.size() + self.bias.size()
        
        # Desglose de memoria
        memory_info = ["Uso de Memoria:"]
        memory_info.append(
            f"  Parámetros: {memory.format_size(size_info['parameters']['total'])} "
            f"(pesos: {memory.format_size(size_info['parameters']['weights_size'])}, "
            f"sesgo: {memory.format_size(size_info['parameters']['bias_size'])})"
        )
        
        if size_info['optimizer']['state']:
            opt_size = sum(size_info['optimizer']['state'].values())
            memory_info.append(f"  Estado del Optimizador: {memory.format_size(opt_size)}")
            for key, value in size_info['optimizer']['state'].items():
                memory_info.append(f"    {key}: {memory.format_size(value)}")
        
        memory_info.append(f"  Objetos Base: {memory.format_size(size_info['optimizer']['size'])}")
        memory_info.append(f"Memoria Total: {memory.format_size(size_info['total'])}")
        
        # Combinar todas las partes
        return (
            f"\n{header}\n{separator}\n\n"
            + "\n".join(structure_info)
            + f"\n\nParámetros Totales: {total_params:,}\n\n"
            + "\n".join(memory_info)
            + f"\n{separator}\n"
        )