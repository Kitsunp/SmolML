from smolml.core.ml_array import MLArray
import math

"""
////////////////////////////
/// FUNCIONES DE ACTIVACIÓN ///
////////////////////////////

Funciones de activación para capas de redes neuronales.
Cada función aplica transformaciones no lineales elemento por elemento.
"""

def _element_wise_activation(x, activation_fn):
    """
    Función auxiliar para aplicar función de activación elemento por elemento a MLArray n-dimensional
    """
    if len(x.shape) == 0:  # escalar
        return MLArray(activation_fn(x.data))
    
    def apply_recursive(data):
        if isinstance(data, list):
            return [apply_recursive(d) for d in data]
        return activation_fn(data)
    
    return MLArray(apply_recursive(x.data))

def relu(x):
    """
    Activación Unidad Lineal Rectificada (ReLU).
    Calcula max(0,x) para cada elemento.
    Elección estándar para redes profundas.
    """
    return _element_wise_activation(x, lambda val: val.relu())

def leaky_relu(x, alpha=0.01):
    """
    Activación ReLU con fuga.
    Retorna x si x > 0, sino alpha * x.
    Previene el problema de ReLU moribundo con pendiente negativa pequeña.
    """
    def leaky_relu_single(val):
        if val > 0:
            return val
        return val * alpha
    
    return _element_wise_activation(x, leaky_relu_single)

def elu(x, alpha=1.0):
    """
    Unidad Lineal Exponencial.
    Retorna x si x > 0, sino alpha * (e^x - 1).
    Alternativa más suave a ReLU con valores negativos.
    """
    def elu_single(val):
        if val > 0:
            return val
        return alpha * (val.exp() - 1)
    
    return _element_wise_activation(x, elu_single)

def sigmoid(x):
    """
    Activación sigmoide.
    Mapea entradas al rango (0,1) usando 1/(1 + e^-x).
    Usada para salida de clasificación binaria.
    """
    def sigmoid_single(val):
        return 1 / (1 + (-val).exp())
    
    return _element_wise_activation(x, sigmoid_single)

def softmax(x, axis=-1):
    """
    Activación softmax.
    Normaliza entradas en distribución de probabilidad.
    Usada para salida de clasificación multi-clase.
    """
    # Manejar caso escalar
    if len(x.shape) == 0:
        return MLArray(1.0)  # Softmax de un escalar siempre es 1
        
    # Manejar eje negativo
    if axis < 0:
        axis += len(x.shape)
        
    # Manejar caso 1D
    if len(x.shape) == 1:
        max_val = x.max()
        exp_x = (x - max_val).exp()
        sum_exp = exp_x.sum()
        return exp_x / sum_exp
    
    # Manejar caso multi-dimensional
    def apply_softmax_along_axis(data, curr_depth=0):
        """
        Aplica recursivamente softmax a lo largo del eje especificado
        """
        if curr_depth == axis:
            if isinstance(data[0], list):
                # Convertir a transpuesta sin usar zip
                transposed = []
                for i in range(len(data[0])):
                    slice_data = [row[i] for row in data]
                    # Encontrar max para estabilidad numérica
                    max_val = max(slice_data)
                    # Calcular exp(x - max)
                    exp_vals = [(val - max_val).exp() for val in slice_data]
                    # Calcular suma
                    sum_exp = sum(exp_vals)
                    # Calcular softmax
                    softmax_vals = [exp_val / sum_exp for exp_val in exp_vals]
                    transposed.append(softmax_vals)
                    
                # Convertir de vuelta desde transpuesta sin usar zip
                result = []
                for i in range(len(data)):
                    row = [transposed[j][i] for j in range(len(transposed))]
                    result.append(row)
                return result
            else:
                # Cálculo directo para segmento 1D
                max_val = max(data)
                exp_vals = [(val - max_val).exp() for val in data]
                sum_exp = sum(exp_vals)
                return [exp_val / sum_exp for exp_val in exp_vals]
        
        # Caso recursivo: aún no en eje objetivo
        return [apply_softmax_along_axis(subarray, curr_depth + 1) 
                for subarray in data]
    
    result = apply_softmax_along_axis(x.data)
    return MLArray(result)

def tanh(x):
    """
    Activación tangente hiperbólica.
    Mapea entradas al rango [-1,1].
    """
    return _element_wise_activation(x, lambda val: val.tanh())

def linear(x):
    """
    Activación lineal/identidad.
    Pasa la entrada sin cambios.
    """
    return x