from smolml.core.ml_array import MLArray

"""
//////////////////////
/// FUNCIONES DE PÉRDIDA ///
//////////////////////

Funciones de pérdida para entrenar modelos de machine learning.
Cada función calcula el error entre predicciones y valores verdaderos.
"""

def mse_loss(y_pred, y_true):
    """
    Pérdida de Error Cuadrático Medio.
    Función de pérdida estándar para regresión.
    """
    diff = y_pred - y_true
    squared_diff = diff * diff
    return squared_diff.mean()

def mae_loss(y_pred, y_true):
    """
    Pérdida de Error Absoluto Medio.
    Menos sensible a valores atípicos que MSE.
    """
    diff = (y_pred - y_true).abs()
    return diff.mean()

def binary_cross_entropy(y_pred, y_true):
    """
    Pérdida de Entropía Cruzada Binaria.
    Para problemas de clasificación binaria.
    Espera que y_pred esté en el rango (0,1).
    """
    epsilon = 1e-15  # Prevenir log(0)
    y_pred = MLArray([[max(min(p, 1 - epsilon), epsilon) for p in row] for row in y_pred.data])
    return -(y_true * y_pred.log() + (1 - y_true) * (1 - y_pred).log()).mean()

def categorical_cross_entropy(y_pred, y_true):
    """
    Pérdida de Entropía Cruzada Categórica.
    Para problemas de clasificación multi-clase.
    Espera que y_pred sea distribución de probabilidad.
    """
    epsilon = 1e-15
    y_pred = MLArray([[max(p, epsilon) for p in row] for row in y_pred.data])
    return -(y_true * y_pred.log()).sum(axis=1).mean()

def huber_loss(y_pred, y_true, delta=1.0):
    """
    Pérdida Huber.
    Combina MSE y MAE - cuadrática para errores pequeños, lineal para grandes.
    Más robusta a valores atípicos que MSE mientras mantiene suavidad.
    """
    diff = y_pred - y_true
    abs_diff = diff.abs()
    quadratic = 0.5 * diff * diff
    linear = delta * abs_diff - 0.5 * delta * delta
    return MLArray([[quad if abs_d <= delta else lin 
                    for quad, lin, abs_d in zip(row_quad, row_lin, row_abs)]
                    for row_quad, row_lin, row_abs in zip(quadratic.data, linear.data, abs_diff.data)]).mean()