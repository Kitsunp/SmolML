from smolml.core.ml_array import MLArray
from smolml.core.value import Value

"""
///////////////
/// ESCALADORES ///
///////////////
"""

class StandardScaler:
    """
    Estandariza características removiendo la media y escalando a varianza unitaria.
    Transforma características para tener media=0 y desviación estándar=1.
    """
    def __init__(self):
        """
        Inicializa escalador con atributos de media y desviación estándar vacíos.
        """
        self.mean = None
        self.std = None
        
    def fit(self, X):
        """
        Calcula media y desviación estándar de características de entrada para escalado posterior.
        Almacena valores internamente para el paso de transformación.
        """
        if not isinstance(X, MLArray):
            X = MLArray(X)
        
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        
        # Manejar desviación estándar cero
        if len(X.shape) <= 1:  # Valor único o array 1D
            if isinstance(self.std.data, (int, float)) and self.std.data == 0:
                self.std = MLArray(1.0)
        else:
            # Reemplazar desviaciones estándar cero con 1
            def replace_zeros(data):
                if isinstance(data, Value):
                    return Value(1.0) if data.data == 0 else data
                return [replace_zeros(d) for d in data]
            
            self.std.data = replace_zeros(self.std.data)

    def transform(self, X):
        """
        Estandariza características usando media y std previamente calculadas.
        Normalización z-score: z = (x - μ) / σ
        """
        if not isinstance(X, MLArray):
            X = MLArray(X)
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        """
        Método de conveniencia que ajusta el escalador y transforma datos en un solo paso.
        """
        self.fit(X)
        return self.transform(X)

class MinMaxScaler:
   """
   Transforma características escalando a un rango fijo, típicamente [0, 1].
   Preserva valores cero y maneja matrices dispersas.
   """
   def __init__(self):
       """
       Inicializa escalador con atributos de min y max vacíos.
       """
       self.max = None
       self.min = None

   def fit(self, X):
       """
       Calcula valores min y max de características de entrada para escalado posterior.
       Almacena valores internamente para el paso de transformación.
       """
       if not isinstance(X, MLArray):
           X = MLArray(X)
       self.max = X.max(axis=0)
       self.min = X.min(axis=0)

   def transform(self, X):
       """
       Escala características usando valores min y max previamente calculados.
       Fórmula MinMax: x_escalado = (x - x_min) / (x_max - x_min)
       """
       return (X - self.min) / (self.max - self.min)

   def fit_transform(self, X):
       """
       Método de conveniencia que ajusta el escalador y transforma datos en un solo paso.
       """
       self.fit(X)
       return self.transform(X)