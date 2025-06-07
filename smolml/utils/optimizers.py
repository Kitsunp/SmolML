from smolml.core.ml_array import zeros

class Optimizer:
    """Clase base de optimizador que define la interfaz para todos los optimizadores"""
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
    
    def update(self, object, object_idx, param_names):
        """Regla de actualización a ser implementada por optimizadores específicos"""
        raise NotImplementedError

class SGD(Optimizer):
    """Optimizador estándar de Descenso de Gradiente Estocástico"""
    def update(self, object, object_idx, param_names):
        """
        Regla de actualización para SGD estándar: θ = θ - α∇θ
        donde α es la tasa de aprendizaje.
        
        Esta es la forma más básica de descenso de gradiente, que actualiza directamente
        los parámetros en la dirección opuesta al gradiente, escalada por la tasa de aprendizaje.
        """
        new_params = tuple(
            getattr(object, name) - self.learning_rate * getattr(object, name).grad()
            for name in param_names
        )
        return new_params

class SGDMomentum(Optimizer):
    """
    Optimizador de Descenso de Gradiente Estocástico con momentum.
    Este optimizador acelera SGD acumulando un vector de velocidad en la dirección de gradientes persistentes,
    ayudando a evitar mínimos locales y acelerar la convergencia.
    """
    def __init__(self, learning_rate: float = 0.01, momentum_coefficient: float = 0.9):
        super().__init__(learning_rate)
        self.momentum_coefficient = momentum_coefficient
        self.velocities = {}
        
    def update(self, object, object_idx, param_names):
        """
        Regla de actualización para SGD con momentum: v = βv + α∇θ, θ = θ - v
        donde β es el coeficiente de momentum y α es la tasa de aprendizaje.
        """
        # Inicializar velocidades para esta capa si no existen
        if object_idx not in self.velocities:
            self.velocities[object_idx] = {
                name: zeros(*getattr(object, name).shape) for name in param_names
            }
        
        new_params = []
        for name in param_names:
            # Actualizar velocidad
            v = self.velocities[object_idx][name]
            v = self.momentum_coefficient * v + self.learning_rate * getattr(object, name).grad()
            self.velocities[object_idx][name] = v
            
            # Calcular nuevo parámetro
            new_params.append(getattr(object, name) - v)
        
        return tuple(new_params)

class AdaGrad(Optimizer):
    """
    Optimizador de Gradiente Adaptativo.
    Adapta la tasa de aprendizaje a los parámetros, realizando actualizaciones más pequeñas 
    para parámetros actualizados frecuentemente y actualizaciones más grandes para los infrecuentes.
    """
    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate)
        self.epsilon = 1e-8
        self.squared_gradients = {}
        
    def update(self, object, object_idx, param_names):
        """
        Regla de actualización para AdaGrad: θ = θ - (α / √(G + ε)) * ∇θ
        donde G es la suma de gradientes cuadrados hasta el paso temporal actual
        """
        # Inicializar gradientes cuadrados para esta capa si no existen
        if object_idx not in self.squared_gradients:
            self.squared_gradients[object_idx] = {
                name: zeros(*getattr(object, name).shape) for name in param_names
            }
        
        new_params = []
        for name in param_names:
            # Actualizar suma de gradientes cuadrados
            self.squared_gradients[object_idx][name] += getattr(object, name).grad()**2
            
            # Calcular nuevo parámetro
            new_params.append(
                getattr(object, name) - (self.learning_rate / 
                (self.squared_gradients[object_idx][name] + self.epsilon).sqrt()) * 
                getattr(object, name).grad()
            )
        
        return tuple(new_params)

class Adam(Optimizer):
    """
    Optimizador Adam (Estimación de Momento Adaptativo).
    Combina los beneficios de:
    1. Momentum: Manteniendo registro del promedio de gradientes con decaimiento exponencial
    2. RMSprop: Manteniendo registro de gradientes cuadrados con decaimiento exponencial
    También incluye términos de corrección de sesgo para manejar la inicialización.
    """
    def __init__(self, learning_rate: float = 0.01, exp_decay_gradients: float = 0.9, exp_decay_squared: float = 0.999):
        super().__init__(learning_rate)
        self.exp_decay_gradients = exp_decay_gradients
        self.exp_decay_squared = exp_decay_squared
        self.gradients_momentum = {}
        self.squared_gradients_momentum = {}
        self.epsilon = 1e-8
        self.timestep = 1
        
    def update(self, object, object_idx, param_names):
        """
        Regla de actualización para Adam: θ = θ - α * m̂ / (√v̂ + ε)
        donde:
        - m̂ es la estimación del primer momento corregida por sesgo
        - v̂ es la estimación del segundo momento corregida por sesgo
        """
        # Inicializar momentums si no existen
        if object_idx not in self.gradients_momentum:
            self.gradients_momentum[object_idx] = {
                name: zeros(*getattr(object, name).shape) for name in param_names
            }
            self.squared_gradients_momentum[object_idx] = {
                name: zeros(*getattr(object, name).shape) for name in param_names
            }
        
        new_params = []
        for name in param_names:
            # Actualizar estimación sesgada del primer momento
            self.gradients_momentum[object_idx][name] = (
                self.exp_decay_gradients * self.gradients_momentum[object_idx][name] + 
                (1 - self.exp_decay_gradients) * getattr(object, name).grad()
            )
            
            # Actualizar estimación sesgada del segundo momento
            self.squared_gradients_momentum[object_idx][name] = (
                self.exp_decay_squared * self.squared_gradients_momentum[object_idx][name] + 
                (1 - self.exp_decay_squared) * getattr(object, name).grad()**2
            )
            
            # Calcular momentos corregidos por sesgo
            m = self.gradients_momentum[object_idx][name] / (1 - self.exp_decay_gradients ** self.timestep)
            v = self.squared_gradients_momentum[object_idx][name] / (1 - self.exp_decay_squared ** self.timestep)
            
            # Calcular nuevo parámetro
            new_params.append(
                getattr(object, name) - self.learning_rate * m / (v.sqrt() + self.epsilon)
            )
        
        self.timestep += 1
        return tuple(new_params)