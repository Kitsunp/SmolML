import math

"""
/////////////
/// VALUE ///
/////////////

Librería de redes neuronales para diferenciación automática y cómputo de gradientes.
Basada en el increíble trabajo de karpathy (https://github.com/karpathy/micrograd).
"""
class Value:
    """
    Almacena un único valor escalar y su gradiente para diferenciación automática.
    Implementa retropropagación a través de un grafo computacional de objetos Value.
    """
    def __init__(self, data, _children=(), _op=""):
        """
        Inicializa un Value con datos e información opcional de cómputo de gradientes.
        Almacena nodos hijos y operación para construir el grafo computacional.
        """
        self.data = data
        self.grad = 0  # Derivada de la salida final con respecto a este valor
        self._backward = lambda: None  # Función para calcular gradientes
        self._prev = set(_children)  # Nodos hijos en el grafo computacional  
        self._op = _op  # Operación que creó este valor

    def __add__(self, other):
        """
        Suma dos Values y configura el cómputo de gradientes para retropropagación.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        """
        Multiplica dos Values y configura el cómputo de gradientes para retropropagación.
        Usa la regla del producto para las derivadas.
        """ 
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad  
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        """
        Eleva Value a una potencia y configura el cómputo de gradientes.
        Actualmente solo soporta potencias int/float.
        """
        other = other.data if isinstance(other, Value) else other
        assert isinstance(other, (int, float)), "solo se soportan potencias int/float/value por ahora"
        out = Value(self.data**other, (self,), f"**{other}")
        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __abs__(self):
        """
        Calcula el valor absoluto y configura el cómputo de gradientes.
        La derivada es 1 para valores positivos, -1 para valores negativos.
        """
        out = Value(abs(self.data), (self,), "abs")
        def _backward():
            self.grad += (1 if self.data >= 0 else -1) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        """
        Implementa división con manejo especial para división por cero.
        Retorna 0 cuando se divide por 0 o números muy pequeños.
        """
        other = other if isinstance(other, Value) else Value(other)
        if abs(other.data) < 1e-10:  # División por cero o número muy pequeño
            out = Value(0.0, (self, other), "/")
            def _backward():
                self.grad += 0.0
                other.grad += 0.0
            out._backward = _backward
            return out
        return self * other**-1

    def __rtruediv__(self, other):
        """
        Implementa división inversa (other / self) con manejo de ceros.
        """
        if abs(self.data) < 1e-10:  # División por cero o número muy pequeño
            out = Value(0.0, (self,), "r/")
            def _backward():
                self.grad += 0.0
            out._backward = _backward
            return out
        return other * self**-1

    def exp(self):
        """
        Calcula el exponencial (e^x) y configura el cómputo de gradientes.
        La derivada de e^x es e^x.
        """
        x = self.data
        out = Value(math.exp(x), (self,), "exp")
        def _backward():
            self.grad += math.exp(x) * out.grad
        out._backward = _backward
        return out

    def log(self):
        """
        Calcula el logaritmo natural (ln) y configura el cómputo de gradientes.
        La derivada de ln(x) es 1/x.
        """
        assert self.data > 0, "log solo está definido para números positivos"
        x = self.data
        out = Value(math.log(x), (self,), "log")
        def _backward():
            self.grad += (1.0 / x) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        """
        Aplica la función de activación ReLU y configura el cómputo de gradientes.
        La derivada es 1 para valores positivos, 0 para valores negativos.
        """
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        """
        Aplica la función de activación tanh y configura el cómputo de gradientes.
        La derivada es 1 - tanh²(x).
        """
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        """
        Realiza retropropagación para calcular gradientes en el grafo computacional.
        Usa ordenamiento topológico para procesar nodos en orden correcto.
        """
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    # Sobrecargas de operadores para conveniencia
    def __neg__(self): return self * -1  # -self
    def __radd__(self, other): return self + other  # other + self
    def __sub__(self, other): return self + (-other)  # self - other
    def __rsub__(self, other): return other + (-self)  # other - self
    def __rmul__(self, other): return self * other  # other * self

    # Métodos de comparación y representación
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    def __eq__(self, other):
        if isinstance(other, Value): return self.data == other.data
        return self.data == other
    def __ne__(self, other): return not self.__eq__(other)
    def __lt__(self, other):
        if isinstance(other, Value): return self.data < other.data
        return self.data < other
    def __le__(self, other):
        if isinstance(other, Value): return self.data <= other.data
        return self.data <= other
    def __gt__(self, other):
        if isinstance(other, Value): return self.data > other.data
        return self.data > other
    def __ge__(self, other):
        if isinstance(other, Value): return self.data >= other.data
        return self.data >= other
    def __hash__(self): return id(self)