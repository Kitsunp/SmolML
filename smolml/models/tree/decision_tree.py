from smolml.core.ml_array import MLArray
import smolml.utils.memory as memory
from collections import Counter
import math

"""
/////////////////////
/// ÁRBOL DE DECISIÓN ///
/////////////////////
"""

class DecisionNode:
    """
    Nodo en árbol de decisión que maneja la lógica de división.
    Puede ser nodo interno (con regla de división) u hoja (con predicción).
    """
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  # Índice de característica para dividir
        self.threshold = threshold      # Valor para dividir la característica
        self.left = left               # Subárbol izquierdo (característica <= umbral)
        self.right = right             # Subárbol derecho (característica > umbral)
        self.value = value             # Valor de predicción (para nodos hoja)

    def __repr__(self):
        if self.value is not None:
            return f"Hoja(valor={self.value})"
        return f"Nodo(característica={self.feature_idx}, umbral={self.threshold:.4f})"

class DecisionTree:
    """
    Implementación de Árbol de Decisión que soporta tanto clasificación como regresión.
    Usa división binaria basada en umbrales de características.
    """
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, task="classification"):
        """
        Inicializa árbol de decisión con criterios de parada.
        
        max_depth: Profundidad máxima del árbol para prevenir sobreajuste
        min_samples_split: Mínimo de muestras requeridas para dividir nodo
        min_samples_leaf: Mínimo de muestras requeridas en nodos hoja
        task: "classification" o "regression"
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.task = task
        self.root = None

    def fit(self, X, y):
        """
        Construye árbol de decisión dividiendo recursivamente los datos.
        """
        if not isinstance(X, MLArray):
            X = MLArray(X)
        if not isinstance(y, MLArray):
            y = MLArray(y)
            
        self.n_classes = len(set(y.flatten(y.data))) if self.task == "classification" else None
        self.root = self._grow_tree(X.data, y.data)

    def _grow_tree(self, X, y, depth=0):
        """
        Hace crecer el árbol recursivamente encontrando las mejores divisiones.
        """
        n_samples = len(X)
        
        # Verificar criterios de parada
        if (self.max_depth is not None and depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            self._is_pure(y)):
            return DecisionNode(value=self._leaf_value(y))

        # Encontrar mejor división
        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:  # No se encontró división válida
            return DecisionNode(value=self._leaf_value(y))

        # Dividir datos
        left_idxs, right_idxs = self._split_data(X, best_feature, best_threshold)
        
        # Verificar min_samples_leaf
        if len(left_idxs) < self.min_samples_leaf or len(right_idxs) < self.min_samples_leaf:
            return DecisionNode(value=self._leaf_value(y))

        # Crear nodos hijos
        left_X = [X[i] for i in left_idxs]
        right_X = [X[i] for i in right_idxs]
        left_y = [y[i] for i in left_idxs]
        right_y = [y[i] for i in right_idxs]

        left = self._grow_tree(left_X, left_y, depth + 1)
        right = self._grow_tree(right_X, right_y, depth + 1)

        return DecisionNode(feature_idx=best_feature, threshold=best_threshold, left=left, right=right)

    def _find_best_split(self, X, y):
        """
        Encuentra la mejor característica y umbral para dividir los datos.
        Usa ganancia de información para clasificación, MSE para regresión.
        """
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None

        n_features = len(X[0])
        
        for feature_idx in range(n_features):
            thresholds = sorted(set(row[feature_idx] for row in X))
            
            for threshold in thresholds:
                left_idxs, right_idxs = self._split_data(X, feature_idx, threshold)
                
                if len(left_idxs) < self.min_samples_leaf or len(right_idxs) < self.min_samples_leaf:
                    continue

                left_y = [y[i] for i in left_idxs]
                right_y = [y[i] for i in right_idxs]
                
                gain = self._calculate_gain(y, left_y, right_y)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _split_data(self, X, feature_idx, threshold):
        """
        Divide datos basándose en característica y umbral.
        """
        left_idxs = []
        right_idxs = []
        
        for i, row in enumerate(X):
            if row[feature_idx] <= threshold:
                left_idxs.append(i)
            else:
                right_idxs.append(i)
                
        return left_idxs, right_idxs

    def _calculate_gain(self, parent, left, right):
        """
        Calcula ganancia de la división:
        - Ganancia de información para clasificación
        - Reducción en MSE para regresión
        """
        if self.task == "classification":
            return self._information_gain(parent, left, right)
        return self._mse_reduction(parent, left, right)

    def _information_gain(self, parent, left, right):
        """
        Calcula ganancia de información usando entropía.
        """
        def entropy(y):
            counts = Counter(y)
            probs = [count/len(y) for count in counts.values()]
            return -sum(p * math.log2(p) for p in probs)

        n = len(parent)
        entropy_parent = entropy(parent)
        entropy_children = (len(left)/n * entropy(left) + 
                          len(right)/n * entropy(right))
        return entropy_parent - entropy_children

    def _mse_reduction(self, parent, left, right):
        """
        Calcula reducción en MSE.
        """
        def mse(y):
            mean = sum(y)/len(y)
            return sum((val - mean)**2 for val in y)/len(y)

        n = len(parent)
        mse_parent = mse(parent)
        mse_children = (len(left)/n * mse(left) + 
                       len(right)/n * mse(right))
        return mse_parent - mse_children

    def _split_data(self, X, feature_idx, threshold):
        """
        Divide datos basándose en característica y umbral.
        """
        left_idxs = [i for i, row in enumerate(X) if row[feature_idx] <= threshold]
        right_idxs = [i for i, row in enumerate(X) if row[feature_idx] > threshold]
        return left_idxs, right_idxs

    def _is_pure(self, y):
        """
        Verifica si el nodo es puro (toda la misma clase/valor).
        """
        return len(set(y)) == 1

    def _leaf_value(self, y):
        """
        Determina valor de predicción para nodo hoja:
        - Clase más común para clasificación
        - Valor medio para regresión
        """
        if self.task == "classification":
            return max(set(y), key=y.count)
        return sum(y)/len(y)

    def predict(self, X):
        """
        Hace predicciones usando árbol entrenado.
        """
        if not isinstance(X, MLArray):
            X = MLArray(X)
            
        return MLArray([self._traverse_tree(x, self.root) for x in X.data])

    def _traverse_tree(self, x, node):
        """
        Recorre árbol para hacer predicción para una sola muestra.
        """
        if node.value is not None:  # Nodo hoja
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def __repr__(self):
        """
        Retorna representación en cadena de árbol de decisión con información de estructura y memoria.
        """
        try:
            import os
            terminal_width = os.get_terminal_size().columns
        except Exception:
            terminal_width = 80
            
        header = f"Árbol de Decisión ({self.task.title()})"
        separator = "=" * terminal_width
        
        # Obtener información de tamaño
        size_info = memory.calculate_decision_tree_size(self)
        
        # Parámetros del modelo
        params = [
            f"Profundidad Máxima: {self.max_depth if self.max_depth is not None else 'Ninguna'}",
            f"Mínimo Muestras División: {self.min_samples_split}",
            f"Mínimo Muestras Hoja: {self.min_samples_leaf}",
            f"Tarea: {self.task}"
        ]
        
        # Información de estructura del árbol
        if self.root:
            structure_info = [
                "Estructura del Árbol:",
                f"  Nodos Internos: {size_info['tree_structure']['internal_nodes']}",
                f"  Nodos Hoja: {size_info['tree_structure']['leaf_nodes']}",
                f"  Profundidad Máxima: {size_info['tree_structure']['max_depth']}",
                f"  Nodos Totales: {size_info['tree_structure']['internal_nodes'] + size_info['tree_structure']['leaf_nodes']}"
            ]
        else:
            structure_info = ["Árbol aún no entrenado"]
        
        # Uso de memoria
        memory_info = ["Uso de Memoria:"]
        memory_info.append(f"  Árbol Base: {memory.format_size(size_info['base_size'])}")
        if self.root:
            memory_info.append(f"  Estructura del Árbol: {memory.format_size(size_info['tree_structure']['total'])}")
        memory_info.append(f"Memoria Total: {memory.format_size(size_info['total'])}")
        
        return (
            f"\n{header}\n{separator}\n\n"
            + "Parámetros:\n" + "\n".join(f"  {param}" for param in params)
            + "\n\n" + "\n".join(structure_info)
            + "\n\n" + "\n".join(memory_info)
            + f"\n{separator}\n"
        )