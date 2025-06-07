from smolml.core.ml_array import MLArray
import random
import smolml.utils.memory as memory
from collections import Counter
from smolml.models.tree.decision_tree import DecisionTree

"""
/////////////////////
/// BOSQUE ALEATORIO ///
/////////////////////
"""

class RandomForest:
    """
    Implementación de Bosque Aleatorio que soporta tanto clasificación como regresión.
    Usa bagging (bootstrap aggregating) y selección aleatoria de características.
    """
    def __init__(self, n_trees=100, max_features=None, max_depth=None, 
                 min_samples_split=2, min_samples_leaf=1, bootstrap=True, 
                 task="classification"):
        """
        Inicializa bosque aleatorio con parámetros para árboles y bagging.
        
        Parámetros:
        n_trees: Número de árboles en el bosque
        max_features: Número de características a considerar para cada división (si None, usar sqrt para clasificación, 1/3 para regresión)
        bootstrap: Si usar muestras bootstrap para cada árbol
        task: "classification" o "regression"
        """
        self.n_trees = n_trees
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.task = task if task in ["classification", "regression"] else None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        
    def _bootstrap_sample(self, X, y):
        """
        Crea una muestra bootstrap con reemplazo.
        """
        n_samples = len(X)
        indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
        
        bootstrap_X = [X[i] for i in indices]
        bootstrap_y = [y[i] for i in indices]
        
        return bootstrap_X, bootstrap_y
    
    def _get_max_features(self, n_features):
        """
        Determina número de características a considerar en cada división.
        """
        if self.max_features is None:
            # Usar sqrt(n_features) para clasificación, n_features/3 para regresión
            if self.task == "classification":
                return max(1, int(n_features ** 0.5))
            else:
                return max(1, n_features // 3)
        return min(self.max_features, n_features)
    
    def fit(self, X, y):
        """
        Construye bosque aleatorio creando y entrenando árboles individuales.
        """
        if not isinstance(X, MLArray):
            X = MLArray(X)
        if not isinstance(y, MLArray):
            y = MLArray(y)
            
        X_data, y_data = X.data, y.data
        n_features = len(X_data[0])
        max_features = self._get_max_features(n_features)
        
        # Crear y entrenar cada árbol
        for _ in range(self.n_trees):
            # Crear muestra bootstrap si está habilitado
            if self.bootstrap:
                sample_X, sample_y = self._bootstrap_sample(X_data, y_data)
            else:
                sample_X, sample_y = X_data, y_data
            
            # Crear y entrenar árbol con selección aleatoria de características
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                task=self.task
            )
            
            # Modificar _find_best_split del árbol para usar subconjunto aleatorio de características
            original_find_best_split = tree._find_best_split
            def random_feature_find_best_split(self, X, y):
                n_features = len(X[0])
                feature_indices = random.sample(range(n_features), max_features)
                
                best_gain = -float('inf')
                best_feature = None
                best_threshold = None
                
                for feature_idx in feature_indices:
                    thresholds = sorted(set(row[feature_idx] for row in X))
                    
                    for threshold in thresholds:
                        left_idxs, right_idxs = tree._split_data(X, feature_idx, threshold)
                        
                        if len(left_idxs) < tree.min_samples_leaf or len(right_idxs) < tree.min_samples_leaf:
                            continue
                        
                        left_y = [y[i] for i in left_idxs]
                        right_y = [y[i] for i in right_idxs]
                        
                        gain = tree._calculate_gain(y, left_y, right_y)
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_feature = feature_idx
                            best_threshold = threshold
                
                return best_feature, best_threshold
            
            # Reemplazar _find_best_split del árbol con nuestra versión de características aleatorias
            tree._find_best_split = random_feature_find_best_split.__get__(tree)
            
            # Entrenar el árbol
            tree.fit(MLArray(sample_X), MLArray(sample_y))
            self.trees.append(tree)
    
    def predict(self, X):
        """
        Hace predicciones agregando predicciones de todos los árboles.
        Para clasificación: voto mayoritario
        Para regresión: predicción media
        """
        if not isinstance(X, MLArray):
            X = MLArray(X)
        
        # Obtener predicciones de todos los árboles
        tree_predictions = [tree.predict(X) for tree in self.trees]
        
        # Agregar predicciones basándose en la tarea
        if self.task == "classification":
            final_predictions = []
            for i in range(len(X)):
                # Obtener predicciones para esta muestra de todos los árboles
                sample_predictions = [tree_pred.data[i] for tree_pred in tree_predictions]
                # Tomar voto mayoritario
                vote = Counter(sample_predictions).most_common(1)[0][0]
                final_predictions.append(vote)
        elif self.task == "regression":
            final_predictions = []
            for i in range(len(X)):
                # Obtener predicciones para esta muestra de todos los árboles
                sample_predictions = [tree_pred.data[i] for tree_pred in tree_predictions]
                # Tomar media
                mean = sum(sample_predictions) / len(sample_predictions)
                final_predictions.append(mean)
        else:
            raise Exception(f"Tarea en Bosque Aleatorio no asignada a 'classification' o 'regression'")
        
        return MLArray(final_predictions)
    
    def __repr__(self):
        """
        Retorna representación en cadena de bosque aleatorio con información de estructura y memoria.
        """
        try:
            import os
            terminal_width = os.get_terminal_size().columns
        except Exception:
            terminal_width = 80
            
        header = f"Bosque Aleatorio ({self.task.title()})"
        separator = "=" * terminal_width
        
        # Obtener información de tamaño
        size_info = memory.calculate_random_forest_size(self)
        
        # Parámetros del modelo
        params = [
            f"Número de Árboles: {self.n_trees}",
            f"Máximo Características por División: {self.max_features if self.max_features else 'auto'}",
            f"Muestreo Bootstrap: {self.bootstrap}",
            f"Profundidad Máxima: {self.max_depth if self.max_depth else 'Ninguna'}",
            f"Mínimo Muestras División: {self.min_samples_split}",
            f"Mínimo Muestras Hoja: {self.min_samples_leaf}",
            f"Tarea: {self.task}"
        ]
        
        # Información de estructura del bosque
        if self.trees:
            structure_info = [
                "Estructura del Bosque:",
                f"  Árboles Construidos: {len(self.trees)}",
                f"  Profundidad Promedio del Árbol: {size_info['forest_stats']['avg_tree_depth']:.1f}",
                f"  Nodos Promedio por Árbol: {size_info['forest_stats']['avg_tree_nodes']:.1f}"
            ]
            
            # Agregar estadísticas de muestra del primer árbol si están disponibles
            if self.trees:
                first_tree_size = size_info['trees']['individual'][0]
                structure_info.extend([
                    "\nEstructura de Árbol de Muestra (Primer Árbol):",
                    f"  Nodos Internos: {first_tree_size['tree_structure']['internal_nodes']}",
                    f"  Nodos Hoja: {first_tree_size['tree_structure']['leaf_nodes']}",
                    f"  Profundidad Máxima: {first_tree_size['tree_structure']['max_depth']}"
                ])
        else:
            structure_info = ["Bosque aún no entrenado"]
        
        # Uso de memoria
        memory_info = ["Uso de Memoria:"]
        memory_info.append(f"  Bosque Base: {memory.format_size(size_info['base_size'])}")
        if self.trees:
            memory_info.extend([
                f"  Todos los Árboles: {memory.format_size(size_info['trees']['total'])}",
                f"  Promedio por Árbol: {memory.format_size(size_info['trees']['total'] / len(self.trees))}"
            ])
        memory_info.append(f"Memoria Total: {memory.format_size(size_info['total'])}")
        
        return (
            f"\n{header}\n{separator}\n\n"
            + "Parámetros:\n" + "\n".join(f"  {param}" for param in params)
            + "\n\n" + "\n".join(structure_info)
            + "\n\n" + "\n".join(memory_info)
            + f"\n{separator}\n"
        )