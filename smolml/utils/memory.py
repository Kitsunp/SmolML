"""
///////////////
/// MEMORIA ///
///////////////

Utilidades para calcular uso de memoria de diferentes tipos de modelos y estructuras de datos.
"""

import sys
from typing import Dict, Any, Union, TYPE_CHECKING

# Sugerencias de tipo únicamente, sin imports reales
if TYPE_CHECKING:
    from smolml.core.value import Value
    from smolml.core.ml_array import MLArray
    from smolml.models.nn import NeuralNetwork
    from smolml.models.regression import Regression
    from smolml.models.tree.decision_tree import DecisionNode, DecisionTree
    from smolml.models.tree.random_forest import RandomForest

def format_size(size_bytes):
    """Función auxiliar para formatear tamaño en bytes a formato legible por humanos"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"

def calculate_value_size(value: 'Value') -> int:
    """Calcula la huella de memoria de un objeto Value."""
    total = sys.getsizeof(value)
    total += sys.getsizeof(value.data)  # float
    total += sys.getsizeof(value.grad)  # float
    total += sys.getsizeof(value._prev)  # set
    total += sys.getsizeof(value._op)    # str
    total += sys.getsizeof(value._backward)  # function
    return total

def calculate_mlarray_size(arr: 'MLArray') -> int:
    """Calcula la huella de memoria de un objeto MLArray."""
    # Importar Value aquí para evitar dependencia circular
    from smolml.core.value import Value
    
    def get_nested_size(data):
        if isinstance(data, Value):
            return calculate_value_size(data)
        elif isinstance(data, list):
            size = sys.getsizeof(data)
            size += sum(get_nested_size(item) for item in data)
            return size
        else:
            return sys.getsizeof(data)
    
    return sys.getsizeof(arr) + get_nested_size(arr.data)

def calculate_decision_node_size(node: 'DecisionNode') -> Dict[str, Any]:
    """Calcula la huella de memoria de un DecisionNode y su subárbol."""
    size_info = {
        'total': 0,
        'node_size': 0,
        'children': {'left': 0, 'right': 0},
        'node_type': 'hoja' if node.value is not None else 'interno'
    }
    
    # Tamaño base del nodo
    size_info['node_size'] += sys.getsizeof(node)
    
    # Agregar tamaño de atributos
    if node.feature_idx is not None:
        size_info['node_size'] += sys.getsizeof(node.feature_idx)
    if node.threshold is not None:
        size_info['node_size'] += sys.getsizeof(node.threshold)
    if node.value is not None:
        size_info['node_size'] += sys.getsizeof(node.value)
    
    # Calcular recursivamente tamaños de hijos
    if node.left:
        left_size = calculate_decision_node_size(node.left)
        size_info['children']['left'] = left_size['total']
        size_info['total'] += left_size['total']
    
    if node.right:
        right_size = calculate_decision_node_size(node.right)
        size_info['children']['right'] = right_size['total']
        size_info['total'] += right_size['total']
    
    size_info['total'] += size_info['node_size']
    return size_info

def calculate_regression_size(model: 'Regression') -> Dict[str, Any]:
    """Calcula la huella de memoria de un modelo de regresión."""
    size_info = {
        'total': 0,
        'parameters': {
            'weights_size': calculate_mlarray_size(model.weights),
            'bias_size': calculate_mlarray_size(model.bias)
        },
        'optimizer': {
            'size': sys.getsizeof(model.optimizer),
            'state': {}
        }
    }
    
    # Calcular tamaños de parámetros
    params_total = (size_info['parameters']['weights_size'] + 
                   size_info['parameters']['bias_size'])
    size_info['parameters']['total'] = params_total
    size_info['total'] += params_total
    
    # Agregar tamaño de estado del optimizador si existe
    if hasattr(model.optimizer, '_state'):
        for key, value in model.optimizer._state.items():
            if isinstance(value, dict):
                state_size = sum(calculate_mlarray_size(v) for v in value.values())
            else:
                state_size = calculate_mlarray_size(value)
            size_info['optimizer']['state'][key] = state_size
            size_info['total'] += state_size
    
    # Agregar tamaños de objetos base
    size_info['total'] += (
        sys.getsizeof(model) +
        sys.getsizeof(model.loss_function) +
        size_info['optimizer']['size']
    )
    
    return size_info

def calculate_neural_network_size(model: 'NeuralNetwork') -> Dict[str, Any]:
    """Calcula la huella de memoria de una red neuronal."""
    size_info = {
        'total': 0,
        'layers': [],
        'optimizer': {
            'size': sys.getsizeof(model.optimizer),
            'state': {}
        }
    }
    
    # Calcular tamaño de cada capa
    for layer in model.layers:
        layer_info = {
            'weights_size': calculate_mlarray_size(layer.weights),
            'biases_size': calculate_mlarray_size(layer.biases),
        }
        layer_info['total'] = layer_info['weights_size'] + layer_info['biases_size']
        size_info['layers'].append(layer_info)
        size_info['total'] += layer_info['total']
    
    # Agregar estado del optimizador
    if hasattr(model.optimizer, '_state'):
        for key, value in model.optimizer._state.items():
            if isinstance(value, dict):
                state_size = sum(calculate_mlarray_size(v) for v in value.values())
            else:
                state_size = calculate_mlarray_size(value)
            size_info['optimizer']['state'][key] = state_size
            size_info['total'] += state_size
    
    # Agregar tamaños de objetos base
    size_info['total'] += (
        sys.getsizeof(model) +
        sys.getsizeof(model.layers) +
        sys.getsizeof(model.loss_function) +
        size_info['optimizer']['size']
    )
    
    return size_info

def calculate_decision_tree_size(model: 'DecisionTree') -> Dict[str, Any]:
    """Calcula la huella de memoria de un árbol de decisión."""
    size_info = {
        'total': 0,
        'base_size': 0,
        'tree_structure': {
            'total': 0,
            'internal_nodes': 0,
            'leaf_nodes': 0,
            'max_depth': 0
        }
    }
    
    # Calcular tamaño del objeto árbol base
    size_info['base_size'] = (
        sys.getsizeof(model) +
        sys.getsizeof(model.max_depth) +
        sys.getsizeof(model.min_samples_split) +
        sys.getsizeof(model.min_samples_leaf) +
        sys.getsizeof(model.task)
    )
    
    # Calcular tamaño de estructura del árbol si está entrenado
    if model.root:
        root_size = calculate_decision_node_size(model.root)
        size_info['tree_structure'].update(_analyze_tree_structure(model.root))
        size_info['tree_structure']['total'] = root_size['total']
    
    size_info['total'] = size_info['base_size'] + size_info['tree_structure']['total']
    return size_info

def _analyze_tree_structure(node: 'DecisionNode', depth: int = 0) -> Dict[str, int]:
    """Función auxiliar para analizar estructura del árbol."""
    if node is None:
        return {'internal_nodes': 0, 'leaf_nodes': 0, 'max_depth': depth}
    
    left = _analyze_tree_structure(node.left, depth + 1)
    right = _analyze_tree_structure(node.right, depth + 1)
    
    return {
        'internal_nodes': (
            left['internal_nodes'] + 
            right['internal_nodes'] + 
            (1 if node.value is None else 0)
        ),
        'leaf_nodes': (
            left['leaf_nodes'] + 
            right['leaf_nodes'] + 
            (1 if node.value is not None else 0)
        ),
        'max_depth': max(left['max_depth'], right['max_depth'])
    }

def calculate_random_forest_size(model: 'RandomForest') -> Dict[str, Any]:
    """Calcula la huella de memoria de un bosque aleatorio."""
    import sys
    from smolml.utils.memory import calculate_decision_tree_size
    
    size_info = {
        'total': 0,
        'base_size': 0,
        'trees': {
            'total': 0,
            'individual': []
        },
        'forest_stats': {
            'n_trees': model.n_trees,
            'max_features': model.max_features,
            'avg_tree_depth': 0,
            'avg_tree_nodes': 0
        }
    }
    
    # Calcular tamaño del objeto bosque base
    size_info['base_size'] = (
        sys.getsizeof(model) +
        sys.getsizeof(model.n_trees) +
        sys.getsizeof(model.max_features) +
        sys.getsizeof(model.bootstrap) +
        sys.getsizeof(model.task) +
        sys.getsizeof(model.trees)
    )
    
    # Calcular tamaño de cada árbol
    if model.trees:
        total_depth = 0
        total_nodes = 0
        
        for tree in model.trees:
            tree_size = calculate_decision_tree_size(tree)
            size_info['trees']['individual'].append(tree_size)
            size_info['trees']['total'] += tree_size['total']
            
            # Recopilar estadísticas
            total_depth += tree_size['tree_structure']['max_depth']
            total_nodes += (tree_size['tree_structure']['internal_nodes'] + 
                          tree_size['tree_structure']['leaf_nodes'])
        
        # Calcular promedios
        size_info['forest_stats']['avg_tree_depth'] = total_depth / len(model.trees)
        size_info['forest_stats']['avg_tree_nodes'] = total_nodes / len(model.trees)
    
    size_info['total'] = size_info['base_size'] + size_info['trees']['total']
    return size_info