from smolml.core.value import Value
import random
import math

"""
///////////////
/// MLARRAY ///
///////////////
"""

class MLArray:
    """
    Clase que representa un Array N-Dimensional para aplicaciones de ML.
    """
    
    """
    ///////////////
    /// General ///
    ///////////////
    """

    def __init__(self, data) -> None:
        """
        Crea un nuevo MLArray dados algunos datos (escalar, 1D -usando lista de python-, o >=2D -usando listas anidadas-)
        """
        self.data = self._process_data(data)

    def _process_data(self, data):
        """
        Procesa recursivamente los datos de entrada y todos los valores se inicializan como Value para diferenciación automática
        """
        if isinstance(data, (int, float)):
            return Value(data)
        elif isinstance(data, list):
            return [self._process_data(item) for item in data]
        elif isinstance(data, (Value, MLArray)):
            return data
        else:
            raise TypeError(f"Tipo de datos no soportado: {type(data)}")

    def _infer_shape(self, data):
        """
        Obtiene la forma del MLArray basándose en sus datos actuales.
        """
        if isinstance(data, Value):
            return ()
        elif isinstance(data, list):
            return (len(data),) + self._infer_shape(data[0])
        else:
            return ()

    """
    ///////////////////////////
    /// Operaciones Estándar ///
    ///////////////////////////
    """
        
    def __neg__(self):
        return self * -1
        
    def __add__(self, other):
        return self._element_wise_operation(other, lambda x, y: x + y)
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self._element_wise_operation(other, lambda x, y: x - y)
    
    def __rsub__(self, other):
        return MLArray(other) - self
    
    def __mul__(self, other):
        return self._element_wise_operation(other, lambda x, y: x * y)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self._element_wise_operation(other, lambda x, y: x / y)
    
    def __rtruediv__(self, other):
        return MLArray(other) / self
    
    def __pow__(self, other):
        return self._element_wise_operation(other, lambda x, y: x ** y)

    """
    ///////////////////////////
    /// Operaciones Avanzadas ///
    ///////////////////////////
    """

    def __T__(self):
        return self.transpose()

    def transpose(self, axes=None):
        """
        Transpone un MLArray multi-dimensional basándose en ciertos ejes.
        """
        if len(self.shape) <= 1: # Array escalar o 1D
            return self
        
        if axes is None: # Si no hay ejes, invertir los ejes actuales
            axes = list(range(len(self.shape)))[::-1]
        
        new_shape = tuple(self.shape[i] for i in axes)
        
        def _all_possible_indices(shape):
            """
            Genera todas las combinaciones de índices posibles dada cierta forma.
            """
            if len(shape) == 0:
                yield []
            else:
                for i in range(shape[0]):
                    for rest in _all_possible_indices(shape[1:]):
                        yield [i] + rest

        new_data = self._create_nested_list(new_shape) # Crear lista vacía con nueva forma transpuesta

        for indices in _all_possible_indices(self.shape): # Agregar elementos transpuestos 
            new_indices = [indices[i] for i in axes]
            value = self._get_item(self.data, indices)
            self._set_item(new_data, new_indices, value)

        return MLArray(new_data)

    def __matmul__(self, other):
        return self.matmul(other)

    def matmul(self, other):
        """
        Realiza una multiplicación de matrices entre dos MLArrays.
        Soporta arrays multi-dimensionales.
        """
        if not isinstance(other, MLArray):
            other = MLArray(other)

        # Manejar multiplicación escalar
        if len(self.shape) == 0 or len(other.shape) == 0:
            return self * other
        
        # 1D
        if len(self.shape) == 1:
            a = MLArray([self.data])
        else:
            a = self

        if len(other.shape) == 1:
            b = MLArray([other.data]).transpose()
        else:
            b = other

        # Redimensionar entradas si es necesario
        a = a.reshape(-1, a.shape[-1]) if len(a.shape) > 2 else a
        b = b.reshape(b.shape[0], -1) if len(b.shape) > 2 else b

        if a.shape[-1] != b.shape[0]:
            raise ValueError(f"Formas incompatibles para multiplicación de matrices: {self.shape} y {other.shape}")

        # Realizar multiplicación de matrices
        index_b = 0 if len(b.shape) == 1 else 1
        result = self._create_nested_list((a.shape[0], b.shape[index_b]))
        for i in range(a.shape[0]):
            for j in range(b.shape[index_b]):
                result[i][j] = sum(a.data[i][k] * b.data[k][j] for k in range(a.shape[1]))

        # Redimensionar resultado si es necesario
        if len(self.shape) > 2 or len(other.shape) > 2:
            new_shape = self.shape[:-1] + other.shape[1:]
            return MLArray(result).reshape(new_shape)
        else:
            return MLArray(result)

    def sum(self, axis=None):
        """
        Retorna la suma de todos los valores dentro de un MLArray.
        Si se especifica axis, realiza la suma a lo largo de ese eje.
        """
        if axis is None:
            def recursive_sum(data):
                if isinstance(data, Value):
                    return data
                elif isinstance(data, list):
                    return sum(recursive_sum(x) for x in data)
                else:
                    return 0
            
            if len(self.shape) == 0:  # escalar
                return self
            else:
                return MLArray(recursive_sum(self.data))
        
        # Manejar eje negativo
        if axis < 0:
            axis += len(self.shape)
            
        if axis < 0 or axis >= len(self.shape):
            raise ValueError(f"Eje inválido {axis} para MLArray con forma {self.shape}")
        
        def sum_along_axis(data, current_depth):
            if current_depth == axis:
                if isinstance(data[0], list):
                    # Transponer los datos en este nivel y sumar
                    transposed = list(zip(*data))
                    return [sum(slice) for slice in transposed]
                else:
                    return sum(data)
            
            if isinstance(data[0], list):
                return [sum_along_axis(subdata, current_depth + 1) for subdata in data]
            return data
        
        result = sum_along_axis(self.data, 0)
        
        # Manejar el caso donde result se convierte en escalar
        if not isinstance(result, list):
            return MLArray(result)
        
        return MLArray(result)

    def min(self, axis=None):
        """
        Retorna el elemento más pequeño en el MLArray
        """
        return self.reduce_operation(min, axis)

    def max(self, axis=None):
        """
        Retorna el elemento más grande en el MLArray
        """
        return self.reduce_operation(max, axis)

    def mean(self, axis=None):
        """
        Calcula la media a lo largo del eje especificado o globalmente si axis=None
        """
        if axis is None:
            flat_data = self.flatten(self.data)
            return MLArray(sum(flat_data)) / len(flat_data)
        
        return self.reduce_operation(sum, axis=axis) / self.shape[axis]

    def std(self, axis=None):
        """
        Calcula la desviación estándar a lo largo del eje especificado o globalmente si axis=None.
        Usa una implementación más compatible con Value.
        """
        mean = self.mean(axis=axis)
        
        if axis is None:
            # Para std global, calcular diferencias aplanadas
            flat_diffs = [(Value(x) - Value(mean.data)) * (Value(x) - Value(mean.data)) 
                        for x in self.flatten(self.data)]
            squared_diff = MLArray([diff.data for diff in flat_diffs])
            return (squared_diff.sum() / squared_diff.size()).sqrt()
        
        # Para std específico de eje:
        # 1. Crear media compatible con broadcasting
        broadcast_shape = list(self.shape)
        broadcast_shape[axis] = 1
        mean_broadcast = mean.reshape(*broadcast_shape)
        
        # 2. Calcular diferencias cuadradas manualmente para evitar restricción de pow en Value
        diff = self - mean_broadcast
        squared_diff = MLArray([[x * x for x in row] for row in diff.data])
        
        # 3. Tomar media de diferencias cuadradas y hacer sqrt
        return (squared_diff.sum(axis=axis) / self.shape[axis]).sqrt()

    def sqrt(self):
        """
        Calcula la raíz cuadrada de cada elemento en el array.
        """
        if len(self.shape) == 0:  # caso escalar
            return MLArray(math.sqrt(self.data.data))
        
        def sqrt_value(x):
            if isinstance(x, (int, float)):
                return math.sqrt(x)
            return Value(math.sqrt(x.data))
        
        flat_data = [sqrt_value(x) for x in self.flatten(self.data)]
        if len(self.shape) == 0:
            return MLArray(flat_data[0])
        return MLArray(flat_data).reshape(*self.shape)
    
    def exp(self):
        return self._element_wise_function(lambda val: val.exp())

    def log(self):
        return self._element_wise_function(lambda val: val.log())

    def abs(self):
        return self._element_wise_function(lambda val: abs(val))

    def __len__(self):
        return len(self.data)

    """
    /////////////////////////
    /// Funciones de Utilidad ///
    /////////////////////////
    """
    
    def _get_item(self, data, indices):
        for idx in indices:
            data = data[idx]
        return data

    def _set_item(self, data, indices, value):
        for idx in indices[:-1]:
            data = data[idx]
        data[indices[-1]] = value
    
    def to_list(self):
        """
        Llama a la función recursiva _to_list() con self.data como parámetro para convertir self.data en una lista estándar de python
        """
        return self._to_list(self.data)
    
    def _to_list(self, data):
        """
        Función recursiva que quita la clase Value de los datos, retornando una lista estándar de python con valores estándar.
        """
        if isinstance(data, (Value)):
            return data.data
        elif isinstance(data, list):
            return [self._to_list(item) for item in data]
        
    def restart(self):
        """
        Reemplaza todos los objetos Value en el MLArray con nuevos objetos Value que contienen los mismos datos, reiniciando efectivamente el grafo computacional.
        """
        self._restart_data(self.data)
        return self

    def _restart_data(self, data):
        """
        Recorre recursivamente todos los objetos Value y establece sus gradientes a 0.
        """
        if isinstance(data, Value):
            data.grad = 0
            data.prev = ()
        elif isinstance(data, list):
            for item in data:
                self._restart_data(item)
    
    def backward(self):
        """
        Realiza el paso hacia atrás de todos los datos dentro del MLArray.
        """
        # Aplanar MLArray para obtener todos los objetos Value
        flat_data = self.flatten(self.data)

        # Encontrar el Value de salida (asumido como escalar o tomamos la suma)
        if len(flat_data) == 1:
            output_value = flat_data[0]
        else:
            output_value = sum(flat_data)
        
        # Llamar backward en el Value de salida
        output_value.backward()
    
    def flatten(self, data):
        """
        Aplana todos los datos dentro del MLArray, retornando una lista simple con todos los Values sin importar la dimensionalidad.
        """
        if isinstance(data, Value):
            return [data]
        elif isinstance(data, list):
            return [item for sublist in data for item in self.flatten(sublist)]
        else:
            return []

    def unflatten(self, flat_list, shape):
        """
        Des-aplana una lista en una estructura anidada basándose en la forma dada.
        """
        if len(shape) == 1:
            return flat_list[:shape[0]]
        else:
            stride = len(flat_list) // shape[0]
            return [self.unflatten(flat_list[i*stride:(i+1)*stride], shape[1:]) for i in range(shape[0])]

    def _create_nested_list(self, shape):
        """
        Crea una estructura de datos de lista vacía basándose en cierta forma.
        """
        if len(shape) == 1:
            return [None] * shape[0]
        return [self._create_nested_list(shape[1:]) for _ in range(shape[0])]

    def update_values(self, new_data):
        """
        Actualiza los Values existentes en el MLArray con nuevos datos mientras preserva la estructura del array.
        """
        def update_recursive(current_data, new_data):
            if isinstance(current_data, Value):
                current_data.data = new_data.data if isinstance(new_data, Value) else new_data
            elif isinstance(current_data, list):
                for i, (curr, new) in enumerate(zip(current_data, new_data)):
                    update_recursive(curr, new)
                    
        update_recursive(self.data, new_data.data if isinstance(new_data, MLArray) else new_data)
        return self
    
    @staticmethod
    def ensure_array(*args):
        """
        Convierte cualquier número de argumentos en MLArrays si no lo son ya.
        """
        def _convert_single_arg(arg):
            # Si ya es MLArray, retornar tal como está
            if isinstance(arg, MLArray):
                return arg
                
            # Si es array numpy, convertir a lista primero 
            if str(type(arg).__module__) == 'numpy':
                arg = arg.tolist()
                
            # Manejar diferentes tipos de entrada
            if isinstance(arg, (int, float)):
                return MLArray([arg])
            elif isinstance(arg, list):
                # Verificar si la lista contiene solo números
                def is_numeric_list(lst):
                    for item in lst:
                        if isinstance(item, list):
                            if not is_numeric_list(item):
                                return False
                        elif not isinstance(item, (int, float)):
                            return False
                    return True
                
                if is_numeric_list(arg):
                    return MLArray(arg)
                else:
                    raise TypeError(f"La lista contiene valores no numéricos: {type(arg)}")
            else:
                raise TypeError(f"No se puede convertir el tipo {type(arg)} a MLArray")
        
        # Convertir cada argumento
        converted = []
        for i, arg in enumerate(args):
            try:
                converted.append(_convert_single_arg(arg))
            except Exception as e:
                raise TypeError(f"Error convirtiendo argumento {i}: {str(e)}")
        
        # Retornar tupla de arrays convertidos
        return tuple(converted) if len(converted) > 1 else converted[0]
    
    @staticmethod
    def _broadcast_shapes(shape1, shape2):
        """
        Retorna la forma resultante de broadcasting dadas dos formas de entrada. Acepta multi-dimensionalidad.
        Por ejemplo: (3,4,5) | (4, 1) -> (3,4,5)
        """
        # Asegurar que shape1 sea la forma más larga
        if len(shape2) > len(shape1):
            shape1, shape2 = shape2, shape1
        
        # Rellenar la forma más corta con 1s
        shape2 = (1,) * (len(shape1) - len(shape2)) + shape2
        
        result = []
        for s1, s2 in zip(shape1, shape2):
            if s1 == s2:
                result.append(s1)
            elif s1 == 1 or s2 == 1:
                result.append(max(s1, s2))
            else:
                raise ValueError(f"No se pueden hacer broadcasting de las formas {shape1} y {shape2}")
        return tuple(result)

    def _broadcast_and_apply(self, data1, data2, shape1, shape2, target_shape, op):
        """
        Función recursiva que aplica una operación op entre dos MLArrays, haciendo broadcasting según sea necesario en el proceso.
        """
        if not shape1 and not shape2:  # Ambos escalares
            return op(data1, data2)
        elif not shape1:  # data1 es escalar -> Llamada recursiva para aplicar la operación al escalar data1 y cada elemento de data2
            return [self._broadcast_and_apply(data1, d2, (), shape2[1:], target_shape[1:], op) for d2 in data2]
        elif not shape2:  # data2 es escalar -> Llamada recursiva para aplicar la operación al escalar data2 y cada elemento de data1
            return [self._broadcast_and_apply(d1, data2, shape1[1:], (), target_shape[1:], op) for d1 in data1]
        else: # Ambos arrays
            if len(shape1) > len(shape2):
                # Rellenar data2 con dimensiones extra
                data2 = [data2] * target_shape[0]
                shape2 = (target_shape[0],) + shape2
            elif len(shape2) > len(shape1):
                # Rellenar data1 con dimensiones extra
                data1 = [data1] * target_shape[0]
                shape1 = (target_shape[0],) + shape1

            if shape1[0] == target_shape[0] and shape2[0] == target_shape[0]: # Ambas primeras dimensiones coinciden con la forma objetivo -> Llamada recursiva para aplicar la operación a cada elemento de data1 y data2
                return [self._broadcast_and_apply(d1, d2, shape1[1:], shape2[1:], target_shape[1:], op) for d1, d2 in zip(data1, data2)]
            elif shape1[0] == 1: # Primera dimensión de shape1 es 1 (broadcasting necesario) -> Llamada recursiva para aplicar la operación a data1 y cada elemento de data2
                return [self._broadcast_and_apply(data1[0], d2, shape1[1:], shape2[1:], target_shape[1:], op) for d2 in data2]
            elif shape2[0] == 1: # Primera dimensión de shape2 es 1 (broadcasting necesario) -> Llamada recursiva para aplicar la operación a data2 y cada elemento de data1
                return [self._broadcast_and_apply(d1, data2[0], shape1[1:], shape2[1:], target_shape[1:], op) for d1 in data1]

    def _element_wise_operation(self, other, op):
        """
        Realiza una operación elemento por elemento entre dos MLArrays.
        """
        if isinstance(other, (int, float, Value)):
            other = MLArray(other)
        
        if not isinstance(other, MLArray):
            raise TypeError(f"Tipo de operando no soportado: {type(other)}")
        
        target_shape = self._broadcast_shapes(self.shape, other.shape)
        result = self._broadcast_and_apply(self.data, other.data, self.shape, other.shape, target_shape, op)
        return MLArray(result)

    def _element_wise_function(self, fn):
        """
        Función auxiliar para aplicar funciones elemento por elemento a MLArray n-dimensional
        """
        if len(self.shape) == 0:  # escalar
            return MLArray(fn(self.data))
        
        def apply_recursive(data):
            if isinstance(data, list):
                return [apply_recursive(d) for d in data]
            return fn(data)
        
        return MLArray(apply_recursive(self.data))

    def reduce_operation(self, op, axis=None):
        """
        Realiza operación de reducción a lo largo del eje especificado.
        """
        # Caso 1: Reducción global (reducir todos los elementos)
        if axis is None:
            return op(self.flatten(self.data), key=lambda x: x)
            
        # Caso 2: Reducir a lo largo de eje específico
        if not isinstance(axis, int):
            raise TypeError("Axis debe ser None o un entero")
            
        # Manejar eje negativo
        if axis < 0:
            axis += len(self.shape)
            
        if axis < 0 or axis >= len(self.shape):
            raise ValueError(f"Eje inválido {axis} para MLArray con forma {self.shape}")
            
        def reduce_recursive(data, current_depth, target_axis, shape):
            # Caso base: alcanzado eje objetivo
            if current_depth == target_axis:
                if isinstance(data[0], list):
                    # Transponer los datos en este nivel
                    transposed = list(zip(*data))
                    # Aplicar reducción a cada segmento transpuesto
                    return [op(slice) for slice in transposed]
                else:
                    return op(data)
                    
            # Caso recursivo: aún no en eje objetivo
            return [reduce_recursive(subarray, current_depth + 1, target_axis, shape[1:]) 
                    for subarray in data]
        
        # Obtener nueva forma después de reducción
        new_shape = list(self.shape)
        new_shape.pop(axis)
        
        # Realizar reducción
        result = reduce_recursive(self.data, 0, axis, self.shape)
        
        # Manejar el caso donde result es escalar
        if not new_shape:
            return result
        
        return MLArray(result)

    def reshape(self, *new_shape):
        """
        Redimensiona el array a la nueva forma.
        """
        # Calcular el tamaño total
        total_size = self.size()
        
        # Manejar -1 en new_shape
        if -1 in new_shape:
            # Calcular el producto de todas las dimensiones excepto -1
            known_dim_product = 1
            unknown_dim_index = new_shape.index(-1)
            
            for i, dim in enumerate(new_shape):
                if i != unknown_dim_index:
                    known_dim_product *= dim
                    
            # Calcular la dimensión faltante
            if total_size % known_dim_product != 0:
                raise ValueError(f"No se puede redimensionar array de tamaño {total_size} a forma {new_shape}")
            
            missing_dim = total_size // known_dim_product
            new_shape = list(new_shape)
            new_shape[unknown_dim_index] = missing_dim
            new_shape = tuple(new_shape)
        
        # Calcular el producto de las dimensiones de la nueva forma
        new_size = 1
        for dim in new_shape:
            new_size *= dim
        
        # Verificar si la nueva forma es válida
        if new_size != total_size:
            raise ValueError(f"No se puede redimensionar array de tamaño {total_size} a forma {new_shape}")
        
        # Aplanar el array y crear nueva estructura
        flat_data = self.flatten(self.data)
        new_data = self.unflatten(flat_data, new_shape)
        return MLArray(new_data)

    def size(self):
        """
        Retorna el número de elementos que componen el MLArray.
        """
        return len(self.flatten(self.data))
    
    def grad(self):
        """
        Retorna un nuevo MLArray que contiene los gradientes de todos los objetos Value en este MLArray.
        """
        def extract_grad(data):
            if isinstance(data, Value):
                return data.grad
            elif isinstance(data, list):
                return [extract_grad(item) for item in data]
            else:
                raise TypeError(f"Tipo inesperado en MLArray: {type(data)}")

        return MLArray(extract_grad(self.data))
        
    def __repr__(self):
        def format_array(arr, indent=0):
            if not isinstance(arr, list):
                return str(arr)
            
            if not arr:
                return '[]'
            
            if isinstance(arr[0], list):
                # 2D o superior
                rows = [format_array(row, indent + 1) for row in arr]
                return '[\n' + ',\n'.join(' ' * (indent + 1) + row for row in rows) + '\n' + ' ' * indent + ']'
            else:
                # 1D
                return '[' + ', '.join(str(item) for item in arr) + ']'

        formatted_data = format_array(self.data)
        return f"MLArray(shape={self.shape},\ndata={formatted_data})"

    """
    ///////////////////////
    /// Subscriptabilidad ///
    ///////////////////////
    """

    def __getitem__(self, index):
        """
        Habilita indexación de arrays con operador [].
        Soporta indexación entera y tuplas para múltiples dimensiones.
        
        Ejemplos:
            arr[0]     # obtener primer elemento
            arr[1, 2]  # obtener elemento en fila 1, columna 2
        """
        if not isinstance(index, tuple):
            index = (index,)
            
        def get_item_recursive(data, index):
            if len(index) == 1:
                return data[index[0]]
            
            curr_index = index[0]
            return get_item_recursive(data[curr_index], index[1:])
            
        return MLArray(get_item_recursive(self.data, index))
            

    def __setitem__(self, index, value):
        """
        Habilita asignación de arrays con operador [].
        Soporta indexación entera y tuplas para múltiples dimensiones.
        Ejemplos:
            arr[0] = 1      # establecer primer elemento
            arr[1, 2] = 3   # establecer elemento en fila 1, columna 2
        """
        if not isinstance(index, tuple):
            index = (index,)
        
        # Convertir value a MLArray si no lo es ya
        if not isinstance(value, MLArray):
            value = MLArray(value)
            
        def set_item_recursive(data, index, value):
            if len(index) == 1:
                data[index[0]] = value.data
            else:
                curr_index = index[0]
                set_item_recursive(data[curr_index], index[1:], value)
                
        set_item_recursive(self.data, index, value)

    """
    ////////////////////
    /// Classmethods ///
    ////////////////////
    """
    
    @classmethod
    def xavier_uniform(cls, in_features, out_features):
        limit = math.sqrt(6. / (in_features + out_features))
        data = [[random.uniform(-limit, limit) for _ in range(out_features)] for _ in range(in_features)]
        return cls(data)

    @classmethod
    def xavier_normal(cls, in_features, out_features):
        std = math.sqrt(2. / (in_features + out_features))
        data = [[random.gauss(0, std) for _ in range(out_features)] for _ in range(in_features)]
        return cls(data)

    """
    //////////////////
    /// Propiedades ///
    //////////////////
    """

    @property
    def shape(self):
        return self._infer_shape(self.data)

"""
/////////////////////////
/// MLArrays Pre-Hechos ///
/////////////////////////
"""
    
def zeros(*shape):
    """
    Crea un MLArray lleno de 0's dada una forma.
    """
    def _zeros(shape):
        if len(shape) == 0:
            return Value(0.0)
        return [_zeros(shape[1:]) for _ in range(shape[0])]

    return MLArray(_zeros(shape))

def ones(*shape):
    """
    Crea un MLArray lleno de 1's dada una forma.
    """
    def _ones(shape):
        if len(shape) == 0:
            return Value(1.0)
        return [_ones(shape[1:]) for _ in range(shape[0])]

    return MLArray(_ones(shape))

def randn(*shape):
    """
    Crea un MLArray lleno de números aleatorios entre 0 y 1 con distribución gaussiana dada una forma.
    """
    def _randn(shape):
        if len(shape) == 0:
            return Value(random.gauss(0, 1))
        return [_randn(shape[1:]) for _ in range(shape[0])]

    return MLArray(_randn(shape))