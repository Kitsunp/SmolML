# SmolML: Machine Learning desde Cero, ¡Hecho Claro! ✨

**Una librería de machine learning en Python puro construida completamente desde cero con fines educativos. ¡Creada para entender cómo funciona realmente el ML!**

<div align="center">
  <img src="https://github.com/user-attachments/assets/c00b89e9-58a3-44d8-b9c3-4b47052eb150" width="600" alt="Logo SmolML">
</div>

---

## ¿Qué es SmolML? 🤔

¿Alguna vez te has preguntado qué sucede *dentro* de esas poderosas librerías de machine learning como Scikit-learn, PyTorch o TensorFlow? ¿Cómo *realmente* aprende una red neuronal? ¿Cómo se implementa el descenso de gradiente?

¡SmolML es nuestra respuesta! Es una librería de machine learning completamente funcional (aunque simplificada) construida usando **únicamente Python puro** y sus módulos básicos `collections`, `random` y `math`. Sin NumPy, sin SciPy, sin extensiones en C++ – solo Python, de principio a fin.

El objetivo no es competir con las librerías de producción en velocidad o características, sino proporcionar una implementación **transparente, comprensible y educativa** de los conceptos fundamentales del machine learning.

## Recorrido 📖

Puedes leer estas guías de las diferentes secciones de SmolML en cualquier orden, aunque esta lista presenta el orden recomendado para el aprendizaje.

- [SmolML - Core: Diferenciación Automática y Arrays N-Dimensionales](https://github.com/rodmarkun/SmolML/tree/main/smolml/core)
- [SmolML - Regresión: Prediciendo Valores Continuos](https://github.com/rodmarkun/SmolML/tree/main/smolml/models/regression)
- [SmolML - Redes Neuronales: Retropropagación al límite](https://github.com/rodmarkun/SmolML/tree/main/smolml/models/nn)
- [SmolML - Modelos de Árboles: ¡Decisiones, Decisiones!](https://github.com/rodmarkun/SmolML/tree/main/smolml/models/tree)
- [SmolML - K-Means: ¡Encontrando Grupos en tus Datos!](https://github.com/rodmarkun/SmolML/tree/main/smolml/models/unsupervised)
- [SmolML - Preprocesamiento: Haz que tus datos sean significativos](https://github.com/rodmarkun/SmolML/tree/main/smolml/preprocessing)
- [SmolML - ¡El cuarto de utilidades!](https://github.com/rodmarkun/SmolML/tree/main/smolml/utils)
 
## ¿Por qué SmolML? La Filosofía 🎓

Creemos que la mejor manera de entender verdaderamente temas complejos como el machine learning es a menudo **construirlos tú mismo**. Las librerías de producción son herramientas fantásticas, pero su complejidad interna y optimizaciones a veces pueden ocultar los principios fundamentales.

SmolML elimina estas capas para enfocarse en las ideas centrales:
* **Aprender desde Primeros Principios:** Cada componente principal está construido desde cero, permitiéndote rastrear la lógica desde operaciones básicas hasta algoritmos complejos.
* **Desmitificar la Magia:** Ver cómo conceptos como la diferenciación automática (autograd), algoritmos de optimización y arquitecturas de modelos están implementados en código.
* **Dependencias Mínimas:** Depender solo de la librería estándar de Python hace que el código sea accesible y fácil de explorar sin obstáculos de configuración externa.
* **Enfoque en la Claridad:** El código está escrito con comprensión, no rendimiento máximo, como objetivo principal.

Para aprender tanto como sea posible, recomendamos leer las guías, revisar el código y luego intentar implementar tus propias versiones de estos componentes.

## ¿Qué hay Dentro? Características 🛠️

SmolML explica los bloques de construcción esenciales para cualquier librería de ML:

* **La Fundación: Arrays Personalizados y Motor de Autograd:**
    * **Diferenciación Automática (`Value`):** Un motor de autograd simple que rastrea operaciones y calcula gradientes automáticamente – ¡el corazón del entrenamiento de redes neuronales!
    * **Arrays N-dimensionales (`MLArray`):** Una implementación de array personalizada inspirada en NumPy, que soporta operaciones matemáticas comunes necesarias para ML. Extremadamente ineficiente por estar escrita en Python, pero ideal para entender Arrays N-Dimensionales.

* **Preprocesamiento Esencial:**
    * **Escaladores (`StandardScaler`, `MinMaxScaler`):** Herramientas fundamentales para preparar tus datos, porque los algoritmos tienden a funcionar mejor cuando las características están en una escala similar.

* **Construye tus Propias Redes Neuronales:**
    * **Funciones de Activación:** No-linealidades como `relu`, `sigmoid`, `softmax`, `tanh` que permiten a las redes aprender patrones complejos. (Ver `smolml/utils/activation.py`)
    * **Inicializadores de Pesos:** Estrategias inteligentes (`Xavier`, `He`) para establecer pesos iniciales de red para entrenamiento estable. (Ver `smolml/utils/initializers.py`)
    * **Funciones de Pérdida:** Formas de medir el error del modelo (`mse_loss`, `binary_cross_entropy`, `categorical_cross_entropy`). (Ver `smolml/utils/losses.py`)
    * **Optimizadores:** Algoritmos como `SGD`, `Adam` y `AdaGrad` que actualizan los pesos del modelo basándose en gradientes para minimizar la pérdida. (Ver `smolml/utils/optimizers.py`)

* **Modelos ML Clásicos:**
    * **Regresión:** Implementaciones de regresión `Lineal` y `Polinomial`.
    * **Redes Neuronales:** Un framework flexible para construir redes neuronales feed-forward.
    * **Modelos Basados en Árboles:** Implementaciones de `Árbol de Decisión` y `Bosque Aleatorio` para clasificación y regresión.
    * **K-Means:** Algoritmo de clustering `KMeans` para agrupar puntos de datos similares.

## ¿Para Quién es SmolML? 🎯

* **Estudiantes:** Aprendiendo conceptos de ML por primera vez.
* **Desarrolladores:** Curiosos sobre el funcionamiento interno de las librerías de ML que usan diariamente.
* **Educadores:** Buscando una base de código simple y transparente para demostrar principios de ML.
* **Cualquier persona:** ¡Que disfrute aprender construyendo!

## ¡Limitaciones! ⚠️

Seamos cristalinos: SmolML está construido para **aprender**, no para romper récords de velocidad o manejar conjuntos de datos masivos.
* **Rendimiento:** Al ser Python puro, es MUUUY más lento que librerías que usan backends optimizados en C/C++/Fortran (como NumPy).
* **Escala:** Es más adecuado para conjuntos de datos pequeños y problemas de juguete donde entender la mecánica es más importante que el tiempo de cómputo.
* **Uso en Producción:** **No** uses SmolML para aplicaciones de producción. Mantente con librerías probadas en batalla como Scikit-learn, PyTorch, TensorFlow, JAX, etc., para tareas del mundo real.

¡Piénsalo como aprender a construir un motor de go-kart desde cero antes de conducir un auto de Fórmula 1. Te enseña los fundamentos de manera práctica!

## Comenzando

La mejor manera de usar SmolML es clonar este repositorio y explorar el código y ejemplos (si están disponibles).

```bash
git clone https://github.com/rodmarkun/SmolML
cd SmolML
# ¡Explora el código en el directorio smolml/!
```

También puedes ejecutar las múltiples pruebas en la carpeta `tests/`. Solo instala el `requirements.txt` (esto es para comparar SmolML contra otras librerías estándar como TensorFlow, sklearn, etc., y generar gráficos con matplotlib).

```bash
cd tests
pip install -r requirements.txt
```

## Contribuyendo

¡Las contribuciones siempre son bienvenidas! Si estás interesado en contribuir a SmolML, por favor haz fork del repositorio y crea una nueva rama para tus cambios. Cuando termines con tus cambios, envía un pull request para fusionar tus cambios en la rama principal.

## Traducción al Español

Esta traducción completa al español fue realizada por **[@Kyokopom](https://x.com/Kyokopom)** con el objetivo de hacer el machine learning más accesible para la comunidad hispanohablante. La traducción mantiene:

- ✅ **Terminología técnica consistente** en español
- ✅ **Valor educativo completo** del proyecto original  
- ✅ **Funcionalidad 100% preservada** en todo el código
- ✅ **Documentación bilingüe** con referencias al inglés cuando es útil

¡Esperamos que esta versión en español ayude a más personas a aprender machine learning desde cero!

## Apoyando SmolML

Si quieres apoyar SmolML, puedes:
- **Dar una estrella** :star: ¡al proyecto en Github!
- **Donar** :coin: a mi página de [Ko-fi](https://ko-fi.com/rodmarkun)!
- **Compartir** :heart: ¡el proyecto con tus amigos!

### Apoyando la Traducción

Si esta traducción al español te ha sido útil, también puedes:
- **Seguir** a [@Kyokopom](https://x.com/Kyokopom) en X/Twitter
- **Compartir** la versión en español con la comunidad hispanohablante
- **Contribuir** mejorando la traducción si encuentras algo que se pueda perfeccionar