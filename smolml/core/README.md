# SmolML - Core: Automatic Differentiation & N-Dimensional Arrays

Welcome to the core of SmolML! This is where the magic begins if you want to understand how machine learning models, especially neural networks, learn from data. We'll break down two fundamental concepts: how computers *calculate* the direction for learning (automatic differentiation) and how we handle the multi-dimensional data involved.

This part of SmolML focuses on handling this data and calculating gradients automatically. Why gradients? Why automatic? Let's dive in.

## Why Do We Need Gradients Anyway?

Think about teaching a computer to recognize a cat in a photo. The computer makes a prediction based on its current internal settings (parameters or weights). Initially, these settings are random, so the prediction is likely wrong. We measure *how wrong* using a "loss function" (we have those in `smolml\utils\losses.py`, explained in another section) – a lower loss means a better prediction.

The goal is to adjust the computer's parameters to *minimize* this loss. But how do we know *which way* to adjust each parameter? Should we increase it? Decrease it? By how much?

This is where **gradients** come in. The gradient of the loss function with respect to a specific parameter tells us the "slope" of the loss at that parameter's current value. It points in the direction of the *steepest increase* in the loss. So, if we want to *decrease* the loss, we nudge the parameter in the *opposite* direction of the gradient. The size of the gradient also tells us how sensitive the loss is to that parameter – a larger gradient suggests a bigger adjustment might be needed.

Calculating these gradients for every parameter allows the model to iteratively improve, step-by-step, reducing its error. This process is the heart of training most ML models.

## Why "Automatic" Differentiation?

Okay, so we need gradients. For a simple function like $y = a \times b + c$, we can find the gradients ($\frac{\partial y}{\partial a}$, $\frac{\partial y}{\partial b}$, $\frac{\partial y}{\partial c}$) using basic calculus rules (like the chain rule).

But modern neural networks are *vastly* more complex. They are essentially giant, nested mathematical functions with potentially millions of parameters. Calculating all those gradients manually is practically impossible and incredibly prone to errors.

**Automatic Differentiation (AutoDiff)** is the solution. It's a technique where the computer itself keeps track of every single mathematical operation performed, building a computational graph. Then, by applying the chain rule systematically backwards through this graph (a process called **backpropagation**), it can efficiently compute the gradient of the final output (the loss) with respect to every single input and parameter involved.

## Implementing AutoDiff with `Value`

This library uses the `Value` class to implement Automatic Differentiation.

*(If you want a deep dive, the concept of backpropagation and automatic differentiation is greatly explained in this [Andrej Karpathy video](https://www.youtube.com/watch?v=VMj-3S1tku0), highly recommended!)*

**What is a `Value`?**

Think of a `Value` object as a smart container for a single number (a scalar). It doesn't just hold the number; it prepares for gradient calculations. It stores:
1.  `data`: The actual numerical value (e.g., 5.0, -3.2).
2.  `grad`: The gradient of the *final output* of our entire computation graph with respect to *this* specific `Value`'s data. It starts at 0 and gets filled during backpropagation.

**Building the Calculation Trail (Computational Graph)**

The clever part happens when you perform mathematical operations (like `+`, `*`, `exp`, `tanh`, etc.) using `Value` objects. Each time you combine `Value` objects, you create a *new* `Value` object. This new object internally remembers:
* Its own `data` (the result of the operation).
* The operation that created it (`_op`).
* The original `Value` objects that were its inputs (`_children` or `_prev`).

This implicitly builds a computational graph, step-by-step, tracing the lineage of the calculation from the initial inputs to the final result. For example:

```python
a = Value(2.0)
b = Value(3.0)
c = Value(4.0)
# d 'knows' it resulted from a * b
d = a * b  # d._op = "*", d._prev = {a, b}
# y 'knows' it resulted from d + c
y = d + c  # y._op = "+", y._prev = {d, c}
```

**Backpropagation: Calculating Gradients Automatically**

So, how do we get the gradients without doing the calculus ourselves? By calling the `.backward()` method on the final `Value` object (the one representing our overall loss, like y in the simple example).

Here’s the process conceptually:

1. Start at the end (`y`). The gradient of `y` with respect to itself is 1. So, we set `y.grad = 1`.

2. `y` knows it came from `d + c`. Using the chain rule for addition ($\frac{\partial y}{\partial d} = 1$, $\frac{\partial y}{\partial c} = 1$), it passes its gradient back. It tells `d` to add `1 * y.grad` to its gradient, and tells `c` to add `1 * y.grad` to its gradient. So, `d.grad` becomes 1 and `c.grad` becomes 1.

3. `d` knows it came from `a * b`. Using the chain rule for multiplication ($\frac{\partial d}{\partial a} = b$, $\frac{\partial d}{\partial b} = a$), it passes its received gradient (`d.grad`) back. It tells `a` to add `b.data * d.grad` to its gradient, and tells `b` to add `a.data * d.grad` to its gradient. So `a.grad` becomes $3.0 \times 1 = 3.0$ and `b.grad` becomes $2.0 \times 1 = 2.0$.

4. This process continues recursively backwards through the graph until all `Value` objects that contributed to the final result have their gradients computed.

Each `Value` object stores a tiny function (`_backward`) that knows how to compute the local gradients for the operation it represents (+, *, tanh, etc.). The `.backward()` method orchestrates calling these small functions in the correct reverse order (using a topological sort) to implement the chain rule across the entire graph.

After calling `.backward()`, `a.grad`, `b.grad`, and `c.grad` hold the gradients, telling you exactly how sensitive the final output `y` is to changes in the initial inputs `a`, `b`, and `c`.

## Handling N-Dimensional Arrays of Values with `MLArray`

> **IMPORTANT NOTE**: `MLArray` is by far the most complex class of the library. If you are implementing this library by yourself while following along, I recommend just copying the provided `mlarray.py` for now, and make your own implementation at the end, as making it from scratch will probably take you a lot of time (and headaches!).

While `Value` objects handle the core AutoDiff for single numbers, machine learning thrives on vectors, matrices, and higher-dimensional tensors. We need an efficient way to manage collections of these smart Value objects. This is the job of the MLArray class.

What is an `MLArray`?

Think of `MLArray` as an N-dimensional array (like NumPy arrays, but built from scratch here). It can be a list (1D), a list-of-lists (2D matrix), or nested deeper for higher dimensions.

The crucial difference from a standard Python list is that every number you put into an MLArray is automatically converted into a `Value` object. This happens recursively in the `_process_data` method during initialization.

```python
# Creates a 2x2 MLArray; 1.0, 2.0, 3.0, 4.0 are now Value objects
my_matrix = MLArray([[1.0, 2.0], [3.0, 4.0]])
```

This ensures that any calculations performed using the `MLArray` will automatically build the computational graph needed for backpropagation.

**Operations on Arrays**

The idea of `MLArray` is to support many standard numerical operations, designed to work seamlessly with the underlying `Value` objects:

- **Element-wise operations**: `+`, `-`, `*`, `/`, `**` (power), `exp()`, `log()`, `tanh()`, etc.. When you add two `MLArray`s, the library iterates through corresponding elements and performs the `Value.__add__` (or the corresponding operation, like `__mul__` for multiplication, and so on) operation on each pair (handled by `_element_wise_operation`). This builds the graph element by element.
- **Matrix Multiplication**: `matmul()` or the `@` operator. This performs the dot product logic, correctly combining the underlying `Value` multiplications and additions to construct the appropriate graph structure.
- **Reductions**: `sum()`, `mean()`, `max()`, `min()`, `std()`. These operations aggregate `Value` objects across the array or along specified axes, again building the graph correctly.
- **Shape Manipulation**: `transpose()` (or `.T`), `reshape()`. These rearrange the `Value` objects into different array shapes without changing the objects themselves, preserving the graph connections.
- **Broadcasting**: Operations between arrays of different but compatible shapes (e.g., adding a vector to each row of a matrix) are handled automatically, figuring out how to apply operations element-wise based on broadcasting rules (`_broadcast_shapes`, `_broadcast_and_apply`).

**Getting Gradients for Arrays**

Just like with a single `Value`, you can perform a complex sequence of `MLArray` operations. Typically, your final result will be a single scalar `Value` representing the loss (often achieved by using `.sum()` or `.mean()` at the end).

You then call `.backward()` on this final scalar `MLArray` (or the `Value` inside it). This triggers the backpropagation process through all the interconnected `Value` objects that were part of the calculation.

After `.backward()` has run, you can access the computed gradients for any `MLArray` involved in the original computation using its `.grad()` method. This returns a new `MLArray` of the same shape, where each element holds the gradient corresponding to the original `Value` object.

```python
# Example using ml_array.py and value.py

# Input data (e.g., batch of 2 samples, 2 features each)
x = MLArray([[1.0, 2.0], [3.0, 4.0]])
# Weights (e.g., simple linear layer mapping 2 features to 1 output)
w = MLArray([[0.5], [-0.5]])

# --- Forward Pass (Builds the graph) ---
# Matrix multiplication
z = x @ w
# Apply activation function (element-wise)
a = z.tanh()
# Calculate total loss (e.g., sum of activations)
L = a.sum() # L is now a scalar MLArray containing one Value

# --- Backward Pass (Calculates gradients) ---
L.backward() # Triggers Value.backward() across the graph

# --- Inspect Gradients ---
# Gradient of L w.r.t. each element in x
print("Gradient w.r.t. x:")
print(x.grad())
# Gradient of L w.r.t. each element in w
print("\nGradient w.r.t. w:")
print(w.grad())
```

**Utility Functions & Properties**

The library also includes helpers:

- `zeros()`, `ones()`, `randn()`: Create `MLArray`s filled with zeros, ones, or standard random numbers.
- `xavier_uniform()`, `xavier_normal()`: Implement common strategies for initializing weight matrices in neural networks.
- `to_list()`: Convert an `MLArray` back to a standard Python list (this extracts the `.data` and discards gradient information).
- `shape`: A property to quickly get the dimensions tuple of the `MLArray`.
- `size()`: Returns the total number of elements in the `MLArray`.

## Putting It Together

With these two core components:

- A `Value` class that encapsulates a single number and enables automatic differentiation by tracking operations and applying the chain rule via backpropagation.
- An `MLArray` class that organizes these `Value` objects into N-dimensional arrays and extends mathematical operations (like matrix multiplication, broadcasting) to work seamlessly within the AutoDiff framework.

We have the essential toolkit to build and, more importantly, train various machine learning models, like simple neural networks, entirely from scratch. We can now define network layers, compute loss functions, and use the calculated gradients to update parameters and make our models learn!