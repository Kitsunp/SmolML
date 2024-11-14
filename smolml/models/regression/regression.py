import sys
import smolml.utils.initializers as initializers
import smolml.utils.optimizers as optimizers
from smolml.core.ml_array import MLArray
import smolml.core.ml_array as ml_array
import smolml.utils.losses as losses
import smolml.utils.memory as memory

"""
//////////////////
/// REGRESSION ///
//////////////////
"""

class Regression:
    """
    Base class for regression algorithms implementing common functionality.
    Provides framework for fitting models using gradient descent optimization.
    Specific regression types should inherit from this class.
    """
    def __init__(self, input_size: int, loss_function: callable = losses.mse_loss, optimizer: optimizers.Optimizer = optimizers.SGD, initializer: initializers.WeightInitializer = initializers.XavierUniform):
        """
        Initializes base regression model with common parameters.
        """
        self.input_size = input_size
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.initializer = initializer
        self.weights = self.initializer.initialize((self.input_size,))
        self.bias = ml_array.ones((1))

    def fit(self, X, y, iterations: int = 100, verbose: bool = True, print_every: int = 1):
        """
        Trains the regression model using gradient descent.
        """
        X, y = MLArray.ensure_array(X, y)
        losses = []
        for i in range(iterations):
            # Make prediction 
            y_pred = self.predict(X)
            # Compute loss
            loss = self.loss_function(y, y_pred)
            losses.append(loss.data.data)
            # Backward pass
            loss.backward()

            # Update parameters
            self.weights, self.bias = self.optimizer.update(self, self.__class__.__name__, param_names=("weights", "bias"))

            # Reset gradients
            X, y = self.restart(X, y)

            if verbose:
                if (i+1) % print_every == 0:
                    print(f"Iteration {i + 1}/{iterations}, Loss: {loss.data}")

        return losses

    def restart(self, X, y):
        """
        Resets gradients for all parameters and data for next iteration.
        """
        X = X.restart()
        y = y.restart()
        self.weights = self.weights.restart()
        self.bias = self.bias.restart()
        return X, y
    
    def predict(self, X):
        """
        Abstract method for making predictions.
        Must be implemented by specific regression classes.
        """
        raise NotImplementedError("Regression is only base class for Regression algorithms, use one of the classes that inherit from it.")
    
    def __repr__(self):
        """
        Returns a string representation of the regression model.
        Includes model type, parameters, optimizer details, and memory usage.
        """
        try:
            import os
            terminal_width = os.get_terminal_size().columns
        except Exception:
            terminal_width = 80

        # Create header
        header = f"{self.__class__.__name__} Model"
        separator = "=" * terminal_width
        
        # Get size information
        size_info = memory.calculate_regression_size(self)
        
        # Model structure
        structure_info = [
            f"Input Size: {self.input_size}",
            f"Loss Function: {self.loss_function.__name__}",
            f"Optimizer: {self.optimizer.__class__.__name__}(learning_rate={self.optimizer.learning_rate})"
        ]
        
        # Parameters count
        total_params = self.weights.size() + self.bias.size()
        
        # Memory breakdown
        memory_info = ["Memory Usage:"]
        memory_info.append(
            f"  Parameters: {memory.format_size(size_info['parameters']['total'])} "
            f"(weights: {memory.format_size(size_info['parameters']['weights_size'])}, "
            f"bias: {memory.format_size(size_info['parameters']['bias_size'])})"
        )
        
        # Add optimizer state information if it exists
        if size_info['optimizer']['state']:
            opt_size = sum(size_info['optimizer']['state'].values())
            memory_info.append(f"  Optimizer State: {memory.format_size(opt_size)}")
            for key, value in size_info['optimizer']['state'].items():
                memory_info.append(f"    {key}: {memory.format_size(value)}")
        
        memory_info.append(f"  Base Objects: {memory.format_size(size_info['optimizer']['size'])}")
        memory_info.append(f"Total Memory: {memory.format_size(size_info['total'])}")
        
        # Combine all parts
        return (
            f"\n{header}\n{separator}\n\n"
            + "\n".join(structure_info)
            + f"\n\nTotal Parameters: {total_params:,}\n\n"
            + "\n".join(memory_info)
            + f"\n{separator}\n"
        )

class LinearRegression(Regression):
   """
   Implements linear regression using gradient descent optimization.
   The model learns to fit: y = X @ weights + bias
   """
   def __init__(self, input_size: int, loss_function: callable = losses.mse_loss, optimizer: optimizers.Optimizer = optimizers.SGD, initializer: initializers.WeightInitializer = initializers.XavierUniform):
       """
       Initializes regression model with training parameters.
       """
       super().__init__(input_size, loss_function, optimizer, initializer)

   def predict(self, X):
       """
       Makes predictions using linear model equation.
       """
       if not isinstance(X, MLArray):
            raise TypeError(f"Input data must be MLArray, not {type(X)}")
       return X @ self.weights + self.bias

class PolynomialRegression(Regression):
   """
   Extends linear regression to fit polynomial relationships.
   Transforms features into polynomial terms before fitting.
   """
   def __init__(self, input_size: int, degree: int, loss_function: callable = losses.mse_loss, optimizer: optimizers.Optimizer = optimizers.SGD, initializer: initializers.WeightInitializer = initializers.XavierUniform):
       """
       Initializes polynomial model with degree and training parameters.
       """
       # Initialize with degree for number of weights needed
       super().__init__(degree, loss_function, optimizer, initializer)
       self.degree = degree
       
   def transform_features(self, X):
       """
       Creates polynomial features up to specified degree.
       For input X and degree 2, outputs [X, X^2].
       """
       features = [X]
       current = X
       for d in range(2, self.degree + 1):
           # Use element-wise multiplication for power
           current = current * X
           features.append(current)
           
       # Create a new MLArray for concatenated features
       new_data = []
       for i in range(len(X.data)):
           row = []
           for feature in features:
               row.append(feature.data[i][0])  # Extract value from each feature
           new_data.append(row)
           
       return MLArray(new_data)

   def predict(self, X):
       """
       Makes predictions after transforming features to polynomial form.
       """
       if not isinstance(X, MLArray):
            raise TypeError(f"Input data must be MLArray, not {type(X)}")
       X_poly = self.transform_features(X)
       return X_poly @ self.weights + self.bias

   def fit(self, X, y, iterations: int = 100, verbose: bool = True, print_every: int = 1):
       """
       Transforms features to polynomial form before training.
       """
       X_poly = self.transform_features(X)
       return super().fit(X_poly, y, iterations, verbose, print_every)
   
   def __repr__(self):
        """
        Enhanced repr for polynomial regression including degree information.
        """
        try:
            import os
            terminal_width = os.get_terminal_size().columns
        except Exception:
            terminal_width = 80

        # Create header
        header = "Polynomial Regression Model"
        separator = "=" * terminal_width
        
        # Get size information
        size_info = memory.calculate_regression_size(self)
        
        # Model structure
        structure_info = [
            f"Original Input Size: {self.input_size}",
            f"Polynomial Degree: {self.degree}",
            f"Loss Function: {self.loss_function.__name__}",
            f"Optimizer: {self.optimizer.__class__.__name__}(learning_rate={self.optimizer.learning_rate})"
        ]
        
        # Parameters count
        total_params = self.weights.size() + self.bias.size()
        
        # Memory breakdown
        memory_info = ["Memory Usage:"]
        memory_info.append(
            f"  Parameters: {memory.format_size(size_info['parameters']['total'])} "
            f"(weights: {memory.format_size(size_info['parameters']['weights_size'])}, "
            f"bias: {memory.format_size(size_info['parameters']['bias_size'])})"
        )
        
        if size_info['optimizer']['state']:
            opt_size = sum(size_info['optimizer']['state'].values())
            memory_info.append(f"  Optimizer State: {memory.format_size(opt_size)}")
            for key, value in size_info['optimizer']['state'].items():
                memory_info.append(f"    {key}: {memory.format_size(value)}")
        
        memory_info.append(f"  Base Objects: {memory.format_size(size_info['optimizer']['size'])}")
        memory_info.append(f"Total Memory: {memory.format_size(size_info['total'])}")
        
        # Combine all parts
        return (
            f"\n{header}\n{separator}\n\n"
            + "\n".join(structure_info)
            + f"\n\nTotal Parameters: {total_params:,}\n\n"
            + "\n".join(memory_info)
            + f"\n{separator}\n"
        )