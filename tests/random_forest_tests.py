import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from ml_array import MLArray
from random_forest import RandomForest

class TestRandomForest(unittest.TestCase):
    def setUp(self):
        """
        Set up a small subset of Iris dataset for testing
        """
        # Load iris dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Take a small subset (150 samples) while maintaining class distribution
        indices = []
        for class_idx in range(3):
            class_indices = [i for i, label in enumerate(y) if label == class_idx]
            indices.extend(class_indices[:50])
        
        self.X = X[indices]
        self.y = y[indices]
        self.feature_names = iris.feature_names
        self.class_names = iris.target_names
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.33, random_state=42
        )
        
        # Convert to lists for MLArray
        self.X_train = MLArray([[float(x) for x in row] for row in X_train])
        self.y_train = MLArray([float(y) for y in y_train])
        self.X_test = MLArray([[float(x) for x in row] for row in X_test])
        self.y_test = MLArray([float(y) for y in y_test])

    def test_iris_classification(self):
        """
        Test random forest classification on Iris dataset
        """
        print("\nTesting Random Forest on Iris Dataset...")
        print(f"Training samples: {len(self.X_train.data)}")
        print(f"Testing samples: {len(self.X_test.data)}")
        print(f"Features used: {self.feature_names}")
        
        configs = [
            {"n_trees": 5, "max_depth": 3},
            {"n_trees": 10, "max_depth": 5},
            {"n_trees": 50, "max_depth": 15}
        ]
        
        for config in configs:
            print(f"\nTesting with {config['n_trees']} trees, max_depth={config['max_depth']}")
            
            rf = RandomForest(
                n_trees=config['n_trees'],
                max_depth=config['max_depth'],
                min_samples_split=2,
                task="classification"
            )
            
            # Train forest
            rf.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = rf.predict(self.X_test)
            
            # Calculate accuracy
            correct = sum(1 for pred, true in zip(y_pred.data, self.y_test.data) 
                        if pred == true)
            accuracy = correct / len(self.y_test.data)
            
            print(f"Classification Accuracy: {accuracy:.3f}")
            
            # Print detailed predictions vs true values
            print("\nPredictions vs True Values:")
            for pred, true in zip(y_pred.data, self.y_test.data):
                pred_val = pred.data if hasattr(pred, 'data') else pred
                true_val = true.data if hasattr(true, 'data') else true
                print(f"Predicted: {self.class_names[int(pred_val)]}, "
                      f"True: {self.class_names[int(true_val)]}")
            
            self.assertGreaterEqual(accuracy, 0.6,
                f"Classification accuracy ({accuracy:.3f}) too low for {config['n_trees']} trees")

if __name__ == '__main__':
    unittest.main()