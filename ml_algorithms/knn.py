"""
K-Nearest Neighbors (KNN) - Production Ready Implementation
Using scikit-learn for classification tasks
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os


class ProductionKNN:
    """
    Production-ready KNN Classifier wrapper
    Includes preprocessing, hyperparameter tuning, and model persistence
    """
    
    def __init__(self, n_neighbors=5, weights='uniform', metric='minkowski'):
        """
        Initialize KNN classifier
        n_neighbors: number of neighbors to use
        weights: 'uniform' or 'distance' (weight by inverse distance)
        metric: distance metric ('minkowski', 'euclidean', 'manhattan')
        """
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric
        )
        self.scaler = StandardScaler()  # Feature scaling is crucial for KNN
        self.label_encoder = LabelEncoder()  # Encode categorical labels
        self.is_fitted = False
        self.best_params = None
    
    def preprocess_data(self, X, y=None, fit_scaler=True, fit_labels=True):
        """
        Preprocess features and labels
        """
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features (very important for KNN)
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Encode labels if provided
        if y is not None and fit_labels:
            y = self.label_encoder.fit_transform(y)
        elif y is not None:
            y = self.label_encoder.transform(y)
        
        return X_scaled, y
    
    def fit(self, X, y):
        """
        Train the KNN model with preprocessing
        """
        # Preprocess data
        X_processed, y_processed = self.preprocess_data(X, y, fit_scaler=True, fit_labels=True)
        
        # Train model
        self.model.fit(X_processed, y_processed)
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """
        Make predictions on new data
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Preprocess using fitted scaler
        X_processed, _ = self.preprocess_data(X, fit_scaler=False, fit_labels=False)
        
        # Predict and decode labels
        predictions = self.model.predict(X_processed)
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X):
        """
        Get probability estimates for predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_processed, _ = self.preprocess_data(X, fit_scaler=False, fit_labels=False)
        return self.model.predict_proba(X_processed)
    
    def evaluate(self, X, y):
        """
        Comprehensive model evaluation
        """
        predictions = self.predict(X)
        
        # Encode true labels for comparison
        y_encoded = self.label_encoder.transform(y)
        pred_encoded = self.label_encoder.transform(predictions)
        
        return {
            'accuracy': accuracy_score(y_encoded, pred_encoded),
            'classification_report': classification_report(y, predictions),
            'confusion_matrix': confusion_matrix(y_encoded, pred_encoded)
        }
    
    def tune_hyperparameters(self, X, y, cv=5):
        """
        Perform grid search to find best hyperparameters
        """
        X_processed, y_processed = self.preprocess_data(X, y, fit_scaler=True, fit_labels=True)
        
        # Define parameter grid
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        
        # Perform grid search
        grid_search = GridSearchCV(
            KNeighborsClassifier(),
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_processed, y_processed)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.is_fitted = True
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation
        """
        X_processed, y_processed = self.preprocess_data(X, y, fit_scaler=True, fit_labels=True)
        scores = cross_val_score(self.model, X_processed, y_processed, cv=cv, scoring='accuracy')
        
        return {
            'mean_cv_score': scores.mean(),
            'std_cv_score': scores.std(),
            'all_scores': scores
        }
    
    def save_model(self, filepath='models/knn_model.pkl'):
        """
        Save trained model to disk
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'is_fitted': self.is_fitted,
            'best_params': self.best_params
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/knn_model.pkl'):
        """
        Load trained model from disk
        """
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.label_encoder = data['label_encoder']
        self.is_fitted = data['is_fitted']
        self.best_params = data.get('best_params')
        print(f"Model loaded from {filepath}")


# Example usage with production best practices
def demonstrate_production_knn():
    """Demonstrate production-ready KNN classifier"""
    print("=== Production KNN Classifier Demo ===\n")
    
    # Sample data: Iris flower classification
    # Features: [sepal_length, sepal_width, petal_length, petal_width]
    X = np.array([
        [5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [6.7, 3.1, 4.4, 1.4],
        [5.7, 2.8, 4.5, 1.3], [6.3, 3.3, 6.0, 2.5], [5.8, 2.7, 5.1, 1.9],
        [7.1, 3.0, 5.9, 2.1], [6.3, 2.9, 5.6, 1.8], [4.6, 3.1, 1.5, 0.2],
        [5.0, 3.6, 1.4, 0.2], [5.5, 2.5, 4.0, 1.3], [6.5, 2.8, 4.6, 1.5],
        [5.7, 2.5, 5.0, 2.0], [6.4, 2.7, 5.3, 1.9], [4.8, 3.4, 1.6, 0.2],
        [5.4, 3.9, 1.7, 0.4], [6.0, 2.9, 4.5, 1.5], [5.6, 2.9, 3.6, 1.3],
        [6.7, 3.1, 5.6, 2.4], [6.9, 3.1, 5.1, 2.3]
    ])
    y = np.array(['setosa', 'setosa', 'versicolor', 'versicolor', 'virginica', 'virginica',
                  'virginica', 'virginica', 'setosa', 'setosa', 'versicolor', 'versicolor',
                  'virginica', 'virginica', 'setosa', 'setosa', 'versicolor', 'versicolor',
                  'virginica', 'virginica'])
    
    print("Dataset Info:")
    print(f"Samples: {len(X)}")
    print(f"Features: Sepal Length, Sepal Width, Petal Length, Petal Width")
    print(f"Classes: {np.unique(y)}\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples\n")
    
    # Create and train model
    knn = ProductionKNN(n_neighbors=5, weights='distance')
    knn.fit(X_train, y_train)
    print("✓ Model trained successfully\n")
    
    # Evaluate on test set
    print("=== Model Evaluation ===")
    metrics = knn.evaluate(X_test, y_test)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"\nClassification Report:\n{metrics['classification_report']}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}\n")
    
    # Cross-validation
    print("=== Cross-Validation ===")
    cv_results = knn.cross_validate(X_train, y_train, cv=3)
    print(f"Mean CV Accuracy: {cv_results['mean_cv_score']:.4f}")
    print(f"Std CV Accuracy: {cv_results['std_cv_score']:.4f}\n")
    
    # Hyperparameter tuning
    print("=== Hyperparameter Tuning ===")
    tuning_results = knn.tune_hyperparameters(X_train, y_train, cv=3)
    print(f"Best Parameters: {tuning_results['best_params']}")
    print(f"Best CV Score: {tuning_results['best_score']:.4f}\n")
    
    # Make predictions with probability
    print("=== Predictions ===")
    new_samples = np.array([[5.0, 3.5, 1.5, 0.2], [6.5, 3.0, 5.0, 2.0]])
    predictions = knn.predict(new_samples)
    probabilities = knn.predict_proba(new_samples)
    
    print("New samples to classify:")
    for i, (sample, pred, proba) in enumerate(zip(new_samples, predictions, probabilities)):
        print(f"  Sample {i+1}: {sample}")
        print(f"    Predicted: {pred}")
        print(f"    Probabilities: {dict(zip(knn.label_encoder.classes_, proba))}")
    
    # Save model
    print("\n=== Model Persistence ===")
    knn.save_model('models/iris_knn_model.pkl')
    
    # Load and verify
    new_knn = ProductionKNN()
    new_knn.load_model('models/iris_knn_model.pkl')
    verification = new_knn.predict(new_samples)
    print(f"Verification (loaded model): {verification[0]}")


if __name__ == "__main__":
    demonstrate_production_knn()
