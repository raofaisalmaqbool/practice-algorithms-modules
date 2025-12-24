"""
Linear Regression - Production Ready Implementation
Using scikit-learn for real-world applications
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os


class ProductionLinearRegression:
    """
    Production-ready Linear Regression wrapper
    Includes preprocessing, evaluation, and model persistence
    """
    
    def __init__(self, model_type='linear', alpha=1.0):
        """
        Initialize model
        model_type: 'linear', 'ridge', or 'lasso'
        alpha: regularization strength for Ridge/Lasso
        """
        # Select model type
        if model_type == 'ridge':
            self.model = Ridge(alpha=alpha)
        elif model_type == 'lasso':
            self.model = Lasso(alpha=alpha)
        else:
            self.model = LinearRegression()
        
        self.scaler = StandardScaler()  # Feature scaling
        self.is_fitted = False
        self.feature_names = None
    
    def preprocess_data(self, X, y=None, fit_scaler=True):
        """
        Preprocess features: handle missing values and scale
        """
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Fill missing values with mean
        X = X.fillna(X.mean())
        
        # Scale features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y
    
    def fit(self, X, y):
        """
        Train the model with preprocessing
        """
        # Preprocess data
        X_processed, y = self.preprocess_data(X, y, fit_scaler=True)
        
        # Train model
        self.model.fit(X_processed, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """
        Make predictions on new data
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Preprocess using fitted scaler
        X_processed, _ = self.preprocess_data(X, fit_scaler=False)
        
        return self.model.predict(X_processed)
    
    def evaluate(self, X, y):
        """
        Comprehensive model evaluation
        Returns multiple metrics
        """
        predictions = self.predict(X)
        
        metrics = {
            'r2_score': r2_score(y, predictions),
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions)
        }
        
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation
        """
        X_processed, y = self.preprocess_data(X, y, fit_scaler=True)
        scores = cross_val_score(self.model, X_processed, y, cv=cv, scoring='r2')
        
        return {
            'mean_cv_score': scores.mean(),
            'std_cv_score': scores.std(),
            'all_scores': scores
        }
    
    def save_model(self, filepath='models/linear_regression_model.pkl'):
        """
        Save trained model to disk
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/linear_regression_model.pkl'):
        """
        Load trained model from disk
        """
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_fitted = data['is_fitted']
        print(f"Model loaded from {filepath}")
    
    def get_feature_importance(self):
        """
        Get model coefficients (feature importance)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return {
            'coefficients': self.model.coef_,
            'intercept': self.model.intercept_
        }


# Example usage with production best practices
def demonstrate_production_linear_regression():
    """
    Demonstrate production-ready linear regression
    """
    print("=== Production Linear Regression Demo ===\n")
    
    # Sample data: House price prediction
    # Features: [size_sqft, bedrooms, age_years]
    X = np.array([
        [1500, 3, 10], [1800, 4, 8], [2400, 4, 5],
        [2000, 3, 12], [1600, 2, 15], [2200, 4, 7],
        [1400, 2, 20], [1900, 3, 9], [2100, 3, 6],
        [1700, 3, 11], [2300, 4, 4], [1550, 2, 18]
    ])
    y = np.array([300000, 360000, 450000, 380000, 310000, 420000,
                  280000, 370000, 410000, 340000, 440000, 295000])
    
    print("Dataset Info:")
    print(f"Samples: {len(X)}")
    print(f"Features: Size (sqft), Bedrooms, Age (years)")
    print(f"Target: House Price ($)\n")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples\n")
    
    # Create and train model
    model = ProductionLinearRegression(model_type='linear')
    model.fit(X_train, y_train)
    print("✓ Model trained successfully\n")
    
    # Evaluate on test set
    print("=== Model Evaluation ===")
    metrics = model.evaluate(X_test, y_test)
    print(f"R² Score: {metrics['r2_score']:.4f}")
    print(f"RMSE: ${metrics['rmse']:,.2f}")
    print(f"MAE: ${metrics['mae']:,.2f}\n")
    
    # Cross-validation
    print("=== Cross-Validation ===")
    cv_results = model.cross_validate(X_train, y_train, cv=3)
    print(f"Mean CV R² Score: {cv_results['mean_cv_score']:.4f}")
    print(f"Std CV R² Score: {cv_results['std_cv_score']:.4f}\n")
    
    # Make predictions
    print("=== Predictions ===")
    new_houses = np.array([[1750, 3, 8], [2500, 4, 3]])
    predictions = model.predict(new_houses)
    print("New houses to predict:")
    for i, (features, price) in enumerate(zip(new_houses, predictions)):
        print(f"  House {i+1}: {features[0]} sqft, {features[1]} bed, {features[2]} yrs → ${price:,.2f}")
    
    # Feature importance
    print("\n=== Model Coefficients ===")
    importance = model.get_feature_importance()
    print(f"Coefficients: {importance['coefficients']}")
    print(f"Intercept: {importance['intercept']:.2f}")
    
    # Save model
    print("\n=== Model Persistence ===")
    model.save_model('models/house_price_model.pkl')
    
    # Load and verify
    new_model = ProductionLinearRegression()
    new_model.load_model('models/house_price_model.pkl')
    verification = new_model.predict(new_houses)
    print(f"Verification (loaded model): ${verification[0]:,.2f}")


if __name__ == "__main__":
    demonstrate_production_linear_regression()
