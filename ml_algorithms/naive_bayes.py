"""
Naive Bayes Classifier - Production Ready Implementation
Using scikit-learn for probabilistic classification
"""
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib
import os


class ProductionNaiveBayes:
    """
    Production-ready Naive Bayes Classifier wrapper
    Supports Gaussian, Multinomial, and Bernoulli variants
    """
    
    def __init__(self, variant='gaussian', **kwargs):
        """
        Initialize Naive Bayes classifier
        variant: 'gaussian', 'multinomial', or 'bernoulli'
        **kwargs: additional parameters for the specific variant
        """
        # Select Naive Bayes variant
        if variant == 'multinomial':
            self.model = MultinomialNB(**kwargs)
        elif variant == 'bernoulli':
            self.model = BernoulliNB(**kwargs)
        else:  # gaussian (default)
            self.model = GaussianNB(**kwargs)
        
        self.variant = variant
        self.scaler = StandardScaler()  # For Gaussian NB
        self.label_encoder = LabelEncoder()
        self.vectorizer = None  # For text data
        self.is_fitted = False
    
    def preprocess_data(self, X, y=None, fit_scaler=True, fit_labels=True):
        """
        Preprocess features and labels
        """
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features only for Gaussian NB
        if self.variant == 'gaussian':
            if fit_scaler:
                X_processed = self.scaler.fit_transform(X)
            else:
                X_processed = self.scaler.transform(X)
        else:
            X_processed = X.values
        
        # Encode labels if provided
        if y is not None and fit_labels:
            y = self.label_encoder.fit_transform(y)
        elif y is not None:
            y = self.label_encoder.transform(y)
        
        return X_processed, y
    
    def fit(self, X, y):
        """
        Train the Naive Bayes model
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
    
    def predict_log_proba(self, X):
        """
        Get log probability estimates
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_processed, _ = self.preprocess_data(X, fit_scaler=False, fit_labels=False)
        return self.model.predict_log_proba(X_processed)
    
    def evaluate(self, X, y):
        """
        Comprehensive model evaluation
        """
        predictions = self.predict(X)
        
        # Encode true labels for comparison
        y_encoded = self.label_encoder.transform(y)
        pred_encoded = self.label_encoder.transform(predictions)
        
        metrics = {
            'accuracy': accuracy_score(y_encoded, pred_encoded),
            'classification_report': classification_report(y, predictions),
            'confusion_matrix': confusion_matrix(y_encoded, pred_encoded)
        }
        
        # Add ROC-AUC for binary classification
        if len(self.label_encoder.classes_) == 2:
            proba = self.predict_proba(X)[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_encoded, proba)
        
        return metrics
    
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
    
    def get_class_priors(self):
        """
        Get class prior probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return dict(zip(self.label_encoder.classes_, self.model.class_prior_))
    
    def save_model(self, filepath='models/naive_bayes_model.pkl'):
        """
        Save trained model to disk
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'variant': self.variant,
            'is_fitted': self.is_fitted
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/naive_bayes_model.pkl'):
        """
        Load trained model from disk
        """
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.label_encoder = data['label_encoder']
        self.variant = data['variant']
        self.is_fitted = data['is_fitted']
        print(f"Model loaded from {filepath}")


# Example usage with production best practices
def demonstrate_production_naive_bayes():
    """Demonstrate production-ready Naive Bayes classifier"""
    print("=== Production Naive Bayes Classifier Demo ===\n")
    
    # Sample data: Medical diagnosis based on symptoms
    # Features: [fever(°F), cough(0-10), fatigue(0-10), breathing_difficulty(0-10)]
    X = np.array([
        [99.5, 2, 3, 1], [100.2, 3, 4, 2], [98.6, 1, 2, 0],  # Common Cold
        [98.7, 2, 1, 0], [99.0, 3, 2, 1], [98.9, 1, 1, 0],
        [102.5, 7, 8, 6], [103.0, 8, 9, 7], [102.8, 9, 8, 6],  # Flu
        [103.2, 8, 9, 7], [102.0, 7, 7, 5], [103.5, 9, 9, 8],
        [98.6, 0, 1, 0], [98.7, 0, 0, 0], [98.5, 1, 1, 0],  # Healthy
        [98.8, 0, 1, 0], [98.6, 1, 0, 0], [98.7, 0, 0, 0],
        [100.5, 5, 6, 4], [101.0, 6, 7, 5], [100.8, 5, 6, 3],  # Bronchitis
        [101.2, 6, 5, 4], [100.5, 5, 6, 4], [101.5, 7, 7, 5]
    ])
    y = np.array(['cold', 'cold', 'cold', 'cold', 'cold', 'cold',
                  'flu', 'flu', 'flu', 'flu', 'flu', 'flu',
                  'healthy', 'healthy', 'healthy', 'healthy', 'healthy', 'healthy',
                  'bronchitis', 'bronchitis', 'bronchitis', 'bronchitis', 'bronchitis', 'bronchitis'])
    
    print("Dataset Info:")
    print(f"Samples: {len(X)}")
    print(f"Features: Fever (°F), Cough (0-10), Fatigue (0-10), Breathing Difficulty (0-10)")
    print(f"Classes: {np.unique(y)}\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples\n")
    
    # Create and train Gaussian Naive Bayes model
    nb = ProductionNaiveBayes(variant='gaussian')
    nb.fit(X_train, y_train)
    print("✓ Model trained successfully\n")
    
    # Evaluate on test set
    print("=== Model Evaluation ===")
    metrics = nb.evaluate(X_test, y_test)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"\nClassification Report:\n{metrics['classification_report']}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}\n")
    
    # Cross-validation
    print("=== Cross-Validation ===")
    cv_results = nb.cross_validate(X_train, y_train, cv=3)
    print(f"Mean CV Accuracy: {cv_results['mean_cv_score']:.4f}")
    print(f"Std CV Accuracy: {cv_results['std_cv_score']:.4f}\n")
    
    # Get class priors
    print("=== Class Prior Probabilities ===")
    priors = nb.get_class_priors()
    for class_name, prior in priors.items():
        print(f"  P({class_name}): {prior:.4f}")
    
    # Make predictions with probabilities
    print("\n=== Predictions ===")
    new_patients = np.array([
        [99.0, 2, 3, 1],   # Likely cold
        [103.0, 8, 9, 7],  # Likely flu
        [98.6, 0, 0, 0]    # Likely healthy
    ])
    predictions = nb.predict(new_patients)
    probabilities = nb.predict_proba(new_patients)
    
    print("New patients to diagnose:")
    for i, (patient, pred, proba) in enumerate(zip(new_patients, predictions, probabilities)):
        print(f"\n  Patient {i+1}: Fever={patient[0]}°F, Cough={patient[1]}, Fatigue={patient[2]}, Breathing={patient[3]}")
        print(f"    Predicted: {pred}")
        print(f"    Probabilities: {dict(zip(nb.label_encoder.classes_, proba))}")
    
    # Save model
    print("\n=== Model Persistence ===")
    nb.save_model('models/medical_diagnosis_nb_model.pkl')
    
    # Load and verify
    new_nb = ProductionNaiveBayes()
    new_nb.load_model('models/medical_diagnosis_nb_model.pkl')
    verification = new_nb.predict(new_patients)
    print(f"Verification (loaded model): {verification[0]}")


if __name__ == "__main__":
    demonstrate_production_naive_bayes()
