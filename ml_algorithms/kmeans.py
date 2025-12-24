"""
K-Means Clustering - Production Ready Implementation
Using scikit-learn for unsupervised learning tasks
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import joblib
import os


class ProductionKMeans:
    """
    Production-ready K-Means Clustering wrapper
    Includes preprocessing, optimal k selection, and model persistence
    """
    
    def __init__(self, n_clusters=3, init='k-means++', max_iter=300, random_state=42):
        """
        Initialize K-Means clustering
        n_clusters: number of clusters
        init: initialization method ('k-means++' or 'random')
        max_iter: maximum number of iterations
        random_state: random seed for reproducibility
        """
        self.model = SklearnKMeans(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            random_state=random_state,
            n_init=10  # Run algorithm 10 times with different seeds
        )
        self.scaler = StandardScaler()  # Feature scaling
        self.is_fitted = False
        self.labels_ = None
        self.cluster_centers_ = None
    
    def preprocess_data(self, X, fit_scaler=True):
        """
        Preprocess features: handle missing values and scale
        """
        # Convert to DataFrame if numpy array
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features (important for distance-based clustering)
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def fit(self, X):
        """
        Fit the K-Means model with preprocessing
        """
        # Preprocess data
        X_processed = self.preprocess_data(X, fit_scaler=True)
        
        # Fit model
        self.model.fit(X_processed)
        self.is_fitted = True
        self.labels_ = self.model.labels_
        self.cluster_centers_ = self.model.cluster_centers_
        
        return self
    
    def predict(self, X):
        """
        Predict cluster labels for new data
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Preprocess using fitted scaler
        X_processed = self.preprocess_data(X, fit_scaler=False)
        
        return self.model.predict(X_processed)
    
    def fit_predict(self, X):
        """
        Fit model and return cluster labels
        """
        return self.fit(X).labels_
    
    def evaluate(self, X):
        """
        Comprehensive clustering evaluation
        Returns multiple quality metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        X_processed = self.preprocess_data(X, fit_scaler=False)
        
        metrics = {
            'inertia': self.model.inertia_,  # Within-cluster sum of squares
            'silhouette_score': silhouette_score(X_processed, self.labels_),  # -1 to 1, higher is better
            'davies_bouldin_score': davies_bouldin_score(X_processed, self.labels_),  # Lower is better
            'calinski_harabasz_score': calinski_harabasz_score(X_processed, self.labels_)  # Higher is better
        }
        
        return metrics
    
    def find_optimal_k(self, X, k_range=range(2, 11)):
        """
        Use elbow method to find optimal number of clusters
        """
        X_processed = self.preprocess_data(X, fit_scaler=True)
        
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = SklearnKMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_processed)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_processed, kmeans.labels_))
        
        return {
            'k_values': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores
        }
    
    def get_cluster_summary(self, X):
        """
        Get summary statistics for each cluster
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Convert to DataFrame for easier analysis
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        X['cluster'] = self.labels_
        
        summary = X.groupby('cluster').agg(['mean', 'std', 'count'])
        X.drop('cluster', axis=1, inplace=True)
        
        return summary
    
    def save_model(self, filepath='models/kmeans_model.pkl'):
        """
        Save trained model to disk
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'labels': self.labels_,
            'cluster_centers': self.cluster_centers_
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/kmeans_model.pkl'):
        """
        Load trained model from disk
        """
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_fitted = data['is_fitted']
        self.labels_ = data['labels']
        self.cluster_centers_ = data['cluster_centers']
        print(f"Model loaded from {filepath}")


# Example usage with production best practices
def demonstrate_production_kmeans():
    """Demonstrate production-ready K-Means clustering"""
    print("=== Production K-Means Clustering Demo ===\n")
    
    # Sample data: Customer segmentation
    # Features: [annual_income(k$), spending_score(1-100), age]
    X = np.array([
        [15, 39, 19], [15, 81, 20], [16, 6, 21], [16, 77, 22],
        [17, 40, 23], [17, 76, 24], [18, 6, 25], [18, 94, 26],
        [19, 3, 27], [19, 72, 28], [19, 14, 29], [19, 99, 30],
        [20, 15, 31], [20, 77, 32], [20, 13, 33], [20, 79, 34],
        [23, 35, 35], [23, 66, 36], [24, 29, 37], [24, 98, 38],
        [25, 35, 39], [25, 73, 40], [28, 14, 41], [28, 82, 42],
        [28, 32, 43], [28, 61, 44], [29, 31, 45], [29, 87, 46],
        [30, 4, 47], [30, 73, 48], [33, 4, 49], [33, 92, 50],
        [33, 14, 51], [33, 81, 52], [34, 17, 53], [34, 73, 54],
        [37, 26, 55], [37, 75, 56], [38, 35, 57], [38, 92, 58]
    ])
    
    print("Dataset Info:")
    print(f"Samples: {len(X)}")
    print(f"Features: Annual Income (k$), Spending Score (1-100), Age\n")
    
    # Find optimal number of clusters
    print("=== Finding Optimal K ===")
    kmeans_temp = ProductionKMeans()
    elbow_results = kmeans_temp.find_optimal_k(X, k_range=range(2, 8))
    
    print("K-values and their Silhouette Scores:")
    for k, score in zip(elbow_results['k_values'], elbow_results['silhouette_scores']):
        print(f"  k={k}: {score:.4f}")
    
    # Select best k (highest silhouette score)
    best_k = elbow_results['k_values'][np.argmax(elbow_results['silhouette_scores'])]
    print(f"\nOptimal k: {best_k}\n")
    
    # Create and train model with optimal k
    kmeans = ProductionKMeans(n_clusters=best_k, random_state=42)
    labels = kmeans.fit_predict(X)
    print(f"✓ Model trained with {best_k} clusters\n")
    
    # Evaluate clustering quality
    print("=== Clustering Evaluation ===")
    metrics = kmeans.evaluate(X)
    print(f"Inertia: {metrics['inertia']:.2f}")
    print(f"Silhouette Score: {metrics['silhouette_score']:.4f} (closer to 1 is better)")
    print(f"Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f} (lower is better)")
    print(f"Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.2f} (higher is better)\n")
    
    # Cluster centers in original scale
    print("=== Cluster Centers (original scale) ===")
    centers_scaled = kmeans.cluster_centers_
    centers_original = kmeans.scaler.inverse_transform(centers_scaled)
    for i, center in enumerate(centers_original):
        print(f"  Cluster {i}: Income=${center[0]:.1f}k, Score={center[1]:.1f}, Age={center[2]:.1f}")
    
    # Cluster distribution
    print("\n=== Cluster Distribution ===")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"  Cluster {cluster}: {count} customers ({count/len(X)*100:.1f}%)")
    
    # Make predictions for new customers
    print("\n=== Predictions ===")
    new_customers = np.array([
        [20, 80, 25],  # Young, high spender
        [35, 20, 50],  # Middle-aged, low spender
        [25, 60, 30]   # Young, medium spender
    ])
    predictions = kmeans.predict(new_customers)
    
    print("New customers to segment:")
    for i, (customer, cluster) in enumerate(zip(new_customers, predictions)):
        print(f"  Customer {i+1}: Income=${customer[0]}k, Score={customer[1]}, Age={customer[2]} → Cluster {cluster}")
    
    # Get cluster summary
    print("\n=== Cluster Summary Statistics ===")
    summary = kmeans.get_cluster_summary(X)
    print(summary)
    
    # Save model
    print("\n=== Model Persistence ===")
    kmeans.save_model('models/customer_segmentation_model.pkl')
    
    # Load and verify
    new_kmeans = ProductionKMeans()
    new_kmeans.load_model('models/customer_segmentation_model.pkl')
    verification = new_kmeans.predict(new_customers)
    print(f"Verification (loaded model): Cluster {verification[0]}")


if __name__ == "__main__":
    demonstrate_production_kmeans()
