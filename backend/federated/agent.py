import numpy as np
import pandas as pd
import warnings
from typing import Tuple, Dict, Any, List
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.exceptions import ConvergenceWarning

class HospitalAgent:
    """
    Module 3 & 4: Multi-Agent Healthcare Layer & Local Model Training.
    - Encapsulates local patient dataset (never exposes individual patient records).
    - Performs local model training via SGDClassifier with log_loss (Logistic Regression).
    - Exports only model weights (coefficients & intercept) for federated aggregation.
    """
    def __init__(self, agent_id: str, name: str, bias_description: str, local_df: pd.DataFrame, feature_columns: List[str]):
        self.agent_id = agent_id
        self.name = name
        self.bias_description = bias_description
        self.feature_columns = feature_columns
        self.raw_df = local_df.copy()
        
        # Local metadata
        self.sample_count = len(self.raw_df)
        self.mean_age = float(self.raw_df["age"].mean()) if "age" in self.raw_df else 0.0
        self.male_ratio = float((self.raw_df["sex"] == 1).mean()) if "sex" in self.raw_df else 0.0
        dist = self.raw_df["condition"].value_counts().to_dict()
        self.target_distribution = {"healthy": int(dist.get(0, 0)), "disease": int(dist.get(1, 0))}
        
        # Module 2: Local Preprocessing strictly retained on-agent
        self.local_mean = self.raw_df[self.feature_columns].mean().values.astype(np.float64)
        self.local_std = self.raw_df[self.feature_columns].std().replace(0, 1.0).values.astype(np.float64)
        
        # Local feature tensors
        self.X_local = (self.raw_df[self.feature_columns].values - self.local_mean) / self.local_std
        self.y_local = self.raw_df["condition"].values.astype(np.int64)
        
        # Local classifier initialization (warm_start allows continuing training across rounds)
        self.model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.01,
            learning_rate="constant",
            eta0=0.05,
            max_iter=1,
            warm_start=True,
            random_state=42
        )
        # Prime model classes
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            self.model.fit(self.X_local[:2], np.array([0, 1]))
        
        # Model parameter buffers
        self.weights = self.model.coef_[0].copy()
        self.bias = float(self.model.intercept_[0])
        
        self.last_accuracy = 0.0
        self.last_loss = 0.0
        self.status = "Idle"

    def set_global_weights(self, global_weights: np.ndarray, global_bias: float):
        """
        Receives global aggregated parameters and updates local model state.
        """
        self.weights = global_weights.copy()
        self.bias = float(global_bias)
        self.model.coef_ = np.array([self.weights.copy()])
        self.model.intercept_ = np.array([self.bias])

    def train_local_epoch(self, epochs: int = 3) -> Dict[str, Any]:
        """
        Executes local SGD optimization epochs strictly over private patient data.
        Returns training metrics (loss, accuracy) without exposing patient data.
        """
        self.status = "Training"
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            for _ in range(epochs):
                # Shuffle indices locally
                indices = np.random.permutation(self.sample_count)
                X_shuffled = self.X_local[indices]
                y_shuffled = self.y_local[indices]
                self.model.partial_fit(X_shuffled, y_shuffled, classes=np.array([0, 1]))
            
        self.weights = self.model.coef_[0].copy()
        self.bias = float(self.model.intercept_[0])
        
        # Evaluate local metrics
        preds = self.model.predict(self.X_local)
        proba = self.model.predict_proba(self.X_local)
        
        self.last_accuracy = float(accuracy_score(self.y_local, preds))
        self.last_loss = float(log_loss(self.y_local, proba, labels=[0, 1]))
        self.status = "Local Training Complete"
        
        return {
            "agent_id": self.agent_id,
            "sample_count": self.sample_count,
            "accuracy": round(self.last_accuracy, 4),
            "loss": round(self.last_loss, 4)
        }

    def get_model_parameters(self) -> Tuple[np.ndarray, float]:
        """
        Returns model parameters (W_k, b_k).
        Ensures zero-knowledge of raw data points.
        """
        return self.weights.copy(), float(self.bias)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "sample_count": self.sample_count,
            "bias_description": self.bias_description,
            "target_distribution": self.target_distribution,
            "mean_age": round(self.mean_age, 1),
            "male_ratio": round(self.male_ratio, 3),
            "status": self.status,
            "last_accuracy": round(self.last_accuracy, 4),
            "last_loss": round(self.last_loss, 4)
        }
