import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

from federated.agent import HospitalAgent
from federated.privacy import add_laplace_differential_privacy, simulate_secure_aggregation_masks

class FederatedServer:
    """
    Module 5: Federated Aggregation Server implementing FedAvg:
    W_global = sum_{k=1}^K (n_k / N) * W_k
    b_global = sum_{k=1}^K (n_k / N) * b_k
    
    Coordinates client rounds, applies differential privacy and secure aggregation,
    and maintains global model state and training history.
    """
    def __init__(self, agents: Dict[str, HospitalAgent], test_df: pd.DataFrame, feature_columns: List[str]):
        self.agents = agents
        self.feature_columns = feature_columns
        self.test_df = test_df.copy()
        
        # Reference normalization fitted on pooled data for standard global validation
        self.global_scaler = StandardScaler()
        self.X_val = self.global_scaler.fit_transform(self.test_df[self.feature_columns].values)
        self.y_val = self.test_df["condition"].values.astype(np.int64)
        
        self.n_features = len(feature_columns)
        self.current_round = 0
        self.history: List[Dict[str, Any]] = []
        self.logs: List[str] = []
        
        # Initialize global model parameters
        self.global_weights = np.zeros(self.n_features, dtype=np.float64)
        self.global_bias = 0.0
        
        # Broadcast initial weights
        self._broadcast_weights()
        self.log("[FED_SERVER] Initialized Federated Aggregation Server with 3 Autonomous Hospital Agents.")

    def log(self, message: str):
        self.logs.append(message)

    def _broadcast_weights(self):
        for agent in self.agents.values():
            agent.set_global_weights(self.global_weights, self.global_bias)

    def evaluate_global_model(self) -> Tuple[float, float]:
        """
        Evaluates the aggregated global weights on validation dataset using standard sigmoid:
        z = X * W + b
        p = 1 / (1 + exp(-z))
        """
        z = np.dot(self.X_val, self.global_weights) + self.global_bias
        probs = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        preds = (probs >= 0.5).astype(int)
        
        acc = float(accuracy_score(self.y_val, preds))
        # Binary cross-entropy loss with epsilon clipping
        eps = 1e-15
        probs_clipped = np.clip(probs, eps, 1 - eps)
        loss = float(-np.mean(self.y_val * np.log(probs_clipped) + (1 - self.y_val) * np.log(1 - probs_clipped)))
        
        return acc, loss

    def run_round(self, use_dp: bool = False, epsilon: float = 1.0, use_secagg: bool = False) -> Dict[str, Any]:
        """
        Executes one complete Federated Learning communication round:
        1. Local Client SGD Updates.
        2. Module 6: Privacy layer (Differential Privacy noise injection).
        3. Module 6: Secure Aggregation simulation (pairwise blinding masks).
        4. Module 5: FedAvg aggregation.
        5. Global validation & weight broadcast.
        """
        self.current_round += 1
        self.log(f"--- Round {self.current_round} Initiated ---")
        
        agent_metrics = {}
        client_updates: Dict[str, Tuple[np.ndarray, float]] = {}
        client_sample_counts: Dict[str, int] = {}
        
        # 1. Local Agent Training
        for agent_id, agent in self.agents.items():
            metrics = agent.train_local_epoch(epochs=3)
            w, b = agent.get_model_parameters()
            agent_metrics[agent_id] = metrics
            client_updates[agent_id] = (w, b)
            client_sample_counts[agent_id] = agent.sample_count
            self.log(f"[LOCAL_SGD] {agent.name}: {agent.sample_count} samples trained. Acc: {metrics['accuracy']:.4f}, Loss: {metrics['loss']:.4f}")

        # 2. Differential Privacy Noise Injection
        if use_dp:
            self.log(f"[PRIVACY_DP] Applying Laplace Differential Privacy (epsilon = {epsilon:.2f}, sensitivity = 0.10)...")
            noisy_updates = {}
            for agent_id, (w, b) in client_updates.items():
                noisy_w = add_laplace_differential_privacy(w, epsilon=epsilon, sensitivity=0.1)
                noisy_b = float(add_laplace_differential_privacy(np.array([b]), epsilon=epsilon, sensitivity=0.1)[0])
                noisy_updates[agent_id] = (noisy_w, noisy_b)
            client_updates = noisy_updates
            self.log("[PRIVACY_DP] Laplacian perturbations injected into local gradient vectors to prevent membership inference.")

        # 3. Secure Aggregation Blinding
        if use_secagg:
            client_updates, secagg_logs = simulate_secure_aggregation_masks(client_updates)
            for l in secagg_logs:
                self.log(l)

        # 4. Federated Averaging (FedAvg): W_global = sum_{k=1}^K (n_k / N) * W_k
        total_samples = sum(client_sample_counts.values())
        new_global_weights = np.zeros(self.n_features, dtype=np.float64)
        new_global_bias = 0.0

        for agent_id, (w, b) in client_updates.items():
            weight_factor = client_sample_counts[agent_id] / total_samples
            new_global_weights += weight_factor * w
            new_global_bias += weight_factor * b

        self.global_weights = new_global_weights
        self.global_bias = new_global_bias
        self.log(f"[FED_AGGREGATOR] Executed FedAvg across {len(client_updates)} nodes (Total N = {total_samples}).")

        # 5. Broadcast global model to agents
        self._broadcast_weights()
        for agent in self.agents.values():
            agent.status = "Weights Aggregated"

        # 6. Global Evaluation
        global_acc, global_loss = self.evaluate_global_model()
        self.log(f"[GLOBAL_EVAL] Round {self.current_round} Complete -> Global Accuracy: {global_acc:.4f}, Global Loss: {global_loss:.4f}")

        round_summary = {
            "round": self.current_round,
            "accuracy": round(global_acc, 4),
            "loss": round(global_loss, 4),
            "use_dp": use_dp,
            "epsilon": epsilon,
            "use_secagg": use_secagg
        }
        self.history.append(round_summary)

        return {
            "round": self.current_round,
            "global_accuracy": round(global_acc, 4),
            "global_loss": round(global_loss, 4),
            "agent_metrics": agent_metrics,
            "round_summary": round_summary
        }

    def predict_sample(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Runs clinical risk inference on an unseen patient profile using current global model.
        """
        feat_array = np.array([[features[col] for col in self.feature_columns]], dtype=np.float64)
        scaled_feat = self.global_scaler.transform(feat_array)
        
        z = np.dot(scaled_feat, self.global_weights)[0] + self.global_bias
        prob = float(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))))
        has_disease = prob >= 0.5
        
        # Feature impact contributions
        contributions = {}
        for idx, col in enumerate(self.feature_columns):
            impact = float(scaled_feat[0, idx] * self.global_weights[idx])
            contributions[col] = round(impact, 4)
            
        risk_level = "High Risk" if prob >= 0.70 else ("Moderate Risk" if prob >= 0.40 else "Low Risk")
        
        return {
            "has_disease": has_disease,
            "probability": round(prob, 4),
            "risk_level": risk_level,
            "model_round": self.current_round,
            "feature_importance": contributions
        }

    def reset_state(self):
        """
        Resets federated server and hospital agents back to Round 0.
        """
        self.current_round = 0
        self.history = []
        self.logs = []
        self.global_weights = np.zeros(self.n_features, dtype=np.float64)
        self.global_bias = 0.0
        self._broadcast_weights()
        for agent in self.agents.values():
            agent.status = "Idle"
            agent.last_accuracy = 0.0
            agent.last_loss = 0.0
        self.log("[FED_SERVER] Reset server and hospital agent states to baseline (Round 0).")

