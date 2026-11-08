import numpy as np
from typing import Dict, List, Tuple

def add_laplace_differential_privacy(
    weights: np.ndarray,
    epsilon: float = 1.0,
    sensitivity: float = 0.1
) -> np.ndarray:
    """
    Applies Laplacian Differential Privacy to client model update tensors.
    
    Formula:
    b = sensitivity / epsilon
    noise ~ Laplace(0, b)
    noisy_weights = weights + noise
    
    - Smaller epsilon means tighter privacy and higher noise variance.
    - Sensitivity bounds the maximum L1 change a single patient sample can cause on weights.
    """
    if epsilon <= 0:
        raise ValueError("Epsilon must be strictly positive.")
    
    scale = sensitivity / epsilon
    noise = np.random.laplace(loc=0.0, scale=scale, size=weights.shape)
    return weights + noise

def simulate_secure_aggregation_masks(
    agent_weights_dict: Dict[str, Tuple[np.ndarray, float]]
) -> Tuple[Dict[str, Tuple[np.ndarray, float]], List[str]]:
    """
    Simulates Bonawitz et al. / cryptographic pairwise random masking for Secure Aggregation.
    
    For pairs of agents (i, j):
    A symmetric random seed or mask M_{i,j} is agreed upon.
    Agent i adds +M_{i,j} to its weights, and Agent j adds -M_{i,j}.
    When the aggregator computes the sum:
    sum_i (W_i + sum_{j != i} s_{i,j} * M_{i,j}) = sum_i W_i
    All masks cancel out exactly to 0, ensuring individual updates remain secret from the server.
    """
    agent_ids = list(agent_weights_dict.keys())
    k = len(agent_ids)
    logs = []
    
    if k < 2:
        logs.append("[SECAGG] Fewer than 2 clients; masking skipped.")
        return agent_weights_dict, logs
    
    logs.append(f"[SECAGG] Initiating Pairwise Masking across {k} autonomous agents.")
    
    # Copy weights
    masked_weights_dict = {
        agent_id: (w.copy(), b) for agent_id, (w, b) in agent_weights_dict.items()
    }
    
    # Generate pairwise random masks that cancel upon summation
    for i in range(k):
        for j in range(i + 1, k):
            aid_i = agent_ids[i]
            aid_j = agent_ids[j]
            w_shape = agent_weights_dict[aid_i][0].shape
            
            # Generate random mask
            mask = np.random.normal(loc=0.0, scale=0.5, size=w_shape)
            mask_bias = float(np.random.normal(loc=0.0, scale=0.5))
            
            w_i, b_i = masked_weights_dict[aid_i]
            w_j, b_j = masked_weights_dict[aid_j]
            
            # Add to agent i, subtract from agent j
            masked_weights_dict[aid_i] = (w_i + mask, b_i + mask_bias)
            masked_weights_dict[aid_j] = (w_j - mask, b_j - mask_bias)
            
            logs.append(f"[SECAGG] Established encrypted blinding mask between [{aid_i}] <-> [{aid_j}].")
            
    logs.append("[SECAGG] All individual client updates blinded; masks mathematically cancel at aggregator.")
    return masked_weights_dict, logs

