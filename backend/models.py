from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class PatientFeatures(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Age in years (e.g., 55)")
    sex: int = Field(..., ge=0, le=1, description="Sex (1 = male, 0 = female)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=100, le=600, description="Serum cholesterol in mg/dl")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., ge=50, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (1 = yes, 0 = no)")
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise relative to rest")
    slope: int = Field(..., ge=0, le=2, description="Slope of the peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=3, description="Number of major vessels (0-3) colored by flourosopy")
    thal: int = Field(..., ge=0, le=2, description="Thalium stress test result (0-2)")

class TrainRoundRequest(BaseModel):
    rounds: int = Field(1, ge=1, le=50, description="Number of federated communication rounds to run")
    use_dp: bool = Field(False, description="Enable Laplacian Differential Privacy")
    epsilon: float = Field(1.0, ge=0.01, le=50.0, description="Privacy budget epsilon (lower = more noise/privacy)")
    use_secagg: bool = Field(False, description="Simulate Secure Aggregation (pairwise zero-sum masking)")

class AgentSummary(BaseModel):
    agent_id: str
    name: str
    sample_count: int
    bias_description: str
    target_distribution: Dict[str, int]
    mean_age: float
    male_ratio: float
    status: str

class TrainRoundResponse(BaseModel):
    current_round: int
    rounds_executed: int
    global_accuracy: float
    global_loss: float
    history: List[Dict[str, Any]]
    agent_metrics: Dict[str, Dict[str, Any]]
    logs: List[str]
    global_weights: Dict[str, Any]

class PredictionResponse(BaseModel):
    has_disease: bool
    probability: float
    risk_level: str
    model_round: int
    feature_importance: Dict[str, float]

