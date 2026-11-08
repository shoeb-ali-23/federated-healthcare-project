import pandas as pd
import numpy as np
from typing import Dict, Tuple
from pathlib import Path
from sklearn.preprocessing import StandardScaler

class PartitionManager:
    """
    Module 1 & 2:
    - Loads the UCI Heart Disease tabular dataset.
    - Partitions data into 3 Non-IID distributions matching clinical reality:
        * Hospital A: Geriatric Cardiology Specialty (Bias towards age >= 58, higher male ratio)
        * Hospital B: Chest Pain & Angina Center (Bias towards cp == 3 asymptomatic or typical angina)
        * Diagnostic Center C: General Outpatient / Balanced edge distribution
    - Performs local standardization and feature scaling strictly retained per agent.
    """
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.raw_df = pd.read_csv(data_path)
        self.feature_columns = [
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ]
        self.target_column = "condition"
        
        # Global reference scaler fitted on reference demographic distributions for consistent evaluation
        self.reference_scaler = StandardScaler()
        self.reference_scaler.fit(self.raw_df[self.feature_columns].values)

    def partition_data(self) -> Dict[str, Dict]:
        df = self.raw_df.copy()
        
        # Hospital A filter: Geriatric Specialty (Age >= 56 or male older adults)
        cond_a = (df["age"] >= 56) | ((df["age"] >= 50) & (df["sex"] == 1))
        df_a = df[cond_a].copy()
        
        # Hospital B filter: Chest Pain & Ischemic Center (cp >= 2 or exang == 1)
        remaining_b = df[~cond_a].copy()
        cond_b = (df["cp"] >= 2) | (df["exang"] == 1)
        df_b = df[cond_b].copy()
        
        # Diagnostic Center C: Balanced outpatient clinic & younger / atypical cohort
        # To guarantee healthy local datasets, we take stratified allocations
        # Shuffle with fixed seed for determinism
        df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        # Segment into non-IID partitions:
        # Partition A: Older / Male demographic (100 samples)
        part_a_candidates = df_shuffled[(df_shuffled["age"] >= 55) | (df_shuffled["sex"] == 1)]
        df_hospital_a = part_a_candidates.iloc[:100].copy()
        
        # Partition B: Severe chest pain / high cholesterol / angina (100 samples)
        remaining = df_shuffled.drop(index=df_hospital_a.index)
        part_b_candidates = remaining[(remaining["cp"] >= 2) | (remaining["chol"] >= 240)]
        df_hospital_b = part_b_candidates.iloc[:100].copy()
        
        # Partition C: Remaining edge and general clinic cohort (~97 samples)
        df_diagnostic_c = remaining.drop(index=df_hospital_b.index).copy()
        
        partitions = {
            "hospital_a": {
                "name": "Hospital A (Geriatric & Cardiology)",
                "df": df_hospital_a,
                "bias_description": "Biased towards older age (mean > 58) & male cohort"
            },
            "hospital_b": {
                "name": "Hospital B (Chest Pain & Ischemia Center)",
                "df": df_hospital_b,
                "bias_description": "Biased towards symptomatic angina & high cholesterol"
            },
            "diagnostic_c": {
                "name": "Diagnostic Center C (Outpatient Screening)",
                "df": df_diagnostic_c,
                "bias_description": "Balanced baseline distribution and atypical clinical presentations"
            }
        }
        
        return partitions

    def preprocess_local_partition(self, df_part: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """
        Module 2: Local Data Preprocessing.
        Each agent fits its own local scaler to ensure strict data autonomy and avoid data leakage.
        """
        X_raw = df_part[self.feature_columns].values.astype(np.float64)
        y = df_part[self.target_column].values.astype(np.float64)
        
        local_scaler = StandardScaler()
        X_scaled = local_scaler.fit_transform(X_raw)
        
        return X_scaled, y, local_scaler

