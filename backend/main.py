import os
from contextlib import asynccontextmanager
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd

from config import DATA_PATH, FEATURE_COLUMNS, HOST, PORT, API_KEY
from models import PatientFeatures, TrainRoundRequest, TrainRoundResponse, PredictionResponse, AgentSummary
from federated.partitioner import PartitionManager
from federated.agent import HospitalAgent
from federated.aggregator import FederatedServer

server_instance: FederatedServer = None
partitioner_instance: PartitionManager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global server_instance, partitioner_instance
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    
    # Initialize partitioner and dataset partitions
    partitioner_instance = PartitionManager(DATA_PATH)
    partitions = partitioner_instance.partition_data()
    
    # Instantiate autonomous hospital agents (never sharing raw data)
    agents = {}
    for agent_key, pdata in partitions.items():
        agents[agent_key] = HospitalAgent(
            agent_id=agent_key,
            name=pdata["name"],
            bias_description=pdata["bias_description"],
            local_df=pdata["df"],
            feature_columns=FEATURE_COLUMNS
        )
    
    # Instantiate Federated Server
    server_instance = FederatedServer(
        agents=agents,
        test_df=partitioner_instance.raw_df,
        feature_columns=FEATURE_COLUMNS
    )
    yield

app = FastAPI(
    title="Privacy-Preserving Federated Smart Healthcare API",
    description="Implementation of Federated Multi-Agent Architecture for Smart Healthcare (GH Raisoni College of Engineering)",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for all origins so GitHub Pages / custom frontends can communicate seamlessly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {
        "status": "healthy",
        "service": "Federated Smart Healthcare API",
        "port": PORT,
        "round": server_instance.current_round if server_instance else 0,
        "total_agents": len(server_instance.agents) if server_instance else 0
    }

@app.get("/api/agents", response_model=List[AgentSummary])
def get_agents():
    if not server_instance:
        raise HTTPException(status_code=500, detail="Federated server not initialized")
    summaries = [AgentSummary(**agent.get_summary()) for agent in server_instance.agents.values()]
    return summaries

@app.post("/api/train-round", response_model=TrainRoundResponse)
def train_rounds(request: TrainRoundRequest):
    if not server_instance:
        raise HTTPException(status_code=500, detail="Federated server not initialized")
    
    start_log_count = len(server_instance.logs)
    last_res = None
    for _ in range(request.rounds):
        last_res = server_instance.run_round(
            use_dp=request.use_dp,
            epsilon=request.epsilon,
            use_secagg=request.use_secagg
        )
        
    global_acc, global_loss = server_instance.evaluate_global_model()
    agent_metrics = {aid: agent.get_summary() for aid, agent in server_instance.agents.items()}
    new_logs = server_instance.logs[start_log_count:]
    
    weights_summary = {
        col: round(float(w), 4) for col, w in zip(FEATURE_COLUMNS, server_instance.global_weights)
    }
    weights_summary["intercept"] = round(float(server_instance.global_bias), 4)
    
    return TrainRoundResponse(
        current_round=server_instance.current_round,
        rounds_executed=request.rounds,
        global_accuracy=round(global_acc, 4),
        global_loss=round(global_loss, 4),
        history=server_instance.history,
        agent_metrics=agent_metrics,
        logs=new_logs,
        global_weights=weights_summary
    )

@app.post("/api/reset")
def reset_server():
    if not server_instance:
        raise HTTPException(status_code=500, detail="Federated server not initialized")
    server_instance.reset_state()
    return {"message": "Federated server and all agents reset to Round 0 successfully"}

@app.post("/api/predict", response_model=PredictionResponse)
def predict_patient(patient: PatientFeatures):
    if not server_instance:
        raise HTTPException(status_code=500, detail="Federated server not initialized")
    
    features_dict = patient.model_dump()
    result = server_instance.predict_sample(features_dict)
    return PredictionResponse(**result)

if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)

