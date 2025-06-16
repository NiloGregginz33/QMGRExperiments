from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import json

# Import your existing Factory functions
from CGPTFactory import amplify_target_state, forecast_branch_stability

# ---- FastAPI App ----
app = FastAPI()

# ---- Request Models ----
class AmplifierRequest(BaseModel):
    target_bits: List[str]
    charge_history: List[float]
    shots: int
    scaling_factor: float

# ---- Amplifier Endpoint ----
@app.post("/run_amplifier")
async def run_amplifier(request: AmplifierRequest):
    target_bits = tuple(request.target_bits)
    charge_history = np.array(request.charge_history)
    shots = request.shots
    scaling_factor = request.scaling_factor

    print(f"Running amplifier on target: {target_bits}")

    # Run Factory amplifier
    result_charge, result_counts = amplify_target_state(
        target_bits, charge_history, shots=shots, scaling_factor=scaling_factor
    )

    return {
        "charge_history": result_charge.tolist(),
        "counts": result_counts
    }

# ---- Forecast Endpoint ----
@app.get("/forecast")
def forecast():
    print("Running branch forecast...")
    stability_score, forecast_msg = forecast_branch_stability()
    return {
        "stability_score": stability_score,
        "forecast": forecast_msg
    }

# ---- Status Endpoint ----
@app.get("/status")
def status():
    return {"status": "Vision backend is running!"}
