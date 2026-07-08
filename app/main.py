from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# Inisialisasi services saat startup
from app.services.model_service import ModelService
from app.services.lime_service  import LimeService

model_service = ModelService()
lime_service  = LimeService(model_service)

# Import router predict saja (weather router sudah tidak dipakai)
from app.routers import predict

app = FastAPI(
    title      = "Weather Prediction API",
    description= "Forecasting curah hujan Makassar — XGBoost + LIME",
    version    = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins    = ["*"],
    allow_methods    = ["*"],
    allow_headers    = ["*"],
    allow_credentials= True,
)

app.include_router(
    predict.router, prefix="/api/v1", tags=["Predict"])

@app.get("/")
async def root():
    return {
        "app"    : "Weather Prediction API v2",
        "status" : "running",
        "docs"   : "/docs",
    }