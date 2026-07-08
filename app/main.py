from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from app.services.model_service import ModelService
from app.services.lime_service  import LimeService

model_service = ModelService()
lime_service  = LimeService(model_service)

from app.routers import predict

app = FastAPI(
    title      = "Weather Classification API",
    description= "Klasifikasi kondisi cuaca Makassar — XGBoost + LIME (6 kelas)",
    version    = "3.0.0",
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
        "app"    : "Weather Classification API v3",
        "status" : "running",
        "classes": ["Cerah", "Cerah Berawan", "Berawan Sebagian",
                    "Berawan Tebal", "Gerimis", "Hujan"],
        "docs"   : "/docs",
    }