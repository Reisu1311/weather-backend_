from fastapi import APIRouter, HTTPException
from ..services.owm_service import fetch_current, fetch_forecast

router = APIRouter()

@router.get("/weather")
async def get_weather(lat: float, lon: float):
    try:
        current  = await fetch_current(lat, lon)
        forecast = await fetch_forecast(lat, lon)
        return {"current": current, "forecast": forecast}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))