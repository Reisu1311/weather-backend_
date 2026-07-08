from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from ..schemas.prediction import PredictResponse
from ..services.preprocessor import (
    build_features, rain_to_condition,
    rain_to_probability, FEATURES,
)

router = APIRouter()

class PredictRequest(BaseModel):
    lat             : float
    lon             : float
    # Data cuaca realtime dari Open-Meteo (dikirim Flutter)
    temp            : float = 27.0
    humidity        : float = 80.0
    dew_point       : float = 0.0
    pressure        : float = 1010.0
    clouds          : float = 50.0
    wind_speed      : float = 5.0
    wind_gusts      : float = 8.0
    # Histori jam sebelumnya
    rain_history    : List[float] = []
    humidity_history: List[float] = []
    clouds_history  : List[float] = []

@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        from ..main import model_service, lime_service

        # Data cuaca dari Flutter (sudah dari Open-Meteo)
        owm_current = {
            'temp'      : req.temp,
            'humidity'  : req.humidity,
            'dew_point' : req.dew_point,
            'pressure'  : req.pressure,
            'clouds'    : req.clouds,
            'wind_speed': req.wind_speed,
            'wind_gusts': req.wind_gusts,
        }

        # Siapkan history
        rain_hist = req.rain_history or [0.0] * 24
        hum_hist  = req.humidity_history or [req.humidity] * 6
        cld_hist  = req.clouds_history or [req.clouds] * 6

        # Build fitur
        features = build_features(
            owm_current, rain_hist, hum_hist, cld_hist)

        # Prediksi rain sekarang
        result_now = model_service.predict_rain(features)
        rain_now   = result_now["rain_mm"]

        # Prediksi per jam 24 jam ke depan
        hourly_raw = model_service.predict_multi_hour(
            features, owm_current,
            rain_hist, hum_hist, cld_hist, hours=24,
        )
        hourly_out = [
            {
                "hour"     : h["hour"],
                "rain_mm"  : h["rain_mm"],
                "condition": rain_to_condition(
                               h["rain_mm"], int(req.clouds)),
                "rain_prob": rain_to_probability(h["rain_mm"]),
            }
            for h in hourly_raw
        ]

        # Prediksi per hari 7 hari ke depan
        daily_raw = model_service.predict_multi_day(
            owm_current, rain_hist, hum_hist, cld_hist, days=7,
        )
        daily_out = [
            {
                "day"          : d["day"],
                "date"         : d["date"],
                "total_rain_mm": d["total_rain_mm"],
                "max_rain_mm"  : d["max_rain_mm"],
                "condition"    : rain_to_condition(
                                   d["max_rain_mm"],
                                   int(req.clouds)),
                "rain_prob"    : rain_to_probability(
                                   d["max_rain_mm"]),
            }
            for d in daily_raw
        ]

        # LIME explanation
        feat_values = {
            FEATURES[i]: float(features[0][i])
            for i in range(len(FEATURES))
        }
        lime_result = lime_service.explain(features, feat_values)

        return PredictResponse(
            current_rain_mm  = rain_now,
            current_condition= rain_to_condition(
                                 rain_now, int(req.clouds)),
            rain_probability = rain_to_probability(rain_now),
            hourly_forecast  = hourly_out,
            daily_forecast   = daily_out,
            lime_features    = lime_result,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))