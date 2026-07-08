from fastapi import APIRouter, HTTPException
from datetime import datetime
from pydantic import BaseModel
from typing import List

router = APIRouter()

class PredictRequest(BaseModel):
    lat                  : float
    lon                  : float
    temperature_2m       : float = 27.0
    relative_humidity_2m : float = 80.0
    dew_point_2m         : float = 22.0
    surface_pressure     : float = 1010.0
    cloud_cover          : float = 50.0
    wind_speed_10m       : float = 10.0
    wind_gusts_10m       : float = 15.0
    temp_history         : List[float] = []
    humidity_history     : List[float] = []
    rain_history         : List[float] = []
    cloud_history        : List[float] = []
    wind_history         : List[float] = []

@router.post("/predict")
async def predict(req: PredictRequest):
    try:
        from ..main import model_service, lime_service
        from ..services.preprocessor import (
            build_features, FEATURES, CLASS_LABEL_ID,
        )

        current = {
            "temperature_2m"      : req.temperature_2m,
            "relative_humidity_2m": req.relative_humidity_2m,
            "dew_point_2m"        : req.dew_point_2m,
            "surface_pressure"    : req.surface_pressure,
            "cloud_cover"         : req.cloud_cover,
            "wind_speed_10m"      : req.wind_speed_10m,
            "wind_gusts_10m"      : req.wind_gusts_10m,
        }

        # Default histori jika tidak dikirim Flutter
        temp_hist  = req.temp_history     or [req.temperature_2m] * 6
        hum_hist   = req.humidity_history or [req.relative_humidity_2m] * 6
        rain_hist  = req.rain_history     or [0.0] * 6
        cloud_hist = req.cloud_history    or [req.cloud_cover] * 6
        wind_hist  = req.wind_history     or [req.wind_speed_10m] * 6

        now = datetime.now()

        # 1. Prediksi kondisi SAAT INI
        features   = build_features(
            current, temp_hist, hum_hist,
            rain_hist, cloud_hist, wind_hist, now,
        )
        result_now = model_service.predict(features)

        # 2. Prediksi per jam 24 jam ke depan (autoregressive)
        hourly_raw = model_service.predict_multi_hour(
            current, temp_hist, hum_hist,
            rain_hist, cloud_hist, wind_hist, hours=24,
        )
        hourly_out = [
            {
                "hour"      : h["hour"],
                "condition" : CLASS_LABEL_ID.get(
                                h["condition"], h["condition"]),
                "confidence": h["confidence"],
            }
            for h in hourly_raw
        ]

        # 3. Prediksi harian 7 hari ke depan
        daily_raw = model_service.predict_multi_day(
            current, temp_hist, hum_hist,
            rain_hist, cloud_hist, wind_hist, days=7,
        )
        daily_out = [
            {
                "day"       : d["day"],
                "date"      : d["date"],
                "condition" : CLASS_LABEL_ID.get(
                                d["condition"], d["condition"]),
                "confidence": d["confidence"],
            }
            for d in daily_raw
        ]

        # 4. LIME — hanya fitur meteorologis
        feat_values = {
            FEATURES[i]: float(features[0][i])
            for i in range(len(FEATURES))
        }
        lime_result = lime_service.explain(
            features, feat_values, result_now["condition"])

        # 5. Konversi label ke Bahasa Indonesia
        condition_id = CLASS_LABEL_ID.get(
            result_now["condition"], result_now["condition"])
        class_proba_id = {
            CLASS_LABEL_ID.get(k, k): v
            for k, v in result_now["class_proba"].items()
        }

        return {
            "current_condition": condition_id,
            "confidence"       : result_now["confidence"],
            "class_proba"      : class_proba_id,
            "hourly_forecast"  : hourly_out,
            "daily_forecast"   : daily_out,
            "lime_features"    : lime_result,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))