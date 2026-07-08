from fastapi import APIRouter, HTTPException
from datetime import datetime
from ..schemas.prediction import PredictRequest, PredictResponse
from ..services.preprocessor import (
    build_features, FEATURES, CLASS_LABEL_ID,
)

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        from ..main import model_service, lime_service

        # 1. Data cuaca saat ini (dari Open-Meteo via Flutter)
        current = {
            "temperature_2m"      : req.temperature_2m,
            "relative_humidity_2m": req.relative_humidity_2m,
            "dew_point_2m"         : req.dew_point_2m,
            "surface_pressure"     : req.surface_pressure,
            "cloud_cover"           : req.cloud_cover,
            "wind_speed_10m"         : req.wind_speed_10m,
            "wind_gusts_10m"          : req.wind_gusts_10m,
        }

        # 2. Siapkan history (default pakai current jika kosong)
        temp_hist  = req.temp_history     or [req.temperature_2m] * 3
        hum_hist   = req.humidity_history or [req.relative_humidity_2m] * 3
        rain_hist  = req.rain_history      or [0.0] * 3
        cloud_hist = req.cloud_history     or [req.cloud_cover] * 3
        wind_hist  = req.wind_history       or [req.wind_speed_10m] * 3

        now = datetime.now()

        # 3. Build fitur (40 fitur) untuk prediksi SAAT INI
        features = build_features(
            current, temp_hist, hum_hist,
            rain_hist, cloud_hist, wind_hist, now,
        )

        # 4. Prediksi kondisi cuaca sekarang
        result_now = model_service.predict(features)

        # 5. Prediksi per jam (24 jam ke depan)
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

        # 6. Prediksi per hari (7 hari ke depan)
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

        # 7. LIME explanation untuk prediksi saat ini
        feat_values = {
            FEATURES[i]: float(features[0][i])
            for i in range(len(FEATURES))
        }
        lime_result = lime_service.explain(
            features, feat_values, result_now["condition"])

        # 8. Konversi label English ke Indonesia untuk ditampilkan
        condition_id = CLASS_LABEL_ID.get(
            result_now["condition"], result_now["condition"])

        class_proba_id = {
            CLASS_LABEL_ID.get(k, k): v
            for k, v in result_now["class_proba"].items()
        }

        return PredictResponse(
            current_condition= condition_id,
            confidence       = result_now["confidence"],
            class_proba      = class_proba_id,
            hourly_forecast  = hourly_out,
            daily_forecast   = daily_out,
            lime_features    = lime_result,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))