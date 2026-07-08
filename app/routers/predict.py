from fastapi import APIRouter, HTTPException
from ..schemas.prediction import PredictRequest, PredictResponse
from ..services.owm_service import fetch_current
from ..services.preprocessor import (
    build_features, rain_to_condition,
    rain_to_probability, FEATURES,
)

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        from ..main import model_service, lime_service

        # 1. Ambil data OWM realtime
        owm = await fetch_current(req.lat, req.lon)

        # 2. Siapkan history
        rain_hist = req.rain_history or \
                    [owm.get("rain_1h", 0.0)] * 24
        hum_hist  = req.humidity_history or \
                    [float(owm["humidity"])] * 6
        cld_hist  = req.clouds_history or \
                    [float(owm["clouds"])] * 6

        # 3. Build fitur
        features = build_features(
            owm, rain_hist, hum_hist, cld_hist)

        # 4. Prediksi rain sekarang
        result_now = model_service.predict_rain(features)
        rain_now   = result_now["rain_mm"]

        # 5. Prediksi per jam
        hourly_raw = model_service.predict_multi_hour(
            features, owm, rain_hist, hum_hist, cld_hist,
            hours=24,
        )
        hourly_out = [
            {
                "hour"     : h["hour"],
                "rain_mm"  : h["rain_mm"],
                "condition": rain_to_condition(
                               h["rain_mm"], owm["clouds"]),
                "rain_prob": rain_to_probability(h["rain_mm"]),
            }
            for h in hourly_raw
        ]

        # 6. Prediksi per hari
        daily_raw = model_service.predict_multi_day(
            owm, rain_hist, hum_hist, cld_hist, days=7,
        )
        daily_out = [
            {
                "day"          : d["day"],
                "date"         : d["date"],
                "total_rain_mm": d["total_rain_mm"],
                "max_rain_mm"  : d["max_rain_mm"],
                "condition"    : rain_to_condition(
                                   d["max_rain_mm"], owm["clouds"]),
                "rain_prob"    : rain_to_probability(
                                   d["max_rain_mm"]),
            }
            for d in daily_raw
        ]

        # 7. LIME explanation
        feat_values = {
            FEATURES[i]: float(features[0][i])
            for i in range(len(FEATURES))
        }
        lime_result = lime_service.explain(features, feat_values)

        return PredictResponse(
            current_rain_mm  = rain_now,
            current_condition= rain_to_condition(
                                 rain_now, owm["clouds"]),
            rain_probability = rain_to_probability(rain_now),
            hourly_forecast  = hourly_out,
            daily_forecast   = daily_out,
            lime_features    = lime_result,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))