# FILE: backend/app/routers/predict.py
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from ..schemas.prediction import (
    PredictRequest, PredictResponse,
    PointPredictRequest, PointPredictResponse,
)
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

        # 4b. Data forecast ASLI per jam ke depan (dari Open-Meteo, dikirim
        # Flutter). Dipakai untuk prediksi per jam & harian di bawah ini
        # supaya tidak sekadar mengulang kondisi "current".
        forecast = {
            "temperature_2m"      : req.forecast_temp,
            "relative_humidity_2m": req.forecast_humidity,
            "dew_point_2m"         : req.forecast_dew_point,
            "surface_pressure"     : req.forecast_pressure,
            "cloud_cover"           : req.forecast_cloud,
            "wind_speed_10m"         : req.forecast_wind,
            "wind_gusts_10m"          : req.forecast_wind_gusts,
            "rain"                     : req.forecast_rain,
        }

        # 5. Prediksi per jam (24 jam ke depan)
        hourly_raw = model_service.predict_multi_hour(
            current, temp_hist, hum_hist,
            rain_hist, cloud_hist, wind_hist, hours=24,
            forecast=forecast,
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
            forecast=forecast,
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


@router.post("/predict/point", response_model=PointPredictResponse)
async def predict_point(req: PointPredictRequest):
    try:
        from ..main import model_service, lime_service

        current = {
            "temperature_2m"      : req.temperature_2m,
            "relative_humidity_2m": req.relative_humidity_2m,
            "dew_point_2m"         : req.dew_point_2m,
            "surface_pressure"     : req.surface_pressure,
            "cloud_cover"           : req.cloud_cover,
            "wind_speed_10m"         : req.wind_speed_10m,
            "wind_gusts_10m"          : req.wind_gusts_10m,
        }
        temp_hist  = req.temp_history     or [req.temperature_2m] * 3
        hum_hist   = req.humidity_history or [req.relative_humidity_2m] * 3
        rain_hist  = req.rain_history      or [0.0] * 3
        cloud_hist = req.cloud_history     or [req.cloud_cover] * 3
        wind_hist  = req.wind_history       or [req.wind_speed_10m] * 3

        forecast = {
            "temperature_2m"      : req.forecast_temp,
            "relative_humidity_2m": req.forecast_humidity,
            "dew_point_2m"         : req.forecast_dew_point,
            "surface_pressure"     : req.forecast_pressure,
            "cloud_cover"           : req.forecast_cloud,
            "wind_speed_10m"         : req.forecast_wind,
            "wind_gusts_10m"          : req.forecast_wind_gusts,
            "rain"                     : req.forecast_rain,
        }

        # "day" -> ambil jam 12:00 SIANG (kalender) pada tanggal target
        # sebagai representasi kondisi dominan hari itu.
        #
        # Bug fix: sebelumnya `hour_offset = target_index*24 + 12` dihitung
        # relatif dari waktu SAAT REQUEST dibuat (now), bukan relatif dari
        # kalender. Akibatnya "jam 12 siang" bisa meleset jauh dari jam 12
        # siang yang sesungguhnya tergantung jam berapa user membuka app --
        # misal kalau dibuka jam 08:00, target_index=0 (besok) malah
        # menghitung ke jam 20:00 HARI INI, bukan jam 12:00 besok. Ini juga
        # tidak konsisten dengan `date` yang dikembalikan oleh
        # predict_multi_day (yang sudah dihitung berbasis tanggal kalender
        # via `future_day = now + timedelta(days=d+1)`).
        #
        # Sekarang dihitung berbasis tanggal kalender yang sama persis
        # dengan yang dipakai predict_multi_day, supaya popup LIME di
        # Flutter (yang juga mencari titik jam 12:00 di tanggal kalender
        # yang sama) selalu menjelaskan titik waktu yang benar-benar sama
        # dengan yang ditampilkan.
        now = datetime.now()
        if req.target_type == "day":
            target_date = (now + timedelta(days=req.target_index + 1)).date()
            target_noon = datetime.combine(
                target_date, datetime.min.time()) + timedelta(hours=12)
            hour_offset = max(1, round(
                (target_noon - now).total_seconds() / 3600))
        else:
            hour_offset = req.target_index + 1

        features, result, target_time = model_service.predict_point(
            current, temp_hist, hum_hist, rain_hist, cloud_hist, wind_hist,
            hour_offset=hour_offset, forecast=forecast,
        )

        feat_values = {
            FEATURES[i]: float(features[0][i])
            for i in range(len(FEATURES))
        }
        lime_result = lime_service.explain(
            features, feat_values, result["condition"])

        condition_id = CLASS_LABEL_ID.get(
            result["condition"], result["condition"])
        class_proba_id = {
            CLASS_LABEL_ID.get(k, k): v
            for k, v in result["class_proba"].items()
        }

        return PointPredictResponse(
            label=target_time.strftime("%Y-%m-%d %H:%M"),
            condition=condition_id,
            confidence=result["confidence"],
            class_proba=class_proba_id,
            lime_features=lime_result,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))