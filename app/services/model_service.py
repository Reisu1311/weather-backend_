# FILE: backend/app/services/model_service.py
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

MODEL_DIR = Path(__file__).parent.parent / "models"

class ModelService:
    def __init__(self):
        self.model  = joblib.load(MODEL_DIR / "weather_model.pkl")
        self.scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        self.le     = joblib.load(MODEL_DIR / "label_encoder.pkl")
        print("✅ Model klasifikasi cuaca (6 kelas) loaded")
        print(f"   Kelas: {list(self.le.classes_)}")

    def predict(self, features: np.ndarray) -> dict:
        """Klasifikasi kondisi cuaca dari 40 fitur"""
        X_scaled = self.scaler.transform(features)
        pred_idx = int(self.model.predict(X_scaled)[0])
        proba    = self.model.predict_proba(X_scaled)[0]

        label = str(self.le.inverse_transform([pred_idx])[0])

        class_proba = {
            str(self.le.inverse_transform([i])[0]): float(p)
            for i, p in enumerate(proba)
        }

        return {
            "condition"  : label,
            "confidence" : float(proba.max()),
            "class_proba": class_proba,
        }

    def predict_multi_hour(
        self,
        owm_current  : dict,
        temp_history : list,
        hum_history  : list,
        rain_history : list,
        cloud_history: list,
        wind_history : list,
        hours        : int = 24,
        forecast     : dict | None = None,
    ) -> list:
        """Prediksi kondisi cuaca per jam (autoregressive).

        `forecast` (opsional) berisi data forecast ASLI per jam ke depan
        dari Open-Meteo, mis:
            {
            "temperature_2m"      : [t+1, t+2, ...],
            "relative_humidity_2m": [...],
            "dew_point_2m"         : [...],
            "surface_pressure"     : [...],
            "cloud_cover"           : [...],
            "wind_speed_10m"         : [...],
            "wind_gusts_10m"          : [...],
            "rain"                    : [...],
            }
        Jika tersedia, nilai per jam ini dipakai langsung sebagai fitur
        cuaca untuk jam tersebut (bukan mengulang nilai `owm_current`),
        sehingga hasil prediksi per jam benar-benar mengikuti kondisi
        yang diramalkan. Jika forecast tidak tersedia untuk suatu jam
        (mis. request lama tanpa data forecast, atau melebihi rentang
        forecast yang dikirim), jam tersebut fallback ke pendekatan lama
        (pakai `owm_current`) supaya tetap kompatibel/tidak error.
        """
        from .preprocessor import build_features

        forecast = forecast or {}

        def forecast_at(key: str, idx: int, fallback: float) -> float:
            arr = forecast.get(key) or []
            if idx < len(arr):
                return float(arr[idx])
            return fallback

        results    = []
        temp_hist  = list(temp_history)
        hum_hist   = list(hum_history)
        rain_hist  = list(rain_history)
        cloud_hist = list(cloud_history)
        wind_hist  = list(wind_history)
        now        = datetime.now()

        for h in range(1, hours + 1):
            idx         = h - 1
            future_time = now + timedelta(hours=h)

            # Nilai cuaca untuk jam ini: pakai forecast asli kalau ada,
            # kalau tidak fallback ke snapshot current (pendekatan lama).
            hour_weather = {
                "temperature_2m": forecast_at(
                    "temperature_2m", idx, owm_current.get("temperature_2m", 27.0)),
                "relative_humidity_2m": forecast_at(
                    "relative_humidity_2m", idx,
                    owm_current.get("relative_humidity_2m", 80.0)),
                "dew_point_2m": forecast_at(
                    "dew_point_2m", idx, owm_current.get("dew_point_2m", 22.0)),
                "surface_pressure": forecast_at(
                    "surface_pressure", idx,
                    owm_current.get("surface_pressure", 1010.0)),
                "cloud_cover": forecast_at(
                    "cloud_cover", idx, owm_current.get("cloud_cover", 50.0)),
                "wind_speed_10m": forecast_at(
                    "wind_speed_10m", idx,
                    owm_current.get("wind_speed_10m", 10.0)),
                "wind_gusts_10m": forecast_at(
                    "wind_gusts_10m", idx,
                    owm_current.get("wind_gusts_10m", 15.0)),
            }
            hour_rain = forecast_at("rain", idx, 0.0)

            feats  = build_features(
                hour_weather, temp_hist, hum_hist,
                rain_hist, cloud_hist, wind_hist,
                future_time,
            )
            result = self.predict(feats)

            results.append({
                "hour"      : future_time.strftime("%H:00"),
                "datetime"  : future_time.isoformat(),
                "condition" : result["condition"],
                "confidence": result["confidence"],
            })

            # Geser history pakai nilai jam ini yang BARU SAJA dipakai
            # (forecast asli kalau ada), bukan selalu owm_current —
            # supaya fitur lag/rolling mencerminkan tren cuaca yang
            # sesungguhnya diramalkan, bukan garis datar.
            temp_hist.insert(0, hour_weather["temperature_2m"])
            hum_hist.insert(0,  hour_weather["relative_humidity_2m"])
            rain_hist.insert(0, hour_rain)
            cloud_hist.insert(0, hour_weather["cloud_cover"])
            wind_hist.insert(0, hour_weather["wind_speed_10m"])

        return results

    def predict_multi_day(
        self,
        owm_current  : dict,
        temp_history : list,
        hum_history  : list,
        rain_history : list,
        cloud_history: list,
        wind_history : list,
        days         : int = 7,
        forecast     : dict | None = None,
    ) -> list:
        """Prediksi kondisi cuaca harian (ambil mode/dominan dari 24 jam)"""
        hourly = self.predict_multi_hour(
            owm_current, temp_history, hum_history,
            rain_history, cloud_history, wind_history,
            hours=days * 24, forecast=forecast,
        )

        now   = datetime.now()
        daily = []
        for d in range(days):
            day_data   = hourly[d * 24 : (d + 1) * 24]
            conditions = [h["condition"] for h in day_data]

            # Ambil kondisi paling sering muncul (mode) di hari itu
            from collections import Counter
            condition_dominan = Counter(conditions).most_common(1)[0][0]
            avg_confidence    = float(np.mean(
                [h["confidence"] for h in day_data]))

            future_day = now + timedelta(days=d + 1)
            daily.append({
                "day"       : future_day.strftime("%A"),
                "date"      : future_day.strftime("%Y-%m-%d"),
                "condition" : condition_dominan,
                "confidence": avg_confidence,
            })
        return daily

    def predict_point(
        self,
        owm_current  : dict,
        temp_history : list,
        hum_history  : list,
        rain_history : list,
        cloud_history: list,
        wind_history : list,
        hour_offset  : int,
        forecast     : dict | None = None,
    ):
        """Hitung fitur (40 fitur) + hasil prediksi untuk SATU titik waktu
        tertentu (hour_offset jam ke depan dari sekarang). Dipakai untuk
        menjelaskan (LIME) satu jam/hari spesifik tanpa perlu menghitung
        LIME untuk semua 24x7 titik sekaligus (berat).

        Mengembalikan tuple: (features, result, target_time)
        - features    : np.ndarray (1, n_fitur) — dipakai untuk LIME
        - result      : dict {condition, confidence, class_proba}
        - target_time : datetime waktu yang dijelaskan
        """
        from .preprocessor import build_features

        forecast = forecast or {}

        def forecast_at(key: str, idx: int, fallback: float) -> float:
            arr = forecast.get(key) or []
            if idx < len(arr):
                return float(arr[idx])
            return fallback

        temp_hist  = list(temp_history)
        hum_hist   = list(hum_history)
        rain_hist  = list(rain_history)
        cloud_hist = list(cloud_history)
        wind_hist  = list(wind_history)
        now        = datetime.now()

        features    = None
        target_time = now

        for h in range(1, hour_offset + 1):
            idx         = h - 1
            target_time = now + timedelta(hours=h)

            hour_weather = {
                "temperature_2m": forecast_at(
                    "temperature_2m", idx, owm_current.get("temperature_2m", 27.0)),
                "relative_humidity_2m": forecast_at(
                    "relative_humidity_2m", idx,
                    owm_current.get("relative_humidity_2m", 80.0)),
                "dew_point_2m": forecast_at(
                    "dew_point_2m", idx, owm_current.get("dew_point_2m", 22.0)),
                "surface_pressure": forecast_at(
                    "surface_pressure", idx,
                    owm_current.get("surface_pressure", 1010.0)),
                "cloud_cover": forecast_at(
                    "cloud_cover", idx, owm_current.get("cloud_cover", 50.0)),
                "wind_speed_10m": forecast_at(
                    "wind_speed_10m", idx,
                    owm_current.get("wind_speed_10m", 10.0)),
                "wind_gusts_10m": forecast_at(
                    "wind_gusts_10m", idx,
                    owm_current.get("wind_gusts_10m", 15.0)),
            }
            hour_rain = forecast_at("rain", idx, 0.0)

            features = build_features(
                hour_weather, temp_hist, hum_hist,
                rain_hist, cloud_hist, wind_hist,
                target_time,
            )

            temp_hist.insert(0, hour_weather["temperature_2m"])
            hum_hist.insert(0,  hour_weather["relative_humidity_2m"])
            rain_hist.insert(0, hour_rain)
            cloud_hist.insert(0, hour_weather["cloud_cover"])
            wind_hist.insert(0, hour_weather["wind_speed_10m"])

        result = self.predict(features)
        return features, result, target_time