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
    ) -> list:
        """Prediksi kondisi cuaca per jam (autoregressive)"""
        from .preprocessor import build_features

        results    = []
        temp_hist  = list(temp_history)
        hum_hist   = list(hum_history)
        rain_hist  = list(rain_history)
        cloud_hist = list(cloud_history)
        wind_hist  = list(wind_history)
        now        = datetime.now()

        for h in range(1, hours + 1):
            future_time = now + timedelta(hours=h)

            feats  = build_features(
                owm_current, temp_hist, hum_hist,
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

            # Geser history (pakai nilai current sebagai approksimasi
            # karena kita tidak tahu nilai aktual masa depan)
            temp_hist.insert(0, owm_current.get("temperature_2m", 27.0))
            hum_hist.insert(0,  owm_current.get("relative_humidity_2m", 80.0))
            rain_hist.insert(0, 0.0)
            cloud_hist.insert(0, owm_current.get("cloud_cover", 50.0))
            wind_hist.insert(0, owm_current.get("wind_speed_10m", 10.0))

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
    ) -> list:
        """Prediksi kondisi cuaca harian (ambil mode/dominan dari 24 jam)"""
        hourly = self.predict_multi_hour(
            owm_current, temp_history, hum_history,
            rain_history, cloud_history, wind_history,
            hours=days * 24,
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