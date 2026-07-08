import joblib
import numpy as np
import math
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter

MODEL_DIR = Path(__file__).parent.parent / "models"

class ModelService:
    def __init__(self):
        self.model  = joblib.load(MODEL_DIR / "weather_model.pkl")
        self.scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        self.le     = joblib.load(MODEL_DIR / "label_encoder.pkl")
        print("✅ Model klasifikasi cuaca (6 kelas) loaded")
        print(f"   Kelas: {list(self.le.classes_)}")

    def predict(self, features: np.ndarray) -> dict:
        X_scaled = self.scaler.transform(features)
        pred_idx = int(self.model.predict(X_scaled)[0])
        proba    = self.model.predict_proba(X_scaled)[0]
        label    = str(self.le.inverse_transform([pred_idx])[0])
        class_proba = {
            str(self.le.inverse_transform([i])[0]): float(p)
            for i, p in enumerate(proba)
        }
        return {
            "condition"  : label,
            "confidence" : float(proba.max()),
            "class_proba": class_proba,
        }

    def _estimate_rain(self, condition: str) -> float:
        """Estimasi curah hujan berdasarkan kondisi terprediksi."""
        if condition == "Rain":
            return 2.5
        elif condition == "Drizzle":
            return 0.5
        return 0.0

    def _estimate_cloud(self, condition: str, prev_cloud: float) -> float:
        """Estimasi tutupan awan berikutnya berdasarkan kondisi."""
        if condition in ["Clear Sky"]:
            return max(5.0, prev_cloud - 8.0)
        elif condition in ["Mainly Clear"]:
            return max(15.0, prev_cloud - 4.0)
        elif condition in ["Partly Cloudy"]:
            # Konvergensi ke 50%
            return prev_cloud + (50.0 - prev_cloud) * 0.2
        elif condition in ["Overcast"]:
            return min(100.0, prev_cloud + 4.0)
        elif condition in ["Drizzle", "Rain"]:
            return min(100.0, prev_cloud + 5.0)
        return prev_cloud

    def _diurnal_temp(self, base_temp: float, hour: int) -> float:
        """
        Estimasi variasi suhu diurnal tropis.
        Puncak ~14:00, minimum ~05:00.
        Amplitudo ~3°C.
        """
        angle = 2 * math.pi * (hour - 5) / 24
        return base_temp + 1.5 * math.sin(angle)

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
        from .preprocessor import build_features

        results    = []
        temp_hist  = list(temp_history)
        hum_hist   = list(hum_history)
        rain_hist  = list(rain_history)
        cloud_hist = list(cloud_history)
        wind_hist  = list(wind_history)
        now        = datetime.now()

        # Nilai awal dari data cuaca saat ini
        base_temp  = owm_current.get("temperature_2m", 27.0)
        prev_hum   = owm_current.get("relative_humidity_2m", 80.0)
        prev_cloud = owm_current.get("cloud_cover", 50.0)
        prev_wind  = owm_current.get("wind_speed_10m", 10.0)

        for h in range(1, hours + 1):
            future_time = now + timedelta(hours=h)

            # Estimasi nilai fitur untuk jam ini
            est_temp  = self._diurnal_temp(base_temp, future_time.hour)
            est_cloud = prev_cloud
            est_hum   = prev_hum

            # Kelembaban cenderung naik malam hari
            if future_time.hour >= 20 or future_time.hour <= 5:
                est_hum = min(95.0, prev_hum + 1.5)
            elif 10 <= future_time.hour <= 16:
                est_hum = max(50.0, prev_hum - 1.0)

            current_for_hour = {
                "temperature_2m"      : est_temp,
                "relative_humidity_2m": est_hum,
                "dew_point_2m"        : owm_current.get("dew_point_2m", 22.0),
                "surface_pressure"    : owm_current.get("surface_pressure", 1010.0),
                "cloud_cover"         : est_cloud,
                "wind_speed_10m"      : prev_wind,
                "wind_gusts_10m"      : owm_current.get("wind_gusts_10m", 15.0),
            }

            feats  = build_features(
                current_for_hour,
                temp_hist, hum_hist,
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

            # Estimasi nilai untuk iterasi berikutnya
            est_rain  = self._estimate_rain(result["condition"])
            next_cloud = self._estimate_cloud(result["condition"], est_cloud)

            # Geser histori dengan nilai estimasi jam ini
            temp_hist.insert(0, est_temp)
            hum_hist.insert(0, est_hum)
            rain_hist.insert(0, est_rain)
            cloud_hist.insert(0, est_cloud)
            wind_hist.insert(0, prev_wind)

            # Potong histori agar tidak terlalu panjang
            temp_hist  = temp_hist[:6]
            hum_hist   = hum_hist[:6]
            rain_hist  = rain_hist[:6]
            cloud_hist = cloud_hist[:6]
            wind_hist  = wind_hist[:6]

            # Update untuk iterasi berikutnya
            prev_cloud = next_cloud
            prev_hum   = est_hum

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
            # Kondisi dominan (modus)
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