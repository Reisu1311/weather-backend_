# FILE: backend/app/schemas/prediction.py
from pydantic import BaseModel
from typing import List, Dict

class PredictRequest(BaseModel):
    lat              : float
    lon              : float
    # Jam SAAT INI di HP pengguna (ISO 8601, mis. "2026-07-17T14:04:00").
    #
    # PENTING: backend (server Render) dan HP pengguna bisa berada di
    # timezone yang BEDA (server biasanya UTC, HP di WIB/UTC+7). Kalau
    # backend memakai jamnya SENDIRI (`datetime.now()`) untuk menghitung
    # "jam berapa sekarang" / "jam 12 siang besok itu berapa jam lagi",
    # hasilnya bisa meleset berjam-jam dari yang sebenarnya, karena:
    #  1) Fitur "hour"/"hour_sin"/"hour_cos" yang dikirim ke model jadi
    #     salah (memakai jam server, bukan jam lokal yang sebenarnya).
    #  2) Titik "jam 12 siang" untuk popup harian bisa menunjuk ke index
    #     array forecast yang salah (array itu disusun Flutter mulai
    #     dari JAM DI HP, bukan jam server) -- inilah yang menyebabkan
    #     nilai "Suhu" di LIME (mis. 29.2°C) berbeda dari suhu jam 12:00
    #     yang ditampilkan di header popup (mis. 30.6°C).
    #
    # Kalau field ini tidak dikirim (klien lama), backend fallback pakai
    # jamnya sendiri seperti sebelumnya -- tetap jalan, cuma berpotensi
    # kurang presisi seperti dijelaskan di atas.
    client_now: str | None = None
    # Data cuaca realtime dari Open-Meteo (dikirim Flutter)
    temperature_2m       : float = 27.0
    relative_humidity_2m : float = 80.0
    dew_point_2m          : float = 22.0
    surface_pressure      : float = 1010.0
    cloud_cover            : float = 50.0
    wind_speed_10m          : float = 10.0
    wind_gusts_10m           : float = 15.0
    # Histori jam sebelumnya (opsional, default pakai current)
    temp_history    : List[float] = []
    humidity_history: List[float] = []
    rain_history     : List[float] = []
    cloud_history    : List[float] = []
    wind_history      : List[float] = []
    # Forecast ASLI per jam ke depan dari Open-Meteo (dikirim Flutter).
    # index 0 = 1 jam dari sekarang, index 1 = 2 jam dari sekarang, dst.
    # Dipakai supaya prediksi per jam & harian benar-benar mengikuti
    # kondisi cuaca yang diramalkan per jam, bukan mengulang nilai
    # "current" terus-menerus (yang sebelumnya membuat hasil prediksi
    # per jam & harian selalu sama dengan prediksi saat ini).
    forecast_temp       : List[float] = []
    forecast_humidity   : List[float] = []
    forecast_dew_point  : List[float] = []
    forecast_pressure   : List[float] = []
    forecast_cloud      : List[float] = []
    forecast_wind       : List[float] = []
    forecast_wind_gusts : List[float] = []
    forecast_rain       : List[float] = []

class LimeFeature(BaseModel):
    feature    : str
    weight     : float
    human_label: str
    value      : str

class HourlyPrediction(BaseModel):
    hour      : str
    condition : str
    confidence: float

class DailyPrediction(BaseModel):
    day       : str
    date      : str
    condition : str
    confidence: float

class PredictResponse(BaseModel):
    current_condition: str
    confidence       : float
    class_proba      : Dict[str, float]
    hourly_forecast  : List[HourlyPrediction]
    daily_forecast   : List[DailyPrediction]
    lime_features    : List[LimeFeature]


# ── Prediksi + LIME untuk SATU titik waktu (dipakai layar Per Jam & Per Hari) ──
class PointPredictRequest(PredictRequest):
    target_type : str  # "hour" atau "day"
    target_index: int  # index ke-berapa (0-based) dari hourly_forecast / daily_forecast

class PointPredictResponse(BaseModel):
    label        : str
    condition    : str
    confidence   : float
    class_proba  : Dict[str, float]
    lime_features: List[LimeFeature]