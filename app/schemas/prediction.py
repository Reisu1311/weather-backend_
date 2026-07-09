from pydantic import BaseModel
from typing import List, Dict

class PredictRequest(BaseModel):
    lat              : float
    lon              : float
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