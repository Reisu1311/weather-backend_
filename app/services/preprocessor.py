import numpy as np
from collections import deque

# ── 40 Fitur persis sesuai urutan training ──────────────────
FEATURES = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "surface_pressure", "cloud_cover", "wind_speed_10m", "wind_gusts_10m",
    "hour", "month", "day_of_week",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "temperature_2m_lag1", "temperature_2m_lag2", "temperature_2m_lag3",
    "relative_humidity_2m_lag1", "relative_humidity_2m_lag2", "relative_humidity_2m_lag3",
    "rain_lag1", "rain_lag2", "rain_lag3",
    "cloud_cover_lag1", "cloud_cover_lag2", "cloud_cover_lag3",
    "wind_speed_10m_lag1", "wind_speed_10m_lag2", "wind_speed_10m_lag3",
    "temperature_2m_roll3", "temperature_2m_roll6",
    "relative_humidity_2m_roll3", "relative_humidity_2m_roll6",
    "rain_roll3", "rain_roll6",
    "cloud_cover_roll3", "cloud_cover_roll6",
    "temp_humidity", "dew_depression", "wind_cloud",
]

# ── 6 Kelas kondisi cuaca (urutan alfabetis sesuai LabelEncoder) ──
CLASS_NAMES = [
    "Clear Sky", "Drizzle", "Mainly Clear",
    "Overcast", "Partly Cloudy", "Rain",
]

CLASS_LABEL_ID = {
    "Clear Sky"    : "Cerah",
    "Mainly Clear" : "Cerah Berawan",
    "Partly Cloudy": "Berawan Sebagian",
    "Overcast"     : "Berawan Tebal",
    "Drizzle"      : "Gerimis",
    "Rain"         : "Hujan",
}

CLASS_EMOJI = {
    "Clear Sky"    : "☀️",
    "Mainly Clear" : "🌤️",
    "Partly Cloudy": "⛅",
    "Overcast"     : "☁️",
    "Drizzle"      : "🌦️",
    "Rain"         : "🌧️",
}

# ── Label nama fitur untuk LIME (ditampilkan ke user) ────────
HUMAN_LABELS = {
    "temperature_2m"             : "Suhu",
    "relative_humidity_2m"       : "Kelembaban",
    "dew_point_2m"                : "Titik Embun",
    "surface_pressure"           : "Tekanan Udara",
    "cloud_cover"                 : "Tutupan Awan",
    "wind_speed_10m"              : "Kecepatan Angin",
    "wind_gusts_10m"               : "Hembusan Angin",
    "hour"                         : "Jam",
    "month"                        : "Bulan",
    "day_of_week"                  : "Hari dalam Minggu",
    "hour_sin"                     : "Pola Jam (sin)",
    "hour_cos"                     : "Pola Jam (cos)",
    "month_sin"                    : "Pola Bulan (sin)",
    "month_cos"                    : "Pola Bulan (cos)",
    "temperature_2m_lag1"          : "Suhu 1 Jam Lalu",
    "temperature_2m_lag2"          : "Suhu 2 Jam Lalu",
    "temperature_2m_lag3"          : "Suhu 3 Jam Lalu",
    "relative_humidity_2m_lag1"    : "Kelembaban 1 Jam Lalu",
    "relative_humidity_2m_lag2"    : "Kelembaban 2 Jam Lalu",
    "relative_humidity_2m_lag3"    : "Kelembaban 3 Jam Lalu",
    "rain_lag1"                    : "Hujan 1 Jam Lalu",
    "rain_lag2"                    : "Hujan 2 Jam Lalu",
    "rain_lag3"                    : "Hujan 3 Jam Lalu",
    "cloud_cover_lag1"             : "Awan 1 Jam Lalu",
    "cloud_cover_lag2"             : "Awan 2 Jam Lalu",
    "cloud_cover_lag3"             : "Awan 3 Jam Lalu",
    "wind_speed_10m_lag1"          : "Angin 1 Jam Lalu",
    "wind_speed_10m_lag2"          : "Angin 2 Jam Lalu",
    "wind_speed_10m_lag3"          : "Angin 3 Jam Lalu",
    "temperature_2m_roll3"         : "Rata-rata Suhu 3 Jam",
    "temperature_2m_roll6"         : "Rata-rata Suhu 6 Jam",
    "relative_humidity_2m_roll3"   : "Rata-rata Kelembaban 3 Jam",
    "relative_humidity_2m_roll6"   : "Rata-rata Kelembaban 6 Jam",
    "rain_roll3"                   : "Rata-rata Hujan 3 Jam",
    "rain_roll6"                   : "Rata-rata Hujan 6 Jam",
    "cloud_cover_roll3"            : "Rata-rata Awan 3 Jam",
    "cloud_cover_roll6"            : "Rata-rata Awan 6 Jam",
    "temp_humidity"                : "Interaksi Suhu × Kelembaban",
    "dew_depression"               : "Selisih Suhu - Titik Embun",
    "wind_cloud"                   : "Interaksi Angin × Awan",
}


def build_features(
    current      : dict,   # data cuaca saat ini dari Open-Meteo
    temp_history : list,   # [t-1, t-2, t-3, ...] suhu jam sebelumnya
    hum_history  : list,   # kelembaban jam sebelumnya
    rain_history : list,   # curah hujan jam sebelumnya
    cloud_history: list,   # tutupan awan jam sebelumnya
    wind_history : list,   # kecepatan angin jam sebelumnya
    dt,                     # datetime objek waktu prediksi
) -> np.ndarray:
    """Bangun array 40 fitur persis sesuai urutan training."""

    def get(arr, idx, default=0.0):
        return float(arr[idx]) if len(arr) > idx else default

    def roll_mean(arr, n):
        vals = [get(arr, i) for i in range(min(n, len(arr)))]
        return float(np.mean(vals)) if vals else 0.0

    hour  = dt.hour
    month = dt.month
    dow   = dt.weekday()

    temp     = current.get("temperature_2m", 27.0)
    humidity = current.get("relative_humidity_2m", 80.0)
    dew      = current.get("dew_point_2m", 22.0)

    row = [
        temp,
        humidity,
        dew,
        current.get("surface_pressure", 1010.0),
        current.get("cloud_cover", 50.0),
        current.get("wind_speed_10m", 10.0),
        current.get("wind_gusts_10m", 15.0),
        float(hour),
        float(month),
        float(dow),
        np.sin(2 * np.pi * hour  / 24),
        np.cos(2 * np.pi * hour  / 24),
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12),
        # Lag suhu
        get(temp_history, 0), get(temp_history, 1), get(temp_history, 2),
        # Lag kelembaban
        get(hum_history, 0), get(hum_history, 1), get(hum_history, 2),
        # Lag rain
        get(rain_history, 0), get(rain_history, 1), get(rain_history, 2),
        # Lag cloud
        get(cloud_history, 0), get(cloud_history, 1), get(cloud_history, 2),
        # Lag wind
        get(wind_history, 0), get(wind_history, 1), get(wind_history, 2),
        # Rolling suhu
        roll_mean(temp_history, 3), roll_mean(temp_history, 6),
        # Rolling humidity
        roll_mean(hum_history, 3), roll_mean(hum_history, 6),
        # Rolling rain
        roll_mean(rain_history, 3), roll_mean(rain_history, 6),
        # Rolling cloud
        roll_mean(cloud_history, 3), roll_mean(cloud_history, 6),
        # Interaksi
        temp * humidity,
        temp - dew,
        current.get("wind_speed_10m", 10.0) * current.get("cloud_cover", 50.0),
    ]
    return np.array([row])