# FILE: backend/app/services/model_service.py
import joblib
import numpy as np
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
        now          : datetime | None = None,
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
        # Bug fix: pakai jam yang dikirim KLIEN (HP pengguna), bukan jam
        # server (`datetime.now()`) -- lihat penjelasan lengkap di
        # `PredictRequest.client_now` (schemas/prediction.py). Server &
        # HP bisa beda timezone, dan fitur "hour"/"hour_sin"/"hour_cos"
        # di bawah butuh jam LOKAL pengguna yang sebenarnya, bukan jam
        # server.
        now = now or datetime.now()

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

        # Haluskan lompatan 1-jam yang "nyempil sendirian" di antara jam-jam
        # yang konsisten (mis. Mendung,Mendung,Cerah,Mendung,Mendung ->
        # jam "Cerah" itu ditarik jadi Mendung). Perubahan cuaca beneran
        # yang berlangsung 2 jam atau lebih TIDAK disentuh -- lihat
        # docstring _smooth_hourly untuk detail.
        results = self._smooth_hourly(results, window=3)

        return results

    def _smooth_hourly(self, results: list, window: int = 3) -> list:
        """Haluskan noise klasifikasi per-jam (majority vote antar-tetangga).

        Setiap jam diprediksi SECARA INDEPENDEN dari fitur cuaca jam itu
        saja (tidak ada "memori" hasil klasifikasi jam sebelumnya). Kalau
        nilai cloud_cover/kelembaban dari forecast Open-Meteo di satu jam
        kebetulan sedikit melewati ambang batas keputusan model, hasilnya
        bisa "lompat" sendirian lalu balik lagi 1 jam kemudian -- misal
        Mendung,Mendung,Mendung,Cerah,Mendung,Mendung. Secara matematis
        model tidak salah (memang begitu isi datanya jam itu), tapi
        secara meteorologi & dari sisi tampilan ke user ini terlihat
        "kedip-kedip" dan membingungkan, karena cuaca nyata jarang
        berubah total lalu balik lagi hanya dalam 1 jam.

        Untuk tiap jam, lihat jendela [jam-1, jam ini, jam+1] (window=3).
        Kalau kondisi jam ini KALAH SUARA dibanding mayoritas jendela itu
        (mis. 2 dari 3 tetangga bilang "Mendung", cuma jam ini "Cerah"),
        kondisi jam ini ditarik ikut mayoritas, dan confidence-nya diganti
        jadi rata-rata confidence entri yang kondisinya sama dengan hasil
        mayoritas tersebut (bukan confidence dari kondisi lama yang sudah
        dibuang).

        Perubahan yang berlangsung 2 jam berturut-turut atau lebih TIDAK
        disentuh -- itu dianggap perubahan cuaca yang genuine, bukan noise
        (karena tidak akan pernah kalah suara 2-lawan-1 di jendela manapun).
        """
        if window < 3 or len(results) < window:
            return results

        half     = window // 2
        smoothed = [dict(r) for r in results]

        for i in range(len(results)):
            lo = max(0, i - half)
            hi = min(len(results), i + half + 1)
            neighborhood = results[lo:hi]
            if len(neighborhood) < window:
                continue  # dekat ujung data, tetangga tak lengkap -> biarkan

            counts = Counter(h["condition"] for h in neighborhood)
            top_condition, top_count = counts.most_common(1)[0]

            if top_condition != results[i]["condition"] and top_count > window / 2:
                matching_conf = [
                    h["confidence"] for h in neighborhood
                    if h["condition"] == top_condition
                ]
                smoothed[i]["condition"]  = top_condition
                smoothed[i]["confidence"] = float(np.mean(matching_conf))

        return smoothed

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
        now          : datetime | None = None,
    ) -> list:
        """Prediksi kondisi cuaca harian -- REKAPAN PENUH satu hari (bukan
        cuma titik jam 12:00 siang).

        Untuk tiap hari, seluruh 24 jam prediksi (yang sudah dihaluskan
        oleh `_smooth_hourly`) dibagi jadi 2 blok waktu:
          - "Siang" : jam 06:00 - 17:59
          - "Malam" : jam 18:00 - 23:59
        Kondisi dominan (mode) dihitung terpisah untuk tiap blok.

        - Kalau kondisi dominan Siang == Malam -> hari itu dianggap
          punya SATU kondisi cuaca sepanjang hari -> backend kirim 1
          segmen saja (`segments` berisi 1 item, `label=None`).
        - Kalau berbeda -> backend kirim 2 segmen ("Siang" & "Malam")
          supaya Flutter bisa menampilkan keduanya (mis. "Siang: Cerah",
          "Malam: Hujan").

        Field `condition`/`confidence` di level atas tetap disediakan
        (= segmen pertama) untuk kompatibilitas kode lama yang belum
        memakai `segments`.

        Bug fix `now`: HARUS berasal dari jam di HP pengguna
        (`client_now`), bukan jam server -- lihat penjelasan lengkap di
        `PredictRequest.client_now` (schemas/prediction.py).
        """
        now = now or datetime.now()

        # Hitung cukup jauh ke depan supaya mencakup jam 23:00 di hari
        # TERAKHIR yang diminta (butuh seluruh hari, bukan cuma sampai
        # jam 12 siang saja seperti versi sebelumnya).
        last_day_date = (now + timedelta(days=days)).date()
        last_day_end  = datetime.combine(
            last_day_date, datetime.min.time()) + timedelta(hours=23)
        max_hour_offset = max(1, round(
            (last_day_end - now).total_seconds() / 3600))

        hourly = self.predict_multi_hour(
            owm_current, temp_history, hum_history,
            rain_history, cloud_history, wind_history,
            hours=max_hour_offset, forecast=forecast, now=now,
        )

        def summarize(entries: list) -> tuple[str, float]:
            """Mode kondisi + rata-rata confidence dari sekumpulan entri jam."""
            if not entries:
                return "Cerah Berawan", 0.5
            counts = Counter(h["condition"] for h in entries)
            cond, _ = counts.most_common(1)[0]
            conf = float(np.mean(
                [h["confidence"] for h in entries if h["condition"] == cond]))
            return cond, conf

        daily = []
        for d in range(days):
            future_day  = now + timedelta(days=d + 1)
            target_date = future_day.date()

            day_entries = [
                h for h in hourly
                if datetime.fromisoformat(h["datetime"]).date() == target_date
            ]
            siang_entries = [
                h for h in day_entries
                if 6 <= datetime.fromisoformat(h["datetime"]).hour <= 17
            ]
            malam_entries = [
                h for h in day_entries
                if datetime.fromisoformat(h["datetime"]).hour >= 18
                or datetime.fromisoformat(h["datetime"]).hour < 6
            ]

            cond_siang, conf_siang = summarize(siang_entries or day_entries)
            cond_malam, conf_malam = summarize(malam_entries or day_entries)

            if cond_siang == cond_malam:
                # Satu kondisi dominan sepanjang hari -> 1 segmen saja.
                avg_conf = float(np.mean([conf_siang, conf_malam]))
                segments = [
                    {"label": None, "condition": cond_siang, "confidence": avg_conf},
                ]
            else:
                # Kondisi siang & malam berbeda -> 2 segmen terpisah.
                segments = [
                    {"label": "Siang", "condition": cond_siang, "confidence": conf_siang},
                    {"label": "Malam", "condition": cond_malam, "confidence": conf_malam},
                ]

            daily.append({
                "day"       : future_day.strftime("%A"),
                "date"      : target_date.strftime("%Y-%m-%d"),
                "condition" : segments[0]["condition"],   # kompatibilitas lama
                "confidence": segments[0]["confidence"],  # kompatibilitas lama
                "segments"  : segments,
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
        smooth       : bool = True,
        now          : datetime | None = None,
    ):
        """Hitung fitur (40 fitur) + hasil prediksi untuk SATU titik waktu
        tertentu (hour_offset jam ke depan dari sekarang). Dipakai untuk
        menjelaskan (LIME) satu jam/hari spesifik tanpa perlu menghitung
        LIME untuk semua 24x7 titik sekaligus (berat).

        Konsistensi dengan daftar per jam/harian (PENTING):
        `predict_multi_hour` sudah menghaluskan noise klasifikasi 1-jam
        (lihat `_smooth_hourly`). Kalau fungsi ini TIDAK ikut menghaluskan
        dengan cara yang sama, popup LIME (yang pakai fungsi ini) bisa
        menampilkan kondisi mentah yang BERBEDA dari yang sudah tampil di
        daftar per jam/harian untuk jam yang sama persis -- memunculkan
        lagi masalah "list vs popup tidak sinkron" yang pernah terjadi.

        Karena itu, saat `smooth=True` (default), fungsi ini juga
        menghitung prediksi di (hour_offset-1) dan (hour_offset+1) lewat
        jalur autoregressive yang sama, lalu menerapkan voting mayoritas
        3-jam yang identik dengan `_smooth_hourly`. Fitur (untuk LIME)
        tetap diambil dari titik hour_offset yang sesungguhnya -- yang
        berubah hanya label "condition"/"confidence" yang ditampilkan
        (dan otomatis ikut menentukan kelas apa yang dijelaskan LIME).

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
        # Bug fix: pakai jam HP pengguna (client_now), bukan jam server --
        # lihat penjelasan di predict_multi_hour/PredictRequest.client_now.
        now = now or datetime.now()

        # Kalau smoothing aktif, kita perlu lanjut 1 jam lagi setelah
        # hour_offset supaya bisa tahu prediksi tetangga (hour_offset+1).
        last_h = hour_offset + 1 if smooth else hour_offset

        target_features = None
        target_result   = None
        target_time     = now
        window_results: dict[int, dict] = {}

        for h in range(1, last_h + 1):
            idx         = h - 1
            future_time = now + timedelta(hours=h)

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

            if h in (hour_offset - 1, hour_offset, hour_offset + 1):
                window_results[h] = {
                    "condition" : result["condition"],
                    "confidence": result["confidence"],
                }

            if h == hour_offset:
                target_features = feats
                target_result   = result
                target_time     = future_time

            temp_hist.insert(0, hour_weather["temperature_2m"])
            hum_hist.insert(0,  hour_weather["relative_humidity_2m"])
            rain_hist.insert(0, hour_rain)
            cloud_hist.insert(0, hour_weather["cloud_cover"])
            wind_hist.insert(0, hour_weather["wind_speed_10m"])

        final_result = dict(target_result)

        # Voting mayoritas 3-jam (sama seperti _smooth_hourly). Hanya
        # dilakukan kalau kedua tetangga (sebelum & sesudah) tersedia --
        # kalau hour_offset==1 (tidak ada "sebelum"), dibiarkan apa adanya,
        # sama seperti perlakuan _smooth_hourly di ujung data.
        if (smooth
                and (hour_offset - 1) in window_results
                and (hour_offset + 1) in window_results):
            neighborhood = [
                window_results[hour_offset - 1],
                window_results[hour_offset],
                window_results[hour_offset + 1],
            ]
            counts = Counter(r["condition"] for r in neighborhood)
            top_condition, top_count = counts.most_common(1)[0]

            if top_condition != final_result["condition"] and top_count > 1:
                matching_conf = [
                    r["confidence"] for r in neighborhood
                    if r["condition"] == top_condition
                ]
                final_result["condition"]  = top_condition
                final_result["confidence"] = float(np.mean(matching_conf))

        return target_features, final_result, target_time