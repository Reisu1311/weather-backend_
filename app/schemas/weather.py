from pydantic import BaseModel
from typing import Optional

class WeatherRequest(BaseModel):
    lat: float
    lon: float

class CurrentWeatherResponse(BaseModel):
    temp       : float
    feels_like : float
    humidity   : int
    condition  : str
    wind_speed : float
    pressure   : int
    visibility : int
    clouds     : int
    dew_point  : float