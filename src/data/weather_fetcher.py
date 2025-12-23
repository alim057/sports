
import requests
from datetime import datetime
from typing import Dict, Optional, Tuple
import pandas as pd

class WeatherFetcher:
    """
    Fetches weather data for games using Open-Meteo (Free, No Key).
    """
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    # Mapping of Team -> (Latitude, Longitude, Stadium Name, Dome?)
    # Dome teams don't need weather.
    STADIUMS = {
        # NFL
        "Kansas City Chiefs": (39.0489, -94.4839, "Arrowhead Stadium", False),
        "Buffalo Bills": (42.7738, -78.7870, "Highmark Stadium", False),
        "Green Bay Packers": (44.5013, -88.0622, "Lambeau Field", False),
        "Chicago Bears": (41.8623, -87.6167, "Soldier Field", False),
        "San Francisco 49ers": (37.4014, -121.9700, "Levi's Stadium", False),
        "Seattle Seahawks": (47.5952, -122.3316, "Lumen Field", False),
        "Denver Broncos": (39.7439, -105.0201, "Empower Field", False),
        "Cleveland Browns": (41.5061, -81.6995, "Cleveland Browns Stadium", False),
        "Pittsburgh Steelers": (40.4468, -80.0158, "Acrisure Stadium", False),
        "Philadelphia Eagles": (39.9012, -75.1675, "Lincoln Financial Field", False),
        "New England Patriots": (42.0909, -71.2643, "Gillette Stadium", False),
        "New York Giants": (40.8128, -74.0742, "MetLife Stadium", False),
        "New York Jets": (40.8128, -74.0742, "MetLife Stadium", False),
        "Baltimore Ravens": (39.2780, -76.6227, "M&T Bank Stadium", False),
        "Cincinnati Bengals": (39.0955, -84.5161, "Paycor Stadium", False),
        "Tennessee Titans": (36.1665, -86.7713, "Nissan Stadium", False),
        "Jacksonville Jaguars": (30.3240, -81.6373, "EverBank Stadium", False),
        "Miami Dolphins": (25.9580, -80.2389, "Hard Rock Stadium", False),
        "Tampa Bay Buccaneers": (27.9759, -82.5033, "Raymond James Stadium", False),
        "Carolina Panthers": (35.2258, -80.8528, "Bank of America Stadium", False),
        "Washington Commanders": (38.9076, -76.8645, "Commanders Field", False),
        
        # Domes (Weather doesn't matter, but good to know)
        "Dallas Cowboys": (32.7473, -97.0945, "AT&T Stadium", True),
        "Minnesota Vikings": (44.9735, -93.2575, "U.S. Bank Stadium", True),
        "Detroit Lions": (42.3400, -83.0456, "Ford Field", True),
        "Atlanta Falcons": (33.7554, -84.4010, "Mercedes-Benz Stadium", True),
        "New Orleans Saints": (29.9511, -90.0812, "Caesars Superdome", True),
        "Houston Texans": (29.6847, -95.4107, "NRG Stadium", True),
        "Indianapolis Colts": (39.7601, -86.1639, "Lucas Oil Stadium", True),
        "Arizona Cardinals": (33.5276, -112.2626, "State Farm Stadium", True),
        "Las Vegas Raiders": (36.0909, -115.1833, "Allegiant Stadium", True),
        "Los Angeles Rams": (33.9534, -118.3390, "SoFi Stadium", True),
        "Los Angeles Chargers": (33.9534, -118.3390, "SoFi Stadium", True),
    }

    def get_weather(self, team_name: str, game_time: datetime) -> Dict:
        """
        Get weather forecast for a game.
        
        Args:
            team_name: Home team name
            game_time: DateTime of the game
            
        Returns:
            Dict with temp_f, wind_mph, precip_prob, condition
        """
        # Default response
        weather = {
            "temp_f": None,
            "wind_mph": None,
            "condition": "Unknown",
            "is_dome": False
        }
        
        info = self.STADIUMS.get(team_name)
        
        # Try partial match if exact match fails
        if not info:
             for key, val in self.STADIUMS.items():
                 if team_name in key or key in team_name:
                     info = val
                     break
        
        if not info:
            return weather
            
        lat, lon, stadium, is_dome = info
        weather["is_dome"] = is_dome
        weather["stadium"] = stadium
        
        if is_dome:
            weather["condition"] = "Dome"
            return weather
            
        try:
            # Open-Meteo API
            # Hourly forecast for temp, wind, rain
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,precipitation_probability,wind_speed_10m",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "start_date": game_time.strftime("%Y-%m-%d"),
                "end_date": game_time.strftime("%Y-%m-%d")
            }
            
            resp = requests.get(self.BASE_URL, params=params)
            data = resp.json()
            
            if "hourly" in data:
                # Find hour index
                hour = game_time.hour
                
                # Verify index bounds
                if 0 <= hour < len(data["hourly"]["time"]):
                    weather["temp_f"] = data["hourly"]["temperature_2m"][hour]
                    weather["wind_mph"] = data["hourly"]["wind_speed_10m"][hour]
                    precip = data["hourly"]["precipitation_probability"][hour]
                    
                    # Determine condition string
                    if precip > 50:
                        weather["condition"] = "Rain/Snow"
                    elif weather["wind_mph"] > 15:
                        weather["condition"] = "Windy"
                    else:
                        weather["condition"] = "Clear"
                        
        except Exception as e:
            print(f"Error fetching weather for {team_name}: {e}")
            
        return weather

if __name__ == "__main__":
    fetcher = WeatherFetcher()
    now = datetime.now()
    
    print("Testing Weather Fetcher...")
    
    # Test Outdoor (Buffalo)
    print("\nBuffalo Bills (Outdoor):")
    w = fetcher.get_weather("Buffalo Bills", now)
    print(w)
    
    # Test Dome (Dallas)
    print("\nDallas Cowboys (Dome):")
    w = fetcher.get_weather("Dallas Cowboys", now)
    print(w)
