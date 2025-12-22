"""
Weather Data Fetcher

Uses Open-Meteo API (FREE, no API key required) for weather data.
Useful for outdoor sports: NFL, NCAAF, MLB.
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import json


class WeatherFetcher:
    """Fetch weather data from Open-Meteo (free, no API key needed)."""
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    # Stadium coordinates for NFL/NCAAF venues
    NFL_STADIUMS = {
        'ARI': (33.5276, -112.2626),  # State Farm Stadium
        'ATL': (33.7553, -84.4009),   # Mercedes-Benz Stadium (dome)
        'BAL': (39.2780, -76.6227),   # M&T Bank Stadium
        'BUF': (42.7738, -78.7870),   # Highmark Stadium
        'CAR': (35.2258, -80.8528),   # Bank of America Stadium
        'CHI': (41.8623, -87.6167),   # Soldier Field
        'CIN': (39.0955, -84.5160),   # Paycor Stadium
        'CLE': (41.5061, -81.6995),   # Cleveland Browns Stadium
        'DAL': (32.7473, -97.0945),   # AT&T Stadium (dome)
        'DEN': (39.7439, -105.0201),  # Empower Field
        'DET': (42.3400, -83.0456),   # Ford Field (dome)
        'GB': (44.5013, -88.0622),    # Lambeau Field
        'HOU': (29.6847, -95.4107),   # NRG Stadium (dome)
        'IND': (39.7601, -86.1639),   # Lucas Oil Stadium (dome)
        'JAX': (30.3239, -81.6373),   # EverBank Stadium
        'KC': (39.0489, -94.4839),    # Arrowhead Stadium
        'LAC': (33.9534, -118.3390),  # SoFi Stadium (dome)
        'LAR': (33.9534, -118.3390),  # SoFi Stadium (dome)
        'LV': (36.0908, -115.1833),   # Allegiant Stadium (dome)
        'MIA': (25.9580, -80.2389),   # Hard Rock Stadium
        'MIN': (44.9736, -93.2575),   # U.S. Bank Stadium (dome)
        'NE': (42.0909, -71.2643),    # Gillette Stadium
        'NO': (29.9511, -90.0812),    # Caesars Superdome (dome)
        'NYG': (40.8128, -74.0742),   # MetLife Stadium
        'NYJ': (40.8128, -74.0742),   # MetLife Stadium
        'PHI': (39.9008, -75.1675),   # Lincoln Financial Field
        'PIT': (40.4468, -80.0158),   # Acrisure Stadium
        'SEA': (47.5952, -122.3316),  # Lumen Field
        'SF': (37.4033, -121.9694),   # Levi's Stadium
        'TB': (27.9759, -82.5033),    # Raymond James Stadium
        'TEN': (36.1665, -86.7713),   # Nissan Stadium
        'WAS': (38.9076, -76.8645),   # FedExField
    }
    
    DOME_STADIUMS = {'ARI', 'ATL', 'DAL', 'DET', 'HOU', 'IND', 'LAC', 'LAR', 'LV', 'MIN', 'NO'}
    
    def __init__(self):
        """Initialize weather fetcher."""
        pass
    
    def get_weather(
        self,
        latitude: float,
        longitude: float,
        date: Optional[datetime] = None
    ) -> Dict:
        """
        Get weather forecast for a location.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            date: Date for forecast (default: today)
            
        Returns:
            Weather data dict
        """
        if date is None:
            date = datetime.now()
        
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'hourly': 'temperature_2m,precipitation,wind_speed_10m,weather_code',
            'temperature_unit': 'fahrenheit',
            'wind_speed_unit': 'mph',
            'precipitation_unit': 'inch',
            'timezone': 'America/New_York',
            'forecast_days': 7
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Extract data for the target date/hour
                hourly = data.get('hourly', {})
                times = hourly.get('time', [])
                
                # Find closest hour to 7pm game time
                target_hour = date.strftime('%Y-%m-%dT19:00')
                
                for i, t in enumerate(times):
                    if t.startswith(date.strftime('%Y-%m-%d')):
                        return {
                            'temperature': hourly['temperature_2m'][i],
                            'precipitation': hourly['precipitation'][i],
                            'wind_speed': hourly['wind_speed_10m'][i],
                            'weather_code': hourly['weather_code'][i],
                            'conditions': self._decode_weather_code(hourly['weather_code'][i])
                        }
                
                return {'error': 'No data for date'}
            else:
                return {'error': f'API error: {response.status_code}'}
        except Exception as e:
            return {'error': str(e)}
    
    def get_game_weather(
        self,
        home_team: str,
        sport: str = 'nfl',
        game_date: Optional[datetime] = None
    ) -> Dict:
        """
        Get weather for a game.
        
        Args:
            home_team: Home team abbreviation
            sport: Sport (nfl, ncaaf)
            game_date: Game date
            
        Returns:
            Weather data or dome indication
        """
        team = home_team.upper()
        
        # Check if dome stadium
        if team in self.DOME_STADIUMS:
            return {
                'is_dome': True,
                'temperature': 72,
                'precipitation': 0,
                'wind_speed': 0,
                'conditions': 'Indoor (Dome)'
            }
        
        # Get stadium coordinates
        coords = self.NFL_STADIUMS.get(team)
        if not coords:
            return {'error': f'Unknown stadium for {team}'}
        
        weather = self.get_weather(coords[0], coords[1], game_date)
        weather['is_dome'] = False
        weather['stadium'] = team
        
        return weather
    
    def _decode_weather_code(self, code: int) -> str:
        """Convert WMO weather code to description."""
        codes = {
            0: 'Clear',
            1: 'Partly Cloudy',
            2: 'Cloudy',
            3: 'Overcast',
            45: 'Fog',
            48: 'Freezing Fog',
            51: 'Light Drizzle',
            53: 'Moderate Drizzle',
            55: 'Heavy Drizzle',
            61: 'Light Rain',
            63: 'Moderate Rain',
            65: 'Heavy Rain',
            66: 'Freezing Rain',
            67: 'Heavy Freezing Rain',
            71: 'Light Snow',
            73: 'Moderate Snow',
            75: 'Heavy Snow',
            77: 'Snow Grains',
            80: 'Light Showers',
            81: 'Moderate Showers',
            82: 'Heavy Showers',
            85: 'Light Snow Showers',
            86: 'Heavy Snow Showers',
            95: 'Thunderstorm',
            96: 'Thunderstorm with Hail',
            99: 'Severe Thunderstorm'
        }
        return codes.get(code, 'Unknown')
    
    def get_weather_impact(self, weather: Dict) -> Dict:
        """
        Calculate how weather affects game.
        
        Returns:
            Impact factors for scoring, passing, etc.
        """
        if weather.get('is_dome') or weather.get('error'):
            return {
                'scoring_modifier': 0,
                'passing_impact': 0,
                'kicking_impact': 0,
                'description': 'No weather impact'
            }
        
        temp = weather.get('temperature', 70)
        wind = weather.get('wind_speed', 0)
        precip = weather.get('precipitation', 0)
        
        # Calculate impacts
        scoring_mod = 0
        passing_impact = 0
        kicking_impact = 0
        
        # Cold weather reduces scoring
        if temp < 32:
            scoring_mod -= 3
            passing_impact -= 0.1
        elif temp < 50:
            scoring_mod -= 1.5
            passing_impact -= 0.05
        
        # Wind affects passing and kicking
        if wind > 20:
            passing_impact -= 0.15
            kicking_impact -= 0.2
        elif wind > 15:
            passing_impact -= 0.08
            kicking_impact -= 0.1
        
        # Precipitation
        if precip > 0.5:
            scoring_mod -= 5
            passing_impact -= 0.2
        elif precip > 0.1:
            scoring_mod -= 2
            passing_impact -= 0.1
        
        return {
            'scoring_modifier': scoring_mod,
            'passing_impact': passing_impact,
            'kicking_impact': kicking_impact,
            'description': f"{weather.get('conditions', 'Unknown')}, {temp:.0f}°F, {wind:.0f}mph wind"
        }


def main():
    """Test weather fetcher."""
    fetcher = WeatherFetcher()
    
    print("Testing Weather Fetcher (Open-Meteo - FREE)")
    print("=" * 50)
    
    # Test some NFL stadiums
    teams = ['GB', 'BUF', 'KC', 'DAL', 'MIA']
    
    for team in teams:
        weather = fetcher.get_game_weather(team, 'nfl')
        impact = fetcher.get_weather_impact(weather)
        
        print(f"\n{team}:")
        if weather.get('is_dome'):
            print(f"  Indoor stadium - no weather impact")
        elif 'error' in weather:
            print(f"  Error: {weather['error']}")
        else:
            print(f"  {impact['description']}")
            print(f"  Scoring impact: {impact['scoring_modifier']:+.1f} points")


if __name__ == '__main__':
    main()
