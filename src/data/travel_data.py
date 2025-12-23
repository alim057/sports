from math import radians, cos, sin, asin, sqrt
from typing import Dict, Tuple, Optional

class TravelCalculator:
    """Calculates travel distances between NBA arenas."""
    
    # NBA Stadium Coordinates (Lat, Lon)
    # Approximate locations of home arenas
    STADIUM_COORDS = {
        'ATL': (33.7573, -84.3963),  # State Farm Arena
        'BOS': (42.3662, -71.0621),  # TD Garden
        'BKN': (40.6826, -73.9754),  # Barclays Center
        'CHA': (35.2251, -80.8392),  # Spectrum Center
        'CHI': (41.8807, -87.6742),  # United Center
        'CLE': (41.4965, -81.6881),  # Rocket Mortgage FieldHouse
        'DAL': (32.7905, -96.8103),  # American Airlines Center
        'DEN': (39.7487, -105.0076), # Ball Arena
        'DET': (42.3411, -83.0553),  # Little Caesars Arena
        'GSW': (37.7680, -122.3877), # Chase Center
        'HOU': (29.7508, -95.3621),  # Toyota Center
        'IND': (39.7640, -86.1555),  # Gainbridge Fieldhouse
        'LAC': (33.9450, -118.2450), # Intuit Dome (Assuming 2024 season) / Crypto.com
        'LAL': (34.0430, -118.2668), # Crypto.com Arena
        'MEM': (35.1381, -90.0506),  # FedExForum
        'MIA': (25.7814, -80.1870),  # Kaseya Center
        'MIL': (43.0451, -87.9172),  # Fiserv Forum
        'MIN': (44.9795, -93.2761),  # Target Center
        'NOP': (29.9490, -90.0821),  # Smoothie King Center
        'NYK': (40.7505, -73.9934),  # Madison Square Garden
        'OKC': (35.4634, -97.5151),  # Paycom Center
        'ORL': (28.5392, -81.3839),  # Kia Center
        'PHI': (39.9012, -75.1720),  # Wells Fargo Center
        'PHX': (33.4457, -112.0712), # Footprint Center
        'POR': (45.5316, -122.6668), # Moda Center
        'SAC': (38.5802, -121.4997), # Golden 1 Center
        'SAS': (29.4270, -98.4375),  # Frost Bank Center
        'TOR': (43.6435, -79.3791),  # Scotiabank Arena
        'UTA': (40.7683, -111.9011), # Delta Center
        'WAS': (38.8982, -77.0209)   # Capital One Arena
    }

    @staticmethod
    def get_coordinates(team_abbr: str) -> Optional[Tuple[float, float]]:
        """Get (lat, lon) for a team's stadium."""
        return TravelCalculator.STADIUM_COORDS.get(team_abbr.upper())

    @staticmethod
    def calculate_distance(team1: str, team2: str) -> float:
        """
        Calculate distance in miles between two team stadiums using Haversine formula.
        """
        coords1 = TravelCalculator.get_coordinates(team1)
        coords2 = TravelCalculator.get_coordinates(team2)
        
        if not coords1 or not coords2:
            return 0.0
            
        return TravelCalculator.haversine(coords1[0], coords1[1], coords2[0], coords2[1])

    @staticmethod
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance in miles between two points 
        on the earth (specified in decimal degrees)
        """
        # Convert decimal degrees to radians 
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # Haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 3956 # Radius of earth in miles. Use 6371 for kilometers
        return c * r
