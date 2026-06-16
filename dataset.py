# ============================================================
# dataset.py – Sample data for the Tourist Route Planner
# Contains cities, roads, attractions, hotels, and probabilities
# ============================================================

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ── City / Attraction data structures ───────────────────────

@dataclass
class Attraction:
    name: str
    entry_fee: float          # INR
    time_required: float      # hours
    satisfaction: float       # 0-10 scale
    category: str             # heritage / nature / adventure / food

@dataclass
class City:
    name: str
    state: str
    hotel_cost: float         # INR per night
    attractions: List[Attraction] = field(default_factory=list)
    latitude: float = 0.0
    longitude: float = 0.0

# ── Road / Edge ──────────────────────────────────────────────

@dataclass
class Road:
    from_city: str
    to_city: str
    distance: float           # km
    travel_cost: float        # INR (bus/train avg)
    travel_time: float        # hours

# ─────────────────────────────────────────────────────────────
#  SAMPLE DATASET  (India – popular tourist circuit)
# ─────────────────────────────────────────────────────────────

CITIES: Dict[str, City] = {
    "Delhi": City(
        name="Delhi", state="Delhi",
        hotel_cost=3000, latitude=28.6139, longitude=77.2090,
        attractions=[
            Attraction("Red Fort",        500,  2.0, 9.0, "heritage"),
            Attraction("Qutub Minar",     600,  1.5, 8.5, "heritage"),
            Attraction("India Gate",        0,  1.0, 7.5, "heritage"),
            Attraction("Chandni Chowk",     0,  2.0, 8.0, "food"),
        ]
    ),
    "Agra": City(
        name="Agra", state="Uttar Pradesh",
        hotel_cost=2500, latitude=27.1767, longitude=78.0081,
        attractions=[
            Attraction("Taj Mahal",      1100,  3.0, 10.0, "heritage"),
            Attraction("Agra Fort",       550,  2.0,  8.5, "heritage"),
            Attraction("Fatehpur Sikri",  610,  2.5,  8.0, "heritage"),
        ]
    ),
    "Jaipur": City(
        name="Jaipur", state="Rajasthan",
        hotel_cost=2800, latitude=26.9124, longitude=75.7873,
        attractions=[
            Attraction("Amber Fort",      500,  3.0, 9.5, "heritage"),
            Attraction("Hawa Mahal",      200,  1.5, 8.5, "heritage"),
            Attraction("City Palace",     700,  2.0, 8.0, "heritage"),
            Attraction("Jantar Mantar",   200,  1.0, 7.5, "heritage"),
        ]
    ),
    "Udaipur": City(
        name="Udaipur", state="Rajasthan",
        hotel_cost=3500, latitude=24.5854, longitude=73.7125,
        attractions=[
            Attraction("City Palace Udaipur", 300, 3.0, 9.5, "heritage"),
            Attraction("Lake Pichola",         0,  2.0, 9.0, "nature"),
            Attraction("Saheliyon Ki Bari",   50,  1.0, 7.5, "nature"),
        ]
    ),
    "Mumbai": City(
        name="Mumbai", state="Maharashtra",
        hotel_cost=5000, latitude=19.0760, longitude=72.8777,
        attractions=[
            Attraction("Gateway of India",    0,  1.0, 8.5, "heritage"),
            Attraction("Marine Drive",         0,  1.5, 8.0, "nature"),
            Attraction("Elephanta Caves",    600,  3.0, 8.5, "heritage"),
            Attraction("Juhu Beach",           0,  2.0, 7.5, "nature"),
        ]
    ),
    "Goa": City(
        name="Goa", state="Goa",
        hotel_cost=4000, latitude=15.2993, longitude=74.1240,
        attractions=[
            Attraction("Baga Beach",          0,  3.0, 9.0, "nature"),
            Attraction("Basilica of Bom Jesus", 0, 1.5, 8.5, "heritage"),
            Attraction("Dudhsagar Falls",    400,  4.0, 9.5, "nature"),
            Attraction("Anjuna Flea Market",   0,  2.0, 7.5, "food"),
        ]
    ),
    "Hyderabad": City(
        name="Hyderabad", state="Telangana",
        hotel_cost=2500, latitude=17.3850, longitude=78.4867,
        attractions=[
            Attraction("Charminar",           25,  1.5, 9.0, "heritage"),
            Attraction("Golconda Fort",       100,  2.5, 8.5, "heritage"),
            Attraction("Ramoji Film City",   1500,  6.0, 9.0, "adventure"),
            Attraction("Hussain Sagar Lake",    0,  1.5, 7.5, "nature"),
        ]
    ),
    "Chennai": City(
        name="Chennai", state="Tamil Nadu",
        hotel_cost=2800, latitude=13.0827, longitude=80.2707,
        attractions=[
            Attraction("Marina Beach",        0,  2.0, 8.5, "nature"),
            Attraction("Kapaleeshwarar Temple", 0, 1.5, 8.0, "heritage"),
            Attraction("Mahabalipuram",       50,  3.0, 9.0, "heritage"),
        ]
    ),
    "Kolkata": City(
        name="Kolkata", state="West Bengal",
        hotel_cost=2500, latitude=22.5726, longitude=88.3639,
        attractions=[
            Attraction("Victoria Memorial",  30,  2.0, 9.5, "heritage"),
            Attraction("Howrah Bridge",        0,  1.0, 8.0, "heritage"),
            Attraction("Sundarbans",        2500,  8.0, 9.5, "nature"),
        ]
    ),
    "Varanasi": City(
        name="Varanasi", state="Uttar Pradesh",
        hotel_cost=2000, latitude=25.3176, longitude=82.9739,
        attractions=[
            Attraction("Dashashwamedh Ghat",  0,  2.0, 10.0, "heritage"),
            Attraction("Kashi Vishwanath",      0,  1.5,  9.5, "heritage"),
            Attraction("Sarnath",             20,  2.0,  8.5, "heritage"),
        ]
    ),
    "Shimla": City(
        name="Shimla", state="Himachal Pradesh",
        hotel_cost=3000, latitude=31.1048, longitude=77.1734,
        attractions=[
            Attraction("The Ridge",            0,  2.0, 8.5, "nature"),
            Attraction("Jakhu Temple",          0,  2.0, 8.0, "heritage"),
            Attraction("Kufri",              500,  3.0, 9.0, "adventure"),
        ]
    ),
    "Manali": City(
        name="Manali", state="Himachal Pradesh",
        hotel_cost=3200, latitude=32.2396, longitude=77.1887,
        attractions=[
            Attraction("Rohtang Pass",       550,  4.0, 9.5, "adventure"),
            Attraction("Solang Valley",      500,  3.0, 9.5, "adventure"),
            Attraction("Hadimba Temple",        0,  1.5, 8.0, "heritage"),
        ]
    ),
}

# ── Roads (bidirectional graph edges) ────────────────────────
#    (from, to, distance_km, cost_INR, time_hours)

ROADS: List[Road] = [
    Road("Delhi",     "Agra",       200,  300, 3.0),
    Road("Delhi",     "Jaipur",     270,  400, 4.5),
    Road("Delhi",     "Varanasi",   820, 1200, 12.0),
    Road("Delhi",     "Kolkata",   1500, 2000, 20.0),
    Road("Delhi",     "Shimla",     350,  500,  7.0),
    Road("Agra",      "Jaipur",     230,  350,  4.0),
    Road("Agra",      "Varanasi",   630,  900, 10.0),
    Road("Jaipur",    "Udaipur",    400,  600,  6.5),
    Road("Udaipur",   "Mumbai",     650,  900, 10.0),
    Road("Mumbai",    "Goa",        590,  800,  9.0),
    Road("Mumbai",    "Hyderabad",  710, 1000, 11.0),
    Road("Goa",       "Hyderabad",  600,  850,  9.5),
    Road("Hyderabad", "Chennai",    630,  900, 10.0),
    Road("Hyderabad", "Kolkata",   1500, 1800, 20.0),
    Road("Chennai",   "Kolkata",   1660, 2000, 22.0),
    Road("Varanasi",  "Kolkata",    670, 1000, 10.0),
    Road("Shimla",    "Manali",     270,  400,  6.0),
    Road("Delhi",     "Manali",     540,  750, 12.0),
    Road("Jaipur",    "Varanasi",   640,  900, 10.0),
    Road("Mumbai",    "Chennai",   1340, 1500, 18.0),
]

# ── Weather / Crowd probabilities (for Bayesian module) ──────
#    P(good_weather), P(heavy_crowd), P(traffic_delay)

CITY_PROBS: Dict[str, Dict[str, float]] = {
    "Delhi":     {"good_weather": 0.55, "heavy_crowd": 0.80, "traffic": 0.75},
    "Agra":      {"good_weather": 0.60, "heavy_crowd": 0.85, "traffic": 0.60},
    "Jaipur":    {"good_weather": 0.65, "heavy_crowd": 0.70, "traffic": 0.55},
    "Udaipur":   {"good_weather": 0.70, "heavy_crowd": 0.50, "traffic": 0.40},
    "Mumbai":    {"good_weather": 0.50, "heavy_crowd": 0.90, "traffic": 0.85},
    "Goa":       {"good_weather": 0.75, "heavy_crowd": 0.65, "traffic": 0.40},
    "Hyderabad": {"good_weather": 0.65, "heavy_crowd": 0.70, "traffic": 0.65},
    "Chennai":   {"good_weather": 0.55, "heavy_crowd": 0.65, "traffic": 0.60},
    "Kolkata":   {"good_weather": 0.55, "heavy_crowd": 0.80, "traffic": 0.70},
    "Varanasi":  {"good_weather": 0.60, "heavy_crowd": 0.75, "traffic": 0.50},
    "Shimla":    {"good_weather": 0.70, "heavy_crowd": 0.55, "traffic": 0.45},
    "Manali":    {"good_weather": 0.65, "heavy_crowd": 0.50, "traffic": 0.35},
}

def get_city_list() -> List[str]:
    """Return sorted list of all city names."""
    return sorted(CITIES.keys())

def get_adjacency() -> Dict[str, List[Tuple[str, float, float, float]]]:
    """
    Build adjacency list: city -> [(neighbour, distance, cost, time), ...]
    Both directions since roads are bidirectional.
    """
    adj: Dict[str, List[Tuple[str, float, float, float]]] = {c: [] for c in CITIES}
    for r in ROADS:
        adj[r.from_city].append((r.to_city, r.distance, r.travel_cost, r.travel_time))
        adj[r.to_city].append((r.from_city, r.distance, r.travel_cost, r.travel_time))
    return adj
