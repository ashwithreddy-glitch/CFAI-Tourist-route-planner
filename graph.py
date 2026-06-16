# ============================================================
# graph.py – Environment / Graph representation (CO1)
# Encapsulates cities, roads, adjacency list, and heuristics.
# ============================================================

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math

from dataset import CITIES, get_adjacency, City, Attraction
from utils import haversine, C, banner, sub_banner, info, separator

# ── Node used during search ──────────────────────────────────

@dataclass(order=True)
class SearchNode:
    """
    Represents a state in the search space.
    'order=True' lets heapq compare nodes by f_cost automatically.
    """
    f_cost:   float        # priority  (used by heapq)
    g_cost:   float        # cost so far (distance / time / money)
    city:     str          # current city name
    path:     List[str]    # cities visited so far
    path_cost_detail: List[float] = field(default_factory=list)  # edge costs

    # Make dataclass hashable for closed-list sets
    def __hash__(self):
        return hash((self.city, tuple(self.path)))

    def __eq__(self, other):
        return self.city == other.city and self.path == other.path


# ── Main Graph class ─────────────────────────────────────────

class TouristGraph:
    """
    Knowledge representation of the tourist environment.

    Internally stores:
      - adjacency list  (graph)
      - city metadata   (hotels, attractions, coordinates)
    """

    def __init__(self):
        self.cities: Dict[str, City] = CITIES
        # adj[city] = [(neighbour, distance_km, cost_INR, time_hr), ...]
        self.adj: Dict[str, List[Tuple[str, float, float, float]]] = get_adjacency()

    # ── Graph queries ────────────────────────────────────────

    def neighbours(self, city: str) -> List[Tuple[str, float, float, float]]:
        """Return list of (neighbour, distance, cost, time)."""
        return self.adj.get(city, [])

    def edge_distance(self, a: str, b: str) -> Optional[float]:
        """Direct road distance between two adjacent cities (km)."""
        for nb, dist, *_ in self.adj.get(a, []):
            if nb == b:
                return dist
        return None

    def edge_cost(self, a: str, b: str) -> Optional[float]:
        """Travel cost INR between two adjacent cities."""
        for nb, _, cost, *_ in self.adj.get(a, []):
            if nb == b:
                return cost
        return None

    def edge_time(self, a: str, b: str) -> Optional[float]:
        """Travel time hours between two adjacent cities."""
        for nb, _, __, time in self.adj.get(a, []):
            if nb == b:
                return time
        return None

    # ── Heuristic ────────────────────────────────────────────

    def heuristic(self, city: str, goal: str) -> float:
        """
        Admissible heuristic: straight-line (Haversine) distance.
        Never overestimates → valid for A*.
        """
        c1 = self.cities.get(city)
        c2 = self.cities.get(goal)
        if c1 is None or c2 is None:
            return 0.0
        return haversine(c1.latitude, c1.longitude, c2.latitude, c2.longitude)

    # ── Convenience ─────────────────────────────────────────

    def total_distance(self, path: List[str]) -> float:
        """Sum road distances along a path. Returns ∞ if broken."""
        total = 0.0
        for i in range(len(path) - 1):
            d = self.edge_distance(path[i], path[i + 1])
            if d is None:
                return math.inf
            total += d
        return total

    def total_cost(self, path: List[str]) -> float:
        """Sum travel costs along a path."""
        total = 0.0
        for i in range(len(path) - 1):
            c = self.edge_cost(path[i], path[i + 1])
            if c is None:
                return math.inf
            total += c
        return total

    def total_time(self, path: List[str]) -> float:
        """Sum travel time along a path."""
        total = 0.0
        for i in range(len(path) - 1):
            t = self.edge_time(path[i], path[i + 1])
            if t is None:
                return math.inf
            total += t
        return total

    def attraction_score(self, path: List[str]) -> float:
        """Sum satisfaction scores of all attractions in visited cities."""
        score = 0.0
        for city in path:
            city_obj = self.cities.get(city)
            if city_obj:
                for att in city_obj.attractions:
                    score += att.satisfaction
        return score

    # ── Display ──────────────────────────────────────────────

    def display_cities(self) -> None:
        banner("TOURIST LOCATIONS", 60)
        for name, city in sorted(self.cities.items()):
            print(f"\n  {C.BOLD}{C.GREEN}{name}{C.RESET} ({city.state})")
            print(f"    Hotel/night : ₹{city.hotel_cost:,.0f}")
            print(f"    Coordinates : {city.latitude:.2f}°N, {city.longitude:.2f}°E")
            print(f"    {C.YELLOW}Attractions:{C.RESET}")
            for att in city.attractions:
                print(f"      • {att.name:<30} Fee: ₹{att.entry_fee:>6.0f}  "
                      f"Time: {att.time_required}h  Score: {att.satisfaction}/10  [{att.category}]")
        separator()

    def display_graph_summary(self) -> None:
        sub_banner("Graph Summary")
        info(f"Cities  : {len(self.cities)}")
        total_edges = sum(len(v) for v in self.adj.values()) // 2
        info(f"Roads   : {total_edges}")
        separator()
