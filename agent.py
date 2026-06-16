# ============================================================
# agent.py – Intelligent Agent definition (CO1)
# PEAS description + Agent class + state/action/transition model
# ============================================================

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional
import logging

from graph import TouristGraph
from utils import C, banner, sub_banner, info, ai_reason, separator, warn

logger = logging.getLogger("TouristPlanner.Agent")

# ════════════════════════════════════════════════════════════
#  PEAS DESCRIPTION
# ════════════════════════════════════════════════════════════
PEAS = {
    "Performance Measure": [
        "Maximum tourist satisfaction score",
        "Minimum travel cost (INR)",
        "Minimum travel time (hours)",
        "Maximum attractions visited",
        "Stay within budget and time constraints",
    ],
    "Environment": [
        "Cities and tourist spots across India",
        "Road network with distances and costs",
        "Weather & crowd conditions (stochastic)",
        "Hotel availability and pricing",
        "Tourist budgets and schedule constraints",
    ],
    "Actuators": [
        "Move to a new city",
        "Select / visit an attraction",
        "Book a hotel",
        "Recalculate route dynamically",
        "Generate daily itinerary",
    ],
    "Sensors": [
        "Current location (GPS / input)",
        "Visited city/attraction history",
        "Budget remaining",
        "Time remaining",
        "Weather reports (Bayesian estimates)",
        "Traffic / crowd levels",
    ],
}

# ════════════════════════════════════════════════════════════
#  STATE REPRESENTATION
# ════════════════════════════════════════════════════════════

@dataclass
class AgentState:
    """
    Complete state of the tourist agent at any point in time.
    Used as the node-state in search algorithms.
    """
    current_location: str
    visited: Set[str]             = field(default_factory=set)
    remaining_budget:  float      = 50_000.0    # INR
    remaining_time:    float      = 168.0       # hours (1 week)
    satisfaction:      float      = 0.0
    path:              List[str]  = field(default_factory=list)
    travel_cost_so_far: float     = 0.0
    travel_time_so_far: float     = 0.0

    def __post_init__(self):
        if not self.path:
            self.path = [self.current_location]
        self.visited.add(self.current_location)

    def clone(self) -> "AgentState":
        return AgentState(
            current_location   = self.current_location,
            visited            = set(self.visited),
            remaining_budget   = self.remaining_budget,
            remaining_time     = self.remaining_time,
            satisfaction       = self.satisfaction,
            path               = list(self.path),
            travel_cost_so_far = self.travel_cost_so_far,
            travel_time_so_far = self.travel_time_so_far,
        )

# ════════════════════════════════════════════════════════════
#  AGENT CLASS
# ════════════════════════════════════════════════════════════

class TouristAgent:
    """
    Rational Agent that plans tourist routes.

    Architecture: Goal-based + Utility-based hybrid.
    The agent maintains a state, applies actions via the
    transition model, and logs its reasoning steps.
    """

    def __init__(self, graph: TouristGraph):
        self.graph  = graph
        self.state: Optional[AgentState] = None
        self._log: List[str] = []

    # ── Initialise / reset ───────────────────────────────────

    def initialise(self, start: str, budget: float, time_limit: float) -> AgentState:
        """Create initial agent state."""
        self.state = AgentState(
            current_location = start,
            remaining_budget = budget,
            remaining_time   = time_limit,
        )
        self._log.clear()
        self._log.append(f"Agent initialised at {start} | Budget ₹{budget:,.0f} | Time {time_limit}h")
        ai_reason("INIT", f"Starting at {C.BOLD}{start}{C.RESET}, budget ₹{budget:,.0f}, time {time_limit}h")
        return self.state

    # ── Actions ──────────────────────────────────────────────

    def action_move(self, destination: str) -> bool:
        """
        Transition model: move to a neighbouring city.
        Returns True if the move is valid and applied.
        """
        if self.state is None:
            warn("Agent not initialised.")
            return False

        neighbours = self.graph.neighbours(self.state.current_location)
        nb_map = {n: (d, c, t) for n, d, c, t in neighbours}

        if destination not in nb_map:
            ai_reason("MOVE_FAIL", f"{destination} is not reachable from {self.state.current_location}")
            return False

        dist, cost, time_h = nb_map[destination]

        # Check constraints
        if cost > self.state.remaining_budget:
            ai_reason("CONSTRAINT", f"Budget exceeded: need ₹{cost:,.0f}, have ₹{self.state.remaining_budget:,.0f}")
            return False
        if time_h > self.state.remaining_time:
            ai_reason("CONSTRAINT", f"Time exceeded: need {time_h}h, have {self.state.remaining_time:.1f}h")
            return False

        # Apply transition
        self.state.current_location    = destination
        self.state.visited.add(destination)
        self.state.path.append(destination)
        self.state.remaining_budget   -= cost
        self.state.remaining_time     -= time_h
        self.state.travel_cost_so_far += cost
        self.state.travel_time_so_far += time_h

        log_msg = (f"Moved {self.state.path[-2]} → {destination} | "
                   f"dist {dist:.0f}km cost ₹{cost:.0f} time {time_h}h")
        self._log.append(log_msg)
        ai_reason("MOVE", log_msg)
        return True

    def action_visit_attraction(self, attraction_name: str) -> bool:
        """Visit an attraction in the current city."""
        if self.state is None:
            return False
        city_obj = self.graph.cities.get(self.state.current_location)
        if city_obj is None:
            return False
        for att in city_obj.attractions:
            if att.name.lower() == attraction_name.lower():
                if att.entry_fee > self.state.remaining_budget:
                    ai_reason("CONSTRAINT", f"Cannot afford {att.name} (₹{att.entry_fee})")
                    return False
                if att.time_required > self.state.remaining_time:
                    ai_reason("CONSTRAINT", f"Not enough time for {att.name} ({att.time_required}h)")
                    return False
                self.state.remaining_budget -= att.entry_fee
                self.state.remaining_time   -= att.time_required
                self.state.satisfaction     += att.satisfaction
                ai_reason("VISIT", f"Visited {att.name} | +{att.satisfaction} satisfaction")
                return True
        return False

    def action_recalculate_route(self, destination: str) -> List[str]:
        """
        Trigger a new A* search from current location.
        Returns recommended path.
        """
        from search_algorithms import SearchAlgorithms
        ai_reason("RECALC", f"Recalculating route to {destination}")
        sa = SearchAlgorithms(self.graph)
        result = sa.astar(self.state.current_location, destination)
        return result.path if result else []

    # ── Path cost model ──────────────────────────────────────

    def path_cost(self, path: List[str]) -> Dict[str, float]:
        """Compute total distance, cost, time for a given path."""
        return {
            "distance_km": self.graph.total_distance(path),
            "cost_INR":    self.graph.total_cost(path),
            "time_hours":  self.graph.total_time(path),
        }

    # ── Rule-based recommendations ───────────────────────────

    def recommend_attractions(self, city: str, budget: float, time_available: float):
        """
        Rule-based filter: suggest attractions within budget & time.
        Returns sorted list of Attraction objects.
        """
        city_obj = self.graph.cities.get(city)
        if not city_obj:
            return []
        eligible = [
            a for a in city_obj.attractions
            if a.entry_fee <= budget and a.time_required <= time_available
        ]
        # Sort by satisfaction descending (greedy rule)
        return sorted(eligible, key=lambda a: a.satisfaction, reverse=True)

    # ── Display PEAS ─────────────────────────────────────────

    def display_peas(self) -> None:
        banner("PEAS DESCRIPTION – Tourist Route Planner Agent", 65)
        for category, items in PEAS.items():
            sub_banner(category)
            for item in items:
                info(item)
        separator()

    def display_log(self) -> None:
        sub_banner("Agent Reasoning Log")
        for i, entry in enumerate(self._log, 1):
            print(f"  {C.MAGENTA}{i:>2}.{C.RESET} {entry}")
        separator()

    def display_state(self) -> None:
        if not self.state:
            warn("No active state.")
            return
        sub_banner("Current Agent State")
        info(f"Location  : {self.state.current_location}")
        info(f"Visited   : {', '.join(sorted(self.state.visited))}")
        info(f"Budget    : ₹{self.state.remaining_budget:,.0f} remaining")
        info(f"Time left : {self.state.remaining_time:.1f}h")
        info(f"Satisfaction: {self.state.satisfaction:.1f}")
        info(f"Path so far : {' → '.join(self.state.path)}")
        separator()
