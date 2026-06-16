# ============================================================
# csp_module.py – Constraint Satisfaction Problem (CO3)
# Implements: Backtracking, Forward Checking, MRV, LCV, Degree,
#             Min-Conflicts, and daily itinerary scheduler.
# ============================================================

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import copy

from graph import TouristGraph
from dataset import CITIES, Attraction
from utils import (C, banner, sub_banner, info, ai_reason, warn,
                   error, separator, success)


# ════════════════════════════════════════════════════════════
#  CONSTRAINT DEFINITIONS
# ════════════════════════════════════════════════════════════

@dataclass
class TripConstraints:
    total_budget:        float = 20_000.0   # INR
    total_days:          int   = 5
    hotel_preference:    str   = "any"      # "budget" / "luxury" / "any"
    transport:           str   = "any"      # "bus" / "train" / "any"
    max_attractions_day: int   = 4
    max_travel_time_day: float = 6.0        # hours of travel per day
    must_visit:          List[str] = field(default_factory=list)
    avoid_cities:        List[str] = field(default_factory=list)


# ════════════════════════════════════════════════════════════
#  DAILY ITINERARY ITEM
# ════════════════════════════════════════════════════════════

@dataclass
class DayPlan:
    day:          int
    city:         str
    hotel_cost:   float
    attractions:  List[Attraction]
    travel_from:  Optional[str] = None
    travel_cost:  float = 0.0
    travel_time:  float = 0.0

    def total_cost(self) -> float:
        att_cost = sum(a.entry_fee for a in self.attractions)
        return self.hotel_cost + att_cost + self.travel_cost

    def total_time(self) -> float:
        att_time = sum(a.time_required for a in self.attractions)
        return att_time + self.travel_time

    def satisfaction(self) -> float:
        return sum(a.satisfaction for a in self.attractions)


# ════════════════════════════════════════════════════════════
#  CSP SOLVER
# ════════════════════════════════════════════════════════════

class CSPSolver:
    """
    Models trip planning as a CSP:
      Variables  : days (Day 1 … Day N)
      Domains    : set of city assignments for each day
      Constraints: budget, time, attraction count, preferences

    Algorithms:
      - Backtracking Search
      - Forward Checking (domain pruning)
      - MRV  (Minimum Remaining Values)
      - Degree heuristic
      - LCV  (Least Constraining Value)
      - Min-Conflicts local search
    """

    def __init__(self, graph: TouristGraph):
        self.graph = graph

    # ────────────────────────────────────────────────────────
    #  CONSTRAINT CHECK
    # ────────────────────────────────────────────────────────

    def _check_city_constraint(
        self,
        city: str,
        day_idx: int,
        assigned: Dict[int, str],
        constraints: TripConstraints,
        budget_used: float,
        time_used: float,
    ) -> Tuple[bool, str]:
        """
        Returns (is_valid, reason_string).
        Checks each active constraint and explains failures.
        """
        city_obj = self.graph.cities.get(city)
        if city_obj is None:
            return False, f"City {city!r} not in dataset"

        if city in constraints.avoid_cities:
            return False, f"'{city}' is in the avoid list"

        # Hotel preference
        if constraints.hotel_preference == "budget" and city_obj.hotel_cost > 3000:
            return False, f"Hotel in {city} (₹{city_obj.hotel_cost}) exceeds budget preference"
        if constraints.hotel_preference == "luxury" and city_obj.hotel_cost < 3000:
            return False, f"Hotel in {city} (₹{city_obj.hotel_cost}) below luxury preference"

        # Budget check (hotel only; attractions checked per-day)
        if budget_used + city_obj.hotel_cost > constraints.total_budget:
            return False, (f"Budget exceeded: hotel ₹{city_obj.hotel_cost} + "
                           f"spent ₹{budget_used:,.0f} > limit ₹{constraints.total_budget:,.0f}")

        # No repeat cities (simple constraint)
        if city in assigned.values():
            return False, f"'{city}' already assigned to another day"

        return True, "OK"

    # ────────────────────────────────────────────────────────
    #  FORWARD CHECKING – Prune domains after assignment
    # ────────────────────────────────────────────────────────

    def _forward_check(
        self,
        domains: Dict[int, List[str]],
        assigned_city: str,
        current_day: int,
        constraints: TripConstraints,
        budget_remaining: float,
    ) -> bool:
        """
        After assigning `assigned_city` to `current_day`,
        remove that city from remaining days' domains.
        Return False if any domain becomes empty (dead end).
        """
        for day in list(domains.keys()):
            if day <= current_day:
                continue
            # Remove already assigned city
            if assigned_city in domains[day]:
                domains[day].remove(assigned_city)
            # Prune cities too expensive
            domains[day] = [
                c for c in domains[day]
                if self.graph.cities[c].hotel_cost <= budget_remaining
            ]
            if not domains[day]:
                ai_reason("FC-PRUNE",
                          f"Day {day} domain empty after assigning {assigned_city}")
                return False
        return True

    # ────────────────────────────────────────────────────────
    #  MRV – Select variable (day) with fewest remaining values
    # ────────────────────────────────────────────────────────

    def _mrv_select_day(self, domains: Dict[int, List[str]], assigned: Dict[int, str]) -> int:
        """
        MRV: pick the unassigned day whose domain is smallest.
        Tie-break: Degree heuristic (day with most constraints on others).
        """
        unassigned = {d: v for d, v in domains.items() if d not in assigned}
        # MRV: minimum domain size
        min_size = min(len(v) for v in unassigned.values())
        candidates = [d for d, v in unassigned.items() if len(v) == min_size]
        # Degree heuristic tie-break: higher day index = more days constrained
        return max(candidates)

    # ────────────────────────────────────────────────────────
    #  LCV – Order values (cities) to be tried first
    # ────────────────────────────────────────────────────────

    def _lcv_order(self, day: int, domain: List[str], domains: Dict[int, List[str]]) -> List[str]:
        """
        LCV: try values that rule out the fewest choices for other days.
        Score = number of future domains that still contain the city.
        Lower score = less constraining = prefer first.
        """
        def score(city: str) -> int:
            count = 0
            for d, vals in domains.items():
                if d != day and city in vals:
                    count += 1
            return count

        return sorted(domain, key=score)

    # ────────────────────────────────────────────────────────
    #  BACKTRACKING SEARCH
    # ────────────────────────────────────────────────────────

    def backtrack(
        self,
        constraints: TripConstraints,
        cities_to_visit: List[str],
    ) -> Optional[Dict[int, str]]:
        """
        Backtracking with Forward Checking + MRV + LCV.
        Returns a day→city assignment or None if unsatisfiable.
        """
        ai_reason("CSP", "Starting backtracking search …")
        # Build initial domains
        available = [c for c in cities_to_visit
                     if c not in constraints.avoid_cities]
        domains: Dict[int, List[str]] = {
            d: list(available) for d in range(1, constraints.total_days + 1)
        }
        assigned: Dict[int, str] = {}
        result = self._bt(domains, assigned, constraints,
                          budget_used=0.0, time_used=0.0)
        return result

    def _bt(
        self,
        domains:     Dict[int, List[str]],
        assigned:    Dict[int, str],
        constraints: TripConstraints,
        budget_used: float,
        time_used:   float,
    ) -> Optional[Dict[int, str]]:

        if len(assigned) == constraints.total_days:
            return dict(assigned)

        # MRV: pick next day
        day = self._mrv_select_day(domains, assigned)
        ai_reason("MRV", f"Selected Day {day} (domain size {len(domains[day])})")

        # LCV: order cities to try
        ordered = self._lcv_order(day, domains[day], domains)

        for city in ordered:
            valid, reason = self._check_city_constraint(
                city, day, assigned, constraints, budget_used, time_used)

            if not valid:
                ai_reason("BACKTRACK", f"Day {day} → {city} ✗ [{reason}]")
                continue

            # Assign
            assigned[day] = city
            city_cost = self.graph.cities[city].hotel_cost
            new_domains = copy.deepcopy(domains)

            # Forward checking
            ok = self._forward_check(new_domains, city, day, constraints,
                                     constraints.total_budget - budget_used - city_cost)
            if not ok:
                ai_reason("FC", f"Forward check failed for Day {day} → {city}")
                del assigned[day]
                continue

            ai_reason("ASSIGN", f"Day {day} → {city} ✓")
            result = self._bt(new_domains, assigned, constraints,
                              budget_used + city_cost, time_used)
            if result is not None:
                return result

            # Undo
            del assigned[day]
            ai_reason("UNDO", f"Undoing Day {day} → {city}")

        return None

    # ────────────────────────────────────────────────────────
    #  MIN-CONFLICTS LOCAL SEARCH
    # ────────────────────────────────────────────────────────

    def min_conflicts(
        self,
        constraints: TripConstraints,
        cities: List[str],
        max_steps: int = 200,
    ) -> Optional[Dict[int, str]]:
        """
        Min-Conflicts: start with a random assignment and
        iteratively fix the most-conflicted variable.
        """
        import random
        ai_reason("MIN-CONFLICTS", "Starting local search …")

        available = [c for c in cities if c not in constraints.avoid_cities]
        if len(available) < constraints.total_days:
            warn("Not enough cities for all days.")
            return None

        # Random initial assignment
        sample = random.sample(available, constraints.total_days)
        assignment: Dict[int, str] = {d + 1: sample[d]
                                       for d in range(constraints.total_days)}

        for step in range(max_steps):
            conflicts = self._find_conflicted(assignment, constraints)
            if not conflicts:
                ai_reason("MIN-CONFLICTS", f"Solution found at step {step}")
                return assignment

            # Pick a conflicted day
            day = random.choice(conflicts)
            # Choose city that minimises conflicts
            best_city = min(
                available,
                key=lambda c: self._count_conflicts(
                    {**assignment, day: c}, constraints)
            )
            assignment[day] = best_city

        warn("Min-Conflicts: max steps reached; returning best found.")
        return assignment

    def _find_conflicted(self, assignment: Dict[int, str], constraints: TripConstraints) -> List[int]:
        conflicts = []
        total_hotel = sum(self.graph.cities[c].hotel_cost
                          for c in assignment.values()
                          if c in self.graph.cities)
        for day, city in assignment.items():
            if city in constraints.avoid_cities:
                conflicts.append(day)
            elif total_hotel > constraints.total_budget:
                conflicts.append(day)
        return list(set(conflicts))

    def _count_conflicts(self, assignment: Dict[int, str], constraints: TripConstraints) -> int:
        total_hotel = sum(self.graph.cities[c].hotel_cost
                          for c in assignment.values()
                          if c in self.graph.cities)
        cnt = 0
        seen = set()
        for day, city in assignment.items():
            if city in constraints.avoid_cities:
                cnt += 1
            if city in seen:
                cnt += 1
            seen.add(city)
        if total_hotel > constraints.total_budget:
            cnt += 1
        return cnt

    # ────────────────────────────────────────────────────────
    #  ITINERARY SCHEDULER
    # ────────────────────────────────────────────────────────

    def build_itinerary(
        self,
        assignment: Dict[int, str],
        constraints: TripConstraints,
        path: List[str],
    ) -> List[DayPlan]:
        """
        Given a day→city assignment and a travel path,
        create a detailed DayPlan for each day.
        """
        itinerary: List[DayPlan] = []
        budget_left = constraints.total_budget

        # Compute travel costs between consecutive cities in path
        travel_info: Dict[Tuple[str, str], Tuple[float, float]] = {}
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            c = self.graph.edge_cost(a, b) or 0
            t = self.graph.edge_time(a, b) or 0
            travel_info[(a, b)] = (c, t)

        prev_city: Optional[str] = None
        for day in sorted(assignment.keys()):
            city = assignment[day]
            city_obj = self.graph.cities.get(city)
            if city_obj is None:
                continue

            hotel_cost   = city_obj.hotel_cost
            travel_cost  = 0.0
            travel_time  = 0.0

            if prev_city and prev_city != city:
                tc, tt = travel_info.get((prev_city, city), (500.0, 4.0))
                travel_cost = tc
                travel_time = tt

            # Select attractions within remaining budget and time
            att_budget = budget_left - hotel_cost - travel_cost
            att_time   = max(0, (24 - travel_time - 8))  # 8h sleep
            chosen_atts = []
            for att in sorted(city_obj.attractions,
                               key=lambda a: a.satisfaction, reverse=True):
                if (len(chosen_atts) < constraints.max_attractions_day
                        and att.entry_fee <= att_budget
                        and att.time_required <= att_time):
                    chosen_atts.append(att)
                    att_budget -= att.entry_fee
                    att_time   -= att.time_required

            day_plan = DayPlan(
                day=day, city=city,
                hotel_cost=hotel_cost,
                attractions=chosen_atts,
                travel_from=prev_city,
                travel_cost=travel_cost,
                travel_time=travel_time,
            )
            itinerary.append(day_plan)
            budget_left -= day_plan.total_cost()
            prev_city = city

        return itinerary

    # ────────────────────────────────────────────────────────
    #  DISPLAY
    # ────────────────────────────────────────────────────────

    def display_itinerary(self, itinerary: List[DayPlan], constraints: TripConstraints) -> None:
        banner("CSP TRIP ITINERARY", 65)
        total_cost = 0.0
        total_sat  = 0.0
        for dp in itinerary:
            sub_banner(f"Day {dp.day} – {dp.city}")
            if dp.travel_from:
                info(f"Travel from : {dp.travel_from} → {dp.city}  "
                     f"(₹{dp.travel_cost:,.0f}, {dp.travel_time:.1f}h)")
            info(f"Hotel       : ₹{dp.hotel_cost:,.0f}/night")
            if dp.attractions:
                info("Attractions :")
                for att in dp.attractions:
                    print(f"    {C.YELLOW}•{C.RESET} {att.name:<32} "
                          f"₹{att.entry_fee:>6.0f}  {att.time_required}h  "
                          f"⭐ {att.satisfaction}")
            else:
                warn("No attractions scheduled (budget/time too tight).")
            day_cost = dp.total_cost()
            info(f"Day total   : ₹{day_cost:,.0f}")
            total_cost += day_cost
            total_sat  += dp.satisfaction()

        separator()
        info(f"Total Trip Cost  : ₹{total_cost:,.0f}  (Budget: ₹{constraints.total_budget:,.0f})")
        info(f"Budget Remaining : ₹{constraints.total_budget - total_cost:,.0f}")
        info(f"Satisfaction     : {total_sat:.1f} points")
        separator()

    # ────────────────────────────────────────────────────────
    #  Full CSP Run (entry point called from main menu)
    # ────────────────────────────────────────────────────────

    def run(
        self,
        path: List[str],
        constraints: TripConstraints,
        use_min_conflicts: bool = False,
    ) -> None:
        banner("CSP TRIP PLANNER", 65)
        ai_reason("CSP", f"Cities: {path}, Budget: ₹{constraints.total_budget:,.0f}, "
                         f"Days: {constraints.total_days}")

        if use_min_conflicts:
            sub_banner("Min-Conflicts Local Search")
            assignment = self.min_conflicts(constraints, path)
        else:
            sub_banner("Backtracking + Forward Checking + MRV + LCV")
            assignment = self.backtrack(constraints, path)

        if assignment is None:
            error("CSP: No valid trip plan found. Try relaxing constraints.")
            return

        success(f"CSP solved! Assignment: {assignment}")
        itinerary = self.build_itinerary(assignment, constraints, path)
        self.display_itinerary(itinerary, constraints)
