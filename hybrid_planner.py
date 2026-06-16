# ============================================================
# hybrid_planner.py – CO6: Hybrid Intelligent Planner
# Integrates Search + CSP + Bayes + Decision into one pipeline.
# Includes: XAI traces, Performance Analysis, Ethics section.
# ============================================================

from __future__ import annotations
import time
from typing import List, Optional, Dict

from graph import TouristGraph
from search_algorithms import SearchAlgorithms, SearchResult
from csp_module import CSPSolver, TripConstraints
from bayesian_module import BayesianModule
from decision_module import DecisionModule
from utils import (C, banner, sub_banner, info, ai_reason, separator,
                   warn, success, format_route, format_time)


class HybridPlanner:
    """
    CO6 – Combines all AI modules in a single end-to-end workflow:

    1. Search    → find candidate routes
    2. CSP       → filter by constraints, build itinerary
    3. Bayes     → estimate uncertainty / risk
    4. Decision  → select optimal plan via Alpha-Beta
    5. Output    → final itinerary with XAI reasoning traces
    """

    def __init__(self, graph: TouristGraph):
        self.graph    = graph
        self.search   = SearchAlgorithms(graph)
        self.csp      = CSPSolver(graph)
        self.bayes    = BayesianModule()
        self.decision = DecisionModule(graph)
        self._trace:  List[str] = []

    # ── XAI trace helper ─────────────────────────────────────

    def _t(self, step: str, msg: str) -> None:
        entry = f"[{step}] {msg}"
        self._trace.append(entry)
        ai_reason(step, msg)

    # ════════════════════════════════════════════════════════
    #  MAIN PIPELINE
    # ════════════════════════════════════════════════════════

    def run(
        self,
        source:      str,
        destination: str,
        constraints: TripConstraints,
        season:      str = "winter",
    ) -> None:
        self._trace.clear()
        banner("HYBRID INTELLIGENT PLANNER  (CO6)", 65)

        # ─── STEP 1: Search ──────────────────────────────────
        sub_banner("Step 1 – Search Algorithms")
        self._t("SEARCH", f"Finding routes from {source} to {destination}")

        t_start = time.perf_counter()
        results: Dict[str, SearchResult] = {}

        for algo, fn in [("BFS", self.search.bfs),
                         ("A*",  self.search.astar),
                         ("UCS", self.search.ucs)]:
            r = fn(source, destination)
            results[algo] = r
            status = "✔" if r.found else "✘"
            self._t(algo, f"{status} path={r.path} dist={r.total_distance:.0f}km "
                          f"nodes={r.nodes_expanded} time={r.exec_time_ms:.2f}ms")

        # Collect valid paths
        valid_paths = [r.path for r in results.values() if r.found and r.path]
        if not valid_paths:
            warn("No route found between selected cities. Try a different pair.")
            return

        # ─── STEP 2: CSP Constraint Filtering ────────────────
        sub_banner("Step 2 – CSP Constraint Check")
        self._t("CSP", f"Checking constraints: budget ₹{constraints.total_budget:,.0f}, "
                       f"days {constraints.total_days}, hotel={constraints.hotel_preference}")

        # Build city list from best A* path
        best_path = results.get("A*", results.get("UCS")).path
        if not best_path:
            best_path = valid_paths[0]

        # Extend path with intermediate cities for CSP
        csp_cities = list(dict.fromkeys(best_path))  # unique, ordered

        assignment = self.csp.backtrack(constraints, csp_cities)
        if assignment is None:
            self._t("CSP", "No valid assignment found; relaxing to min-conflicts …")
            assignment = self.csp.min_conflicts(constraints, csp_cities)

        if assignment is None:
            self._t("CSP", "FAIL – constraints unsatisfiable with this route")
        else:
            self._t("CSP", f"Assignment: {assignment}")

        # ─── STEP 3: Bayesian Risk Assessment ────────────────
        sub_banner("Step 3 – Bayesian Uncertainty Estimation")
        self._t("BAYES", f"Estimating risk for season={season}")

        risk_scores: Dict[str, Dict] = {}
        for city in best_path:
            pred = self.bayes.predict_city(city, season)
            risk_scores[city] = pred
            self._t("BAYES",
                    f"{city}: P(success)={pred['P(trip_success)']:.2f} "
                    f"P(good_weather)={pred['P(good_weather)']:.2f}")

        # Overall route probability
        route_prob = 1.0
        for city in best_path:
            route_prob *= risk_scores[city]["P(trip_success)"]
        self._t("BAYES", f"Route success probability = {route_prob:.4f}")

        # ─── STEP 4: Decision Making ─────────────────────────
        sub_banner("Step 4 – Decision Module (Alpha-Beta)")
        self._t("DECISION", "Building options from all valid paths …")

        options = self.decision.build_options(valid_paths)
        if options:
            ab_val, ab_best = self.decision.alpha_beta(options, depth=3)
            self._t("DECISION",
                    f"Alpha-Beta chose: {ab_best.label()} | utility={ab_val:.4f}")
            chosen_path = ab_best.route
        else:
            chosen_path = best_path

        # ─── STEP 5: Final Itinerary ──────────────────────────
        sub_banner("Step 5 – Final Itinerary")
        if assignment:
            itinerary = self.csp.build_itinerary(assignment, constraints, chosen_path)
            self.csp.display_itinerary(itinerary, constraints)
        else:
            self._t("ITINERARY", "Skipped (CSP failed); showing raw path")
            info(f"Route   : {format_route(chosen_path)}")
            info(f"Distance: {self.graph.total_distance(chosen_path):.1f} km")
            info(f"Cost    : ₹{self.graph.total_cost(chosen_path):,.0f}")
            info(f"Time    : {format_time(self.graph.total_time(chosen_path))}")

        # ─── STEP 6: Summary Report ───────────────────────────
        self._display_summary(source, destination, best_path, route_prob,
                               ab_val if options else 0.0, constraints, season)
        self._display_trace()
        self._display_performance(time.perf_counter() - t_start, results)
        self._display_ethics()

    # ────────────────────────────────────────────────────────
    #  SUMMARY
    # ────────────────────────────────────────────────────────

    def _display_summary(
        self,
        source: str, destination: str,
        path: List[str], route_prob: float, utility: float,
        constraints: TripConstraints, season: str,
    ) -> None:
        banner("FINAL RECOMMENDATION SUMMARY", 65)
        success(f"Route : {format_route(path)}")
        info(f"Distance          : {self.graph.total_distance(path):.1f} km")
        info(f"Travel Cost       : ₹{self.graph.total_cost(path):,.0f}")
        info(f"Travel Time       : {format_time(self.graph.total_time(path))}")
        info(f"Satisfaction Score: {self.graph.attraction_score(path):.1f} pts")
        info(f"Route Probability : {route_prob:.4f}")
        info(f"Utility Score     : {utility:.4f}")
        info(f"Season            : {season}")
        separator()

    # ────────────────────────────────────────────────────────
    #  XAI TRACE
    # ────────────────────────────────────────────────────────

    def _display_trace(self) -> None:
        sub_banner("Explainable AI – Reasoning Trace")
        for i, entry in enumerate(self._trace, 1):
            print(f"  {C.MAGENTA}{i:>2}.{C.RESET} {entry}")
        separator()

    # ────────────────────────────────────────────────────────
    #  PERFORMANCE ANALYSIS
    # ────────────────────────────────────────────────────────

    def _display_performance(
        self,
        total_sec: float,
        results: Dict[str, SearchResult],
    ) -> None:
        sub_banner("Performance Analysis")
        info(f"Total pipeline time : {total_sec * 1000:.2f} ms")
        print(f"\n  {'Algorithm':<10} {'Nodes':>8} {'Runtime(ms)':>12} {'Distance':>10}")
        print("  " + "─" * 44)
        for algo, r in results.items():
            if r.found:
                print(f"  {algo:<10} {r.nodes_expanded:>8} "
                      f"{r.exec_time_ms:>12.3f} {r.total_distance:>10.1f}")
            else:
                print(f"  {algo:<10}  NOT FOUND")

        sub_banner("Failure Analysis")
        for algo, r in results.items():
            if not r.found:
                print(f"  {C.RED}✘ {algo}:{C.RESET} No path found. "
                      "Cities may not be directly connected.")
        separator()

    # ────────────────────────────────────────────────────────
    #  ETHICS
    # ────────────────────────────────────────────────────────

    def _display_ethics(self) -> None:
        sub_banner("Ethical Considerations & Bias Discussion")
        considerations = [
            ("Heuristic Bias",
             "The Haversine heuristic favours geographically close cities, "
             "potentially under-representing remote cultural heritage sites."),
            ("Data Bias",
             "Popularity scores reflect existing tourist footfall, "
             "which can amplify over-tourism in already-popular spots."),
            ("Economic Fairness",
             "Cost-optimised routes may steer tourists away from local, "
             "smaller businesses toward established chains."),
            ("Uncertainty Handling",
             "Bayesian estimates rely on historical probabilities; "
             "rare events (floods, festivals) are not captured."),
            ("Transparency",
             "All AI decisions are logged in the reasoning trace above "
             "so users can audit and override recommendations."),
            ("Sustainability",
             "Future versions should penalise carbon-heavy routes "
             "and incentivise eco-friendly transport."),
        ]
        for topic, detail in considerations:
            print(f"\n  {C.YELLOW}▶ {topic}{C.RESET}")
            print(f"    {detail}")
        separator()
