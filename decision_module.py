# ============================================================
# decision_module.py – Decision Making & Game Search (CO4)
# Minimax, Alpha-Beta Pruning, Utility, Expectimax overview
# ============================================================

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

from graph import TouristGraph
from utils import (C, banner, sub_banner, info, ai_reason, separator,
                   warn, success)


# ════════════════════════════════════════════════════════════
#  TRAVEL OPTION (leaf node in game tree)
# ════════════════════════════════════════════════════════════

@dataclass
class TravelOption:
    """
    Represents a travel choice the planner (MAX player) evaluates.
    The adversary (MIN player) models bad conditions: weather, crowd.
    """
    route:        List[str]
    distance:     float   = 0.0
    total_cost:   float   = 0.0
    travel_time:  float   = 0.0
    satisfaction: float   = 0.0
    weather_risk: float   = 0.0   # 0 = no risk, 1 = max risk
    crowd_factor: float   = 0.0   # 0 = empty, 1 = very crowded

    def label(self) -> str:
        return " → ".join(self.route)


# ════════════════════════════════════════════════════════════
#  DECISION MODULE
# ════════════════════════════════════════════════════════════

class DecisionModule:
    """
    Models route selection as a two-player game:
      MAX player : Tourist planner (maximises satisfaction - cost)
      MIN player : Nature / conditions (minimises utility via weather/crowd)
    """

    def __init__(self, graph: TouristGraph):
        self.graph = graph
        self.nodes_evaluated = 0

    # ── Utility / Evaluation Function ───────────────────────

    def utility(self, option: TravelOption) -> float:
        """
        Composite utility function:
          + satisfaction score (weighted)
          - normalised cost
          - normalised distance
          - weather risk penalty
          - crowd factor penalty
        All components scaled to [-10, +10] range before summing.
        """
        sat_score   =  option.satisfaction * 2.0
        cost_pen    = -(option.total_cost   / 1000.0)
        dist_pen    = -(option.distance     / 100.0)
        weather_pen = -(option.weather_risk * 5.0)
        crowd_pen   = -(option.crowd_factor * 3.0)
        return round(sat_score + cost_pen + dist_pen + weather_pen + crowd_pen, 4)

    # ── Build game tree from travel options ──────────────────

    def _build_tree(
        self,
        options: List[TravelOption],
        depth:   int,
    ) -> List[List[TravelOption]]:
        """
        Expand options into a binary game tree of given depth.
        Even levels = MAX (planner), Odd levels = MIN (conditions).
        Returns a flat list of levels.
        """
        tree = [options]
        current_level = options
        for d in range(1, depth + 1):
            next_level = []
            for opt in current_level:
                # Each node spawns two children with slight variations
                child_a = TravelOption(
                    route=opt.route,
                    distance=opt.distance,
                    total_cost=opt.total_cost * random.uniform(0.9, 1.0),
                    travel_time=opt.travel_time,
                    satisfaction=opt.satisfaction * random.uniform(0.85, 1.0),
                    weather_risk=opt.weather_risk * random.uniform(0.8, 1.2),
                    crowd_factor=opt.crowd_factor * random.uniform(0.9, 1.1),
                )
                child_b = TravelOption(
                    route=opt.route,
                    distance=opt.distance * 1.1,
                    total_cost=opt.total_cost * random.uniform(1.0, 1.2),
                    travel_time=opt.travel_time * 1.1,
                    satisfaction=opt.satisfaction * random.uniform(0.7, 0.95),
                    weather_risk=min(1.0, opt.weather_risk * random.uniform(1.0, 1.5)),
                    crowd_factor=min(1.0, opt.crowd_factor * random.uniform(1.0, 1.3)),
                )
                next_level.extend([child_a, child_b])
            tree.append(next_level)
            current_level = next_level
        return tree

    # ── Minimax ──────────────────────────────────────────────

    def minimax(
        self,
        options: List[TravelOption],
        depth:   int,
        maximising: bool = True,
    ) -> Tuple[float, TravelOption]:
        """
        Minimax search over travel options.
        Returns (best_utility, best_option).
        """
        self.nodes_evaluated += len(options)

        if depth == 0 or not options:
            best = max(options, key=self.utility)
            return self.utility(best), best

        if maximising:
            best_util  = -math.inf
            best_opt   = options[0]
            for opt in options:
                children = self._expand(opt)
                val, _ = self.minimax(children, depth - 1, False)
                if val > best_util:
                    best_util = val
                    best_opt  = opt
            return best_util, best_opt
        else:
            best_util  = math.inf
            best_opt   = options[0]
            for opt in options:
                children = self._expand(opt)
                val, _ = self.minimax(children, depth - 1, True)
                if val < best_util:
                    best_util = val
                    best_opt  = opt
            return best_util, best_opt

    def _expand(self, opt: TravelOption) -> List[TravelOption]:
        """Generate two children with slight variations (condition changes)."""
        return [
            TravelOption(
                route=opt.route,
                distance=opt.distance,
                total_cost=opt.total_cost,
                satisfaction=opt.satisfaction * 0.95,
                weather_risk=min(1.0, opt.weather_risk * 1.1),
                crowd_factor=min(1.0, opt.crowd_factor * 1.0),
            ),
            TravelOption(
                route=opt.route,
                distance=opt.distance,
                total_cost=opt.total_cost * 1.1,
                satisfaction=opt.satisfaction * 0.85,
                weather_risk=min(1.0, opt.weather_risk * 1.3),
                crowd_factor=min(1.0, opt.crowd_factor * 1.2),
            ),
        ]

    # ── Alpha-Beta Pruning ───────────────────────────────────

    def alpha_beta(
        self,
        options:    List[TravelOption],
        depth:      int,
        alpha:      float = -math.inf,
        beta:       float =  math.inf,
        maximising: bool  = True,
    ) -> Tuple[float, TravelOption]:
        """
        Alpha-Beta Pruning on the minimax tree.
        α = best MAX can guarantee; β = best MIN can guarantee.
        Prunes branches where β ≤ α (no point continuing).
        """
        self.nodes_evaluated += 1

        if depth == 0 or not options:
            best = max(options, key=self.utility) if options else options[0]
            return self.utility(best), best

        best_opt = options[0]

        if maximising:
            val = -math.inf
            for opt in options:
                children    = self._expand(opt)
                child_val, _ = self.alpha_beta(children, depth - 1, alpha, beta, False)
                if child_val > val:
                    val      = child_val
                    best_opt = opt
                alpha = max(alpha, val)
                if beta <= alpha:
                    ai_reason("PRUNE", f"β({beta:.2f}) ≤ α({alpha:.2f}) – pruning {len(options)} branches")
                    break
        else:
            val = math.inf
            for opt in options:
                children    = self._expand(opt)
                child_val, _ = self.alpha_beta(children, depth - 1, alpha, beta, True)
                if child_val < val:
                    val      = child_val
                    best_opt = opt
                beta = min(beta, val)
                if beta <= alpha:
                    ai_reason("PRUNE", f"β({beta:.2f}) ≤ α({alpha:.2f}) – pruning {len(options)} branches")
                    break

        return val, best_opt

    # ── Expectimax ──────────────────────────────────────────

    def expectimax(
        self,
        options: List[TravelOption],
        depth:   int,
        maximising: bool = True,
    ) -> Tuple[float, TravelOption]:
        """
        Expectimax: chance nodes (nature) take expected value
        instead of minimax's worst-case assumption.
        Better models stochastic environments.
        """
        self.nodes_evaluated += len(options)
        if depth == 0 or not options:
            best = max(options, key=self.utility)
            return self.utility(best), best

        if maximising:
            best_util = -math.inf
            best_opt  = options[0]
            for opt in options:
                children  = self._expand(opt)
                val, _    = self.expectimax(children, depth - 1, False)
                if val > best_util:
                    best_util = val
                    best_opt  = opt
            return best_util, best_opt
        else:
            # Chance node: expected value over children
            children = []
            for opt in options:
                children.extend(self._expand(opt))
            if not children:
                return self.utility(options[0]), options[0]
            exp_val = sum(self.utility(c) for c in children) / len(children)
            return exp_val, options[0]

    # ── Build Options from Search Results ────────────────────

    def build_options(self, paths: List[List[str]]) -> List[TravelOption]:
        """
        Convert a list of route paths into TravelOption objects,
        pulling real distances, costs, and satisfaction from the graph.
        """
        from dataset import CITY_PROBS
        options = []
        for path in paths:
            dist  = self.graph.total_distance(path)
            cost  = self.graph.total_cost(path)
            time  = self.graph.total_time(path)
            sat   = self.graph.attraction_score(path)

            # Aggregate weather/crowd risk over path
            weather_risks = [1 - CITY_PROBS.get(c, {}).get("good_weather", 0.6) for c in path]
            crowd_factors = [CITY_PROBS.get(c, {}).get("heavy_crowd", 0.5) for c in path]
            avg_weather   = sum(weather_risks) / len(weather_risks) if weather_risks else 0.5
            avg_crowd     = sum(crowd_factors) / len(crowd_factors) if crowd_factors else 0.5

            options.append(TravelOption(
                route=path, distance=dist, total_cost=cost,
                travel_time=time, satisfaction=sat,
                weather_risk=avg_weather, crowd_factor=avg_crowd,
            ))
        return options

    # ── Main run ─────────────────────────────────────────────

    def run(self, paths: List[List[str]], depth: int = 3) -> None:
        banner("DECISION MODULE – Minimax & Alpha-Beta", 65)
        if not paths:
            warn("No paths to evaluate.")
            return

        options = self.build_options(paths)
        sub_banner("Travel Options")
        for i, opt in enumerate(options, 1):
            u = self.utility(opt)
            print(f"  {i}. {opt.label():<40} "
                  f"dist={opt.distance:.0f}km  cost=₹{opt.total_cost:,.0f}  "
                  f"utility={C.YELLOW}{u:.3f}{C.RESET}")

        # ── Minimax ──────────────────────────────────────────
        sub_banner("Minimax Search")
        self.nodes_evaluated = 0
        ai_reason("MINIMAX", f"depth={depth}, options={len(options)}")
        mm_val, mm_best = self.minimax(options, depth)
        info(f"Best option : {mm_best.label()}")
        info(f"Utility     : {mm_val:.4f}")
        info(f"Nodes eval  : {self.nodes_evaluated}")

        # ── Alpha-Beta ───────────────────────────────────────
        sub_banner("Alpha-Beta Pruning")
        self.nodes_evaluated = 0
        ai_reason("AB", f"depth={depth}, α=-∞, β=+∞")
        ab_val, ab_best = self.alpha_beta(options, depth)
        info(f"Best option : {ab_best.label()}")
        info(f"Utility     : {ab_val:.4f}")
        info(f"Nodes eval  : {self.nodes_evaluated}")

        # ── Expectimax ───────────────────────────────────────
        sub_banner("Expectimax (stochastic environment)")
        self.nodes_evaluated = 0
        em_val, em_best = self.expectimax(options, depth)
        info(f"Best option : {em_best.label()}")
        info(f"Utility     : {em_val:.4f}")
        info(f"Nodes eval  : {self.nodes_evaluated}")

        # ── Final decision ───────────────────────────────────
        sub_banner("Final Decision")
        success(f"RECOMMENDED ROUTE : {ab_best.label()}")
        info(f"Utility Score     : {ab_val:.4f}")
        info(f"Distance          : {ab_best.distance:.1f} km")
        info(f"Cost              : ₹{ab_best.total_cost:,.0f}")
        info(f"Satisfaction      : {ab_best.satisfaction:.1f}")
        info(f"Weather Risk      : {ab_best.weather_risk:.2%}")
        ai_reason("DECISION",
                  "Alpha-Beta used because it prunes redundant nodes "
                  "while guaranteeing the same result as full Minimax.")
        separator()
