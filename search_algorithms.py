# ============================================================
# search_algorithms.py – BFS, DFS, UCS, Greedy, A* (CO2)
# Each algorithm tracks: path, distance, nodes expanded, time
# ============================================================

from __future__ import annotations
import heapq
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from graph import TouristGraph, SearchNode
from utils import (C, banner, sub_banner, info, ai_reason, warn,
                   separator, Timer, format_route, format_time)

# ── Result container ─────────────────────────────────────────

@dataclass
class SearchResult:
    algorithm:      str
    path:           List[str]
    total_distance: float          # km
    total_cost:     float          # INR
    total_time:     float          # hours
    nodes_expanded: int
    exec_time_ms:   float
    found:          bool = True

    def display(self) -> None:
        if not self.found or not self.path:
            print(f"  {C.RED}No path found.{C.RESET}")
            return
        sub_banner(f"Result – {self.algorithm}")
        print(f"  Route    : {format_route(self.path)}")
        info(f"Distance : {self.total_distance:.1f} km")
        info(f"Cost     : ₹{self.total_cost:,.0f}")
        info(f"Time     : {format_time(self.total_time)}")
        info(f"Nodes    : {self.nodes_expanded} expanded")
        info(f"Runtime  : {self.exec_time_ms:.3f} ms")
        separator()


# ════════════════════════════════════════════════════════════
#  SEARCH ALGORITHMS CLASS
# ════════════════════════════════════════════════════════════

class SearchAlgorithms:
    """
    Houses BFS, DFS, UCS, Greedy BFS, and A*.

    All algorithms share the same graph and return a SearchResult.
    Open / closed list management is shown in each method.
    Tie-breaking in A* uses a counter as a secondary key.
    """

    def __init__(self, graph: TouristGraph):
        self.graph = graph

    # ── Helper ───────────────────────────────────────────────

    def _no_path(self, algo: str, ne: int, ms: float) -> SearchResult:
        return SearchResult(algo, [], 0, 0, 0, ne, ms, found=False)

    # ────────────────────────────────────────────────────────
    #  BFS – Breadth-First Search
    # ────────────────────────────────────────────────────────
    def bfs(self, start: str, goal: str) -> SearchResult:
        """
        BFS explores all nodes at depth d before depth d+1.
        Guarantees shortest path in terms of hops (not distance).
        Uses a FIFO queue as the open list.
        """
        ai_reason("BFS", f"Start={start}, Goal={goal}")
        nodes_expanded = 0

        with Timer() as t:
            # Open list: queue of (path_so_far)
            queue: deque[List[str]] = deque([[start]])
            # Closed list: visited cities
            closed: set = {start}

            while queue:
                path = queue.popleft()
                current = path[-1]
                nodes_expanded += 1

                if current == goal:
                    dist = self.graph.total_distance(path)
                    cost = self.graph.total_cost(path)
                    time = self.graph.total_time(path)
                    return SearchResult("BFS", path, dist, cost, time,
                                        nodes_expanded, t.elapsed)

                for neighbour, *_ in self.graph.neighbours(current):
                    if neighbour not in closed:
                        closed.add(neighbour)
                        queue.append(path + [neighbour])

        return self._no_path("BFS", nodes_expanded, t.elapsed)

    # ────────────────────────────────────────────────────────
    #  DFS – Depth-First Search
    # ────────────────────────────────────────────────────────
    def dfs(self, start: str, goal: str) -> SearchResult:
        """
        DFS explores as deep as possible before backtracking.
        Uses a LIFO stack; may not find the optimal route.
        """
        ai_reason("DFS", f"Start={start}, Goal={goal}")
        nodes_expanded = 0

        with Timer() as t:
            # Open list: LIFO stack of (path)
            stack: List[List[str]] = [[start]]
            closed: set = set()

            while stack:
                path = stack.pop()
                current = path[-1]

                if current in closed:
                    continue
                closed.add(current)
                nodes_expanded += 1

                if current == goal:
                    dist = self.graph.total_distance(path)
                    cost = self.graph.total_cost(path)
                    time = self.graph.total_time(path)
                    return SearchResult("DFS", path, dist, cost, time,
                                        nodes_expanded, t.elapsed)

                for neighbour, *_ in reversed(self.graph.neighbours(current)):
                    if neighbour not in closed:
                        stack.append(path + [neighbour])

        return self._no_path("DFS", nodes_expanded, t.elapsed)

    # ────────────────────────────────────────────────────────
    #  UCS – Uniform Cost Search
    # ────────────────────────────────────────────────────────
    def ucs(self, start: str, goal: str) -> SearchResult:
        """
        UCS expands the lowest-cost node first.
        Optimal for weighted graphs. Uses a min-heap (priority queue).
        """
        ai_reason("UCS", f"Start={start}, Goal={goal}")
        nodes_expanded = 0
        counter = 0  # tie-breaker

        with Timer() as t:
            # Heap: (cumulative_distance, counter, path)
            heap: List[Tuple] = [(0.0, counter, [start])]
            # cost_so_far tracks best distance to each city
            cost_so_far: Dict[str, float] = {start: 0.0}

            while heap:
                g, _, path = heapq.heappop(heap)
                current = path[-1]
                nodes_expanded += 1

                if current == goal:
                    cost = self.graph.total_cost(path)
                    time = self.graph.total_time(path)
                    return SearchResult("UCS", path, g, cost, time,
                                        nodes_expanded, t.elapsed)

                for nb, dist, *_ in self.graph.neighbours(current):
                    new_g = g + dist
                    if nb not in cost_so_far or new_g < cost_so_far[nb]:
                        cost_so_far[nb] = new_g
                        counter += 1
                        heapq.heappush(heap, (new_g, counter, path + [nb]))

        return self._no_path("UCS", nodes_expanded, t.elapsed)

    # ────────────────────────────────────────────────────────
    #  GREEDY Best-First Search
    # ────────────────────────────────────────────────────────
    def greedy(self, start: str, goal: str) -> SearchResult:
        """
        Greedy BFS always expands the node with smallest h(n).
        Fast but not guaranteed optimal – can miss shorter routes.
        Heuristic: Haversine straight-line distance to goal.
        """
        ai_reason("GREEDY", f"Start={start}, Goal={goal}")
        nodes_expanded = 0
        counter = 0

        with Timer() as t:
            # Heap: (h_cost, counter, g_cost, path)
            h0 = self.graph.heuristic(start, goal)
            heap: List[Tuple] = [(h0, counter, 0.0, [start])]
            closed: set = set()

            while heap:
                h, _, g, path = heapq.heappop(heap)
                current = path[-1]

                if current in closed:
                    continue
                closed.add(current)
                nodes_expanded += 1

                if current == goal:
                    dist = self.graph.total_distance(path)
                    cost = self.graph.total_cost(path)
                    time = self.graph.total_time(path)
                    return SearchResult("Greedy", path, dist, cost, time,
                                        nodes_expanded, t.elapsed)

                for nb, edge_dist, *_ in self.graph.neighbours(current):
                    if nb not in closed:
                        h_nb = self.graph.heuristic(nb, goal)
                        counter += 1
                        heapq.heappush(heap, (h_nb, counter, g + edge_dist,
                                              path + [nb]))

        return self._no_path("Greedy", nodes_expanded, t.elapsed)

    # ────────────────────────────────────────────────────────
    #  A* Search
    # ────────────────────────────────────────────────────────
    def astar(self, start: str, goal: str) -> SearchResult:
        """
        A* uses f(n) = g(n) + h(n).
        g(n): actual cost from start.
        h(n): admissible Haversine heuristic to goal.
        Optimal and complete on finite graphs with consistent heuristic.
        Tie-breaking: prefer nodes with lower g (longer paths = fewer turns).
        """
        ai_reason("A*", f"Start={start}, Goal={goal}")
        nodes_expanded = 0
        counter = 0

        with Timer() as t:
            h0 = self.graph.heuristic(start, goal)
            # Heap: (f, counter, g, path)
            heap: List[Tuple] = [(h0, counter, 0.0, [start])]
            best_g: Dict[str, float] = {start: 0.0}

            while heap:
                f, _, g, path = heapq.heappop(heap)
                current = path[-1]
                nodes_expanded += 1

                if current == goal:
                    cost = self.graph.total_cost(path)
                    time = self.graph.total_time(path)
                    return SearchResult("A*", path, g, cost, time,
                                        nodes_expanded, t.elapsed)

                # Skip if we've already found a better path here
                if g > best_g.get(current, math.inf):
                    continue

                for nb, dist, *_ in self.graph.neighbours(current):
                    new_g = g + dist
                    if new_g < best_g.get(nb, math.inf):
                        best_g[nb] = new_g
                        h_nb = self.graph.heuristic(nb, goal)
                        f_nb = new_g + h_nb
                        counter += 1
                        heapq.heappush(heap, (f_nb, counter, new_g, path + [nb]))

        return self._no_path("A*", nodes_expanded, t.elapsed)

    # ────────────────────────────────────────────────────────
    #  Compare all algorithms
    # ────────────────────────────────────────────────────────
    def compare_all(self, start: str, goal: str) -> None:
        banner(f"ALGORITHM COMPARISON  {start} → {goal}", 65)

        algorithms = [
            ("BFS",    self.bfs),
            ("DFS",    self.dfs),
            ("UCS",    self.ucs),
            ("Greedy", self.greedy),
            ("A*",     self.astar),
        ]

        results = []
        for name, fn in algorithms:
            ai_reason("RUN", f"Running {name} …")
            r = fn(start, goal)
            results.append(r)
            r.display()

        # ── Summary table ────────────────────────────────────
        sub_banner("Summary Comparison Table")
        print(f"\n  {'Algorithm':<12} {'Dist(km)':>10} {'Cost(₹)':>10} "
              f"{'Time':>8} {'Nodes':>7} {'Runtime(ms)':>12}")
        print("  " + "─" * 63)
        for r in results:
            if r.found:
                print(f"  {r.algorithm:<12} {r.total_distance:>10.1f} "
                      f"{r.total_cost:>10,.0f} {format_time(r.total_time):>8} "
                      f"{r.nodes_expanded:>7} {r.exec_time_ms:>12.3f}")
            else:
                print(f"  {r.algorithm:<12} {'NOT FOUND':>53}")
        separator()

        # Identify best
        valid = [r for r in results if r.found]
        if valid:
            best_dist = min(valid, key=lambda r: r.total_distance)
            best_cost = min(valid, key=lambda r: r.total_cost)
            fastest   = min(valid, key=lambda r: r.exec_time_ms)
            print(f"\n  {C.GREEN}Best route (distance):{C.RESET} {best_dist.algorithm}")
            print(f"  {C.GREEN}Best route (cost)    :{C.RESET} {best_cost.algorithm}")
            print(f"  {C.GREEN}Fastest algorithm    :{C.RESET} {fastest.algorithm} "
                  f"({fastest.exec_time_ms:.3f} ms)")
        separator()

    # ────────────────────────────────────────────────────────
    #  Single algorithm menu runner
    # ────────────────────────────────────────────────────────
    def run_single(self, algo_name: str, start: str, goal: str) -> SearchResult:
        mapping = {
            "bfs":    self.bfs,
            "dfs":    self.dfs,
            "ucs":    self.ucs,
            "greedy": self.greedy,
            "a*":     self.astar,
            "astar":  self.astar,
        }
        fn = mapping.get(algo_name.lower())
        if fn is None:
            warn(f"Unknown algorithm: {algo_name}")
            return self._no_path(algo_name, 0, 0)
        banner(f"{algo_name.upper()} Search  {start} → {goal}", 60)
        result = fn(start, goal)
        result.display()
        return result
